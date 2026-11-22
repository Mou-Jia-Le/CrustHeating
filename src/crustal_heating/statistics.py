"""Shared binning, finite-pair selection and zero-intercept regression."""
import numpy as np
import pandas as pd
from scipy.stats import t


def studentized_log_diagnostics(x_values, y_values, alpha=0.05):
    """Single-pass external studentized residuals for positive log-log OLS pairs.

    Bonferroni correction controls family alpha separately for each element.
    Returns positions into the original input, so row identities are preserved.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    x, y = np.asarray(x_values, float), np.asarray(y_values, float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    positions = np.flatnonzero(valid)
    n = len(positions)
    if n < 4:
        raise ValueError("At least four positive pairs are required for studentized screening")
    lx, ly = np.log10(x[valid]), np.log10(y[valid])
    design = np.column_stack([np.ones(n), lx])
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("Studentized screening requires variation in log K2O")
    coefficients = np.linalg.lstsq(design, ly, rcond=None)[0]
    residual = ly - design @ coefficients
    # QR leverage is more numerically stable than explicitly inverting X'X.
    q, _ = np.linalg.qr(design, mode="reduced")
    h = np.sum(q*q, axis=1)
    if np.any(1-h <= np.finfo(float).eps):
        raise ValueError("Studentized residual undefined for unit-leverage observations")
    sse = float(residual @ residual)
    deleted_variance = np.maximum((sse-residual**2/(1-h))/(n-3), 0)
    denominator = np.sqrt(deleted_variance*(1-h))
    if sse <= 100*np.finfo(float).eps**2*max(float(ly @ ly), 1):
        student = np.zeros(n)  # exact power law: no statistical outliers
    else:
        student = np.divide(residual, denominator, out=np.copysign(np.full(n, np.inf), residual), where=denominator>0)
    p = 2*t.sf(np.abs(student), n-3)
    cutoff = float(t.isf(alpha/(2*n), n-3))
    return pd.DataFrame(dict(position=positions, log_residual=residual, leverage=h,
                             external_studentized_residual=student, p_value=p,
                             p_bonferroni=np.minimum(p*n, 1), cutoff=cutoff,
                             keep=np.abs(student)<=cutoff))


def bin_edges(x_column, values):
    """Extend the manuscript bin grid to cover every finite observation."""
    width, origin = (2.0, 47.0) if x_column == "SIO2(WT%)" else ((5.0, 0.0) if x_column == "Mg_number" else (0.5, 0.0))
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.array([origin, origin + width])
    lo = min(origin, origin + np.floor((values.min() - origin) / width) * width)
    hi = origin + (np.floor((values.max() - origin) / width) + 1) * width
    return np.arange(lo, max(hi, lo + width) + width / 2, width)


def select_finite_pairs(df, x_column, y_column):
    """Select usable pairs without clipping extremes or limiting the x range."""
    data = df.copy()
    for name in (x_column, y_column):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    return data.loc[np.isfinite(data[x_column]) & np.isfinite(data[y_column])].copy()


def zero_intercept_regression(x_values, y_values) -> tuple[float, float, float, float]:
    """Figure 1 helper: Fit y = m*x with slope SE, p-value, and uncentered R^2."""

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x != 0)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.dot(x, x) == 0:
        return np.nan, np.nan, np.nan, np.nan

    x_sum_squares = float(np.dot(x, x))
    slope = float(np.dot(x, y) / x_sum_squares)
    residual = y - slope * x
    residual_sum_squares = float(np.dot(residual, residual))
    y_sum_squares = float(np.dot(y, y))
    degrees_of_freedom = x.size - 1
    slope_stderr = float(
        np.sqrt((residual_sum_squares / degrees_of_freedom) / x_sum_squares)
    )
    if slope_stderr == 0:
        p_value = 0.0 if slope != 0 else 1.0
    else:
        t_statistic = slope / slope_stderr
        p_value = float(2 * t.sf(abs(t_statistic), degrees_of_freedom))
    r_squared = np.nan if y_sum_squares == 0 else 1.0 - residual_sum_squares / y_sum_squares
    return slope, slope_stderr, p_value, r_squared
