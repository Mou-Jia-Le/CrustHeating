"""Appendix figure: Th and U versus K2O, fitted with a fractional-crystallization (Rayleigh) model.

Companion to the zero-intercept linear fits in Figure 1 (panels g and h),
        C_i = m * C_K2O.
This script fits the power law that follows from Rayleigh fractional crystallization,
        log10 C_i = a + b * log10 C_K2O ,   b = (D_i - 1) / (D_K - 1),
and compares the two models with the same sample set and a common set of statistics.

Uses the same single-pass Bonferroni studentized-residual selection as Figure 1.
The final appendix contains only two log-log panels: Th and U versus K2O.

Run: python -m crustal_heating appendix
Outputs: Figures S1-S2 in figures/ and Table S1 in tables/.
"""

from pathlib import Path

from .paths import PROJECT_ROOT as ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from crustal_heating.data_processing import load_analysis_dataset
from crustal_heating.paths import FIGURES_DIR
from crustal_heating.statistics import zero_intercept_regression
from crustal_heating.style import save_figure, style_axes

OUTPUT_FIGURE = FIGURES_DIR / "FigureS1_FractionalCrystallization_Th_U_vs_K.png"
TABLES_DIR = ROOT / "tables"

ELEMENTS = {
    "TH(PPM)": {"name": "Th", "label": "Th (ppm)", "xlim": (0, 8), "ylim": (0, 70), "log_ylim": (-1.6, 2.2)},
    "U(PPM)": {"name": "U", "label": "U (ppm)", "xlim": (0, 8), "ylim": (0, 20), "log_ylim": (-2.0, 1.6)},
}
LOG_XLIM = (-1.6, 1.1)
COLOR_POINTS = "#5A5AFF"
COLOR_LINEAR = "black"      # Reference zero-intercept model
COLOR_FC = "#D7191C"        # fractional-crystallization model


# --------------------------------------------------------------------------------------
# Data: identical screening to Figure 1 (plot_element_variation)
# --------------------------------------------------------------------------------------


def common_pairs(df: pd.DataFrame, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Finite, strictly positive (K2O, y) pairs, so both models use exactly the same samples."""
    x = df["K2O(WT%)"].to_numpy(float)
    y = df[y_column].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[ok], y[ok]


# --------------------------------------------------------------------------------------
# Regression helpers
# --------------------------------------------------------------------------------------
def ols_loglog(lx: np.ndarray, ly: np.ndarray) -> dict:
    """Ordinary least squares of ly on lx (both log10)."""
    n = lx.size
    xbar, ybar = lx.mean(), ly.mean()
    sxx = np.sum((lx - xbar) ** 2)
    sxy = np.sum((lx - xbar) * (ly - ybar))
    syy = np.sum((ly - ybar) ** 2)
    b = sxy / sxx
    a = ybar - b * xbar
    sse = syy - b * sxy
    dof = n - 2
    s2 = sse / dof
    se_b = np.sqrt(s2 / sxx)
    se_a = np.sqrt(s2 * (1.0 / n + xbar**2 / sxx))
    tcrit = stats.t.ppf(0.975, dof)
    t_b0 = b / se_b
    t_b1 = (b - 1.0) / se_b
    r = sxy / np.sqrt(sxx * syy)
    r2 = 1.0 - sse / syy
    return {
        "n": int(n), "a": a, "b": b, "se_a": se_a, "se_b": se_b,
        "b_ci95": (b - tcrit * se_b, b + tcrit * se_b),
        "t_b_vs_0": t_b0, "p_b_vs_0": 2 * stats.t.sf(abs(t_b0), dof),
        "t_b_vs_1": t_b1, "p_b_vs_1": 2 * stats.t.sf(abs(t_b1), dof),
        "pearson_r": r, "r2": r2, "r2_adj": 1.0 - (1.0 - r2) * (n - 1) / dof,
        "s_resid": np.sqrt(s2), "sse": sse, "dof": int(dof),
    }


def linear_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    res = y - yhat
    sse = float(np.sum(res**2))
    return {
        "rmse": float(np.sqrt(np.mean(res**2))),
        "mae": float(np.mean(np.abs(res))),
        "bias": float(np.mean(yhat - y)),
        "r2_centered": 1.0 - sse / float(np.sum((y - y.mean()) ** 2)),
        "r2_uncentered": 1.0 - sse / float(np.sum(y**2)),
    }


def log_metrics(ly: np.ndarray, lyhat: np.ndarray, n_params: int) -> dict:
    """Metrics for log10 y, with a Gaussian log-likelihood so AIC/BIC share one response variable."""
    n = ly.size
    res = ly - lyhat
    sse = float(np.sum(res**2))
    loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(sse / n) + 1.0)
    return {
        "rmse_log10": float(np.sqrt(sse / n)),
        "bias_log10": float(np.mean(lyhat - ly)),
        "r2_log10": 1.0 - sse / float(np.sum((ly - ly.mean()) ** 2)),
        "loglik": float(loglik),
        "aic": float(2 * n_params - 2 * loglik),
        "bic": float(n_params * np.log(n) - 2 * loglik),
    }


# --------------------------------------------------------------------------------------
# One element
# --------------------------------------------------------------------------------------
def analyse_element(df: pd.DataFrame, column: str) -> dict:
    x, y = common_pairs(df, column)
    lx, ly = np.log10(x), np.log10(y)

    # Model A: reference linear model (zero intercept) -- reuse the project's own function.
    m, m_se, m_p, m_r2u = zero_intercept_regression(x, y)
    # Model B: fractional crystallization fitted by log-log OLS.
    fc = ols_loglog(lx, ly)

    pred = {
        "A: linear, y = m x (main model)": m * x,
        "B: FC power law, OLS (median)": 10 ** (fc["a"] + fc["b"] * lx),
    }
    n_params = {
        "A: linear, y = m x (main model)": 2, "B: FC power law, OLS (median)": 3,
    }
    comparison = {}
    for name, yhat in pred.items():
        comparison[name] = {**linear_metrics(y, yhat), **log_metrics(ly, np.log10(yhat), n_params[name])}

    return {
        "n": int(x.size),
        "model_A": {"slope_m": m, "se_m": m_se, "p": m_p, "r2_uncentered": m_r2u},
        "model_B_ols": fc,
        "comparison": comparison,
        "data": (x, y),
    }


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def add_log_bins(ax, lx, ly, width=0.15, min_n=50):
    edges = np.arange(np.floor(lx.min() / width) * width, lx.max() + width, width)
    idx = np.digitize(lx, edges)
    cx, my, ey = [], [], []
    for k in np.unique(idx):
        sel = idx == k
        if sel.sum() >= min_n:
            cx.append(0.5 * (edges[k - 1] + edges[k]))
            my.append(ly[sel].mean())
            ey.append(2 * ly[sel].std(ddof=1) / np.sqrt(sel.sum()))
    ax.errorbar(cx, my, yerr=ey, fmt="s-", color="green", mfc="green", mec="black", ms=5, lw=2,
                ecolor="black", capsize=3, zorder=5)


def make_figure(results: dict, df: pd.DataFrame) -> Path:
    """Final appendix: only Th and U log-log panels after studentized screening."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (column, meta), panel_label in zip(axes, ELEMENTS.items(), ["(a)", "(b)"]):
        res = results[meta["name"]]
        x, y = res["data"]
        lx, ly = np.log10(x), np.log10(y)
        ax.scatter(lx, ly, s=12, alpha=0.7, color=COLOR_POINTS,
                   edgecolor="black", linewidths=0.5, rasterized=True)
        add_log_bins(ax, lx, ly)
        xf = np.linspace(LOG_XLIM[0], LOG_XLIM[1], 200)
        A, B = res["model_A"], res["model_B_ols"]
        ax.plot(xf, np.log10(A["slope_m"]) + xf, color=COLOR_LINEAR,
                ls="--", lw=4, label="Linear (main model)")
        ax.plot(xf, B["a"] + B["b"]*xf, color=COLOR_FC, lw=3,
                label="Log-log OLS (comparison)")
        linear = res["comparison"]["A: linear, y = m x (main model)"]
        power = res["comparison"]["B: FC power law, OLS (median)"]
        text = "\n".join([
            rf"$\log_{{10}} {meta['name']} = {B['a']:.3f} + ({B['b']:.3f} \pm {B['se_b']:.3f})\log_{{10}} K_2O$",
            f"n = {res['n']:,}",
            rf"$R^2_{{\mathrm{{log}}}}$: linear {linear['r2_log10']:.3f}; log fit {power['r2_log10']:.3f}",
            f"RMSE (ppm): linear {linear['rmse']:.2f}; log fit {power['rmse']:.2f}",
        ])
        ax.text(.05, .95, text, transform=ax.transAxes, fontsize=10,
                va="top", bbox=dict(facecolor="white", alpha=.6, edgecolor="none"))
        ax.set_xlim(LOG_XLIM)
        ax.set_ylim(meta["log_ylim"])
        ax.set_xlabel("log$_{10}$ K$_2$O (wt%)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"log$_{{10}}$ {meta['label']}", fontsize=11, fontweight="bold")
        ax.text(.95, .95, panel_label, ha="right", va="top", transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        style_axes(ax)
    fig.tight_layout(rect=(0,.03,1,.97))
    return save_figure(fig, OUTPUT_FIGURE, dpi=300, transparent=False)


def make_screening_figure(audit: pd.DataFrame) -> Path:
    """Show all pre-screening pairs, including outliers beyond Figure A1 limits."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    metric_rows = []
    for ax, (column, meta), panel in zip(axes, ELEMENTS.items(), ["(a)", "(b)"]):
        data = audit.loc[audit["element"] == meta["name"]].copy()
        keep = data["keep"].to_numpy(bool)
        lx = np.log10(data["K2O(WT%)"].to_numpy(float))
        ly = np.log10(data[column].to_numpy(float))
        ax.scatter(lx[keep], ly[keep], s=12, color=COLOR_POINTS, alpha=.7,
                   edgecolors="black", linewidths=.5, rasterized=True,
                   label=f"Retained (n = {int(keep.sum()):,})")
        ax.scatter(lx[~keep], ly[~keep], s=36, marker="x", color="red",
                   linewidths=.8, zorder=6,
                   label=f"Excluded (n = {int((~keep).sum()):,})")
        # Data-driven limits display every excluded observation, rather than
        # clipping at the narrower limits used by the fitted-model figure.
        xmin, xmax = lx.min()-.12, lx.max()+.12
        ymin, ymax = ly.min()-.18, ly.max()+.18
        xf = np.linspace(xmin, xmax, 300)
        before = ols_loglog(lx, ly)
        after = ols_loglog(lx[keep], ly[keep])
        # Report each fit on its fitting sample, plus a common-sample comparison
        # in the CSV to separate removal of extreme residuals from refitting.
        for fit_name, fit, evaluation, mask in [
            ("Before", before, "All eligible", np.ones(len(lx), dtype=bool)),
            ("After", after, "Retained", keep),
            ("Before", before, "Retained", keep),
        ]:
            metric = log_metrics(ly[mask], fit["a"] + fit["b"] * lx[mask], 3)
            metric_rows.append(dict(element=meta["name"], model="Log-log OLS", fit=fit_name,
                                    evaluation=evaluation, n=int(mask.sum()),
                                    intercept=fit["a"], slope=fit["b"],
                                    r2_log10=metric["r2_log10"],
                                    rmse_log10_dex=metric["rmse_log10"]))
        x, y = data["K2O(WT%)"].to_numpy(float), data[column].to_numpy(float)
        for name, mask in [("Before", np.ones(len(x), dtype=bool)), ("After", keep)]:
            slope, _, _, r2u = zero_intercept_regression(x[mask], y[mask])
            rmse = float(np.sqrt(np.mean((y[mask] - slope * x[mask])**2)))
            row = dict(element=meta["name"], model="Zero-intercept linear", fit=name,
                       evaluation="All eligible" if name == "Before" else "Retained",
                       n=int(mask.sum()), intercept=0.0, slope=slope,
                       r2_uncentered=r2u, rmse_ppm=rmse)
            metric_rows.append(row)
        ax.plot(xf, before["a"]+before["b"]*xf, color="black", ls="--", lw=3,
                label="OLS before screening", zorder=4)
        ax.plot(xf, after["a"]+after["b"]*xf, color="#0072B2", lw=3,
                label="OLS after screening", zorder=5)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
               xlabel="log$_{10}$ K$_2$O (wt%)", ylabel=f"log$_{{10}}$ {meta['label']}")
        ax.set_xlabel("log$_{10}$ K$_2$O (wt%)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"log$_{{10}}$ {meta['label']}", fontsize=11, fontweight="bold")
        ax.text(.95, .95, panel, transform=ax.transAxes, ha="right", va="top", fontsize=12,
                fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        style_axes(ax)
        # The retained/excluded marker sets partition the complete audit exactly.
        assert keep.sum()+(~keep).sum() == len(data)
        assert ((lx[~keep]>xmin)&(lx[~keep]<xmax)&(ly[~keep]>ymin)&(ly[~keep]<ymax)).all()
    fig.tight_layout(rect=(0, .03, 1, .97))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_rows = [row for row in metric_rows
                  if not (row["fit"] == "Before" and row["evaluation"] == "Retained")]
    pd.DataFrame(table_rows).to_csv(TABLES_DIR / "TableS1_screening_statistics.csv", index=False)
    return save_figure(fig, FIGURES_DIR / "FigureS2_StudentizedResidualFiltering.png",
                       dpi=300, transparent=False)


# --------------------------------------------------------------------------------------
# Output tables
# --------------------------------------------------------------------------------------


def main() -> None:
    """Generate only supplementary Figures S1-S2 and Table S1."""
    data, audit = load_analysis_dataset()
    results = {meta["name"]: analyse_element(data, column)
               for column, meta in ELEMENTS.items()}
    print(f"Saved: {make_figure(results, data)}")
    print(f"Saved: {make_screening_figure(audit)}")
    print(f"Saved: {TABLES_DIR / 'TableS1_screening_statistics.csv'}")


if __name__ == "__main__":
    main()
