"""Reproduce studentized screening from the two manuscript sample tables."""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

from .paths import PLOT_DATA, SCREENED_DATA, CALIBRATION_PATH
from .statistics import select_finite_pairs, zero_intercept_regression, studentized_log_diagnostics


def screen_studentized_elements(df: pd.DataFrame):
    """Mask outlying Th/U values independently; keep rows and other elements.

    Screen once on baseline log-log OLS, then refit downstream; never iterate.
    The source workbook remains unchanged. Diagnostics retain its Excel row IDs.
    """
    output = df.copy()
    audits = []
    for label, column in (("Th", "TH(PPM)"), ("U", "U(PPM)")):
        output[column] = pd.to_numeric(output[column], errors="coerce")
        output.loc[output[column].le(0), column] = np.nan
        diagnostic = studentized_log_diagnostics(df["K2O(WT%)"], output[column])
        positions = diagnostic.pop("position").to_numpy(int)
        identity = [c for c in ("source_excel_row", "SAMPLE NAME", "UNIQUE_ID", "Source File", "ROCK NAME", "K2O(WT%)", "TH(PPM)", "U(PPM)") if c in df]
        audit = df.iloc[positions][identity].reset_index(drop=True).copy()
        audit["element"] = label
        audit = pd.concat([audit, diagnostic], axis=1)
        audits.append(audit)
        output.iloc[positions[~diagnostic["keep"].to_numpy()], output.columns.get_loc(column)] = np.nan
    return output, pd.concat(audits, ignore_index=True)


def load_analysis_dataset(path: Path = PLOT_DATA):
    """Screen the supplied pre-screening sample population once.

    Age, lithology and chemical eligibility have already been applied. Preserve
    source_excel_row as the identifier from the archived source workbook.
    """
    data = pd.read_csv(path, low_memory=False, float_precision="round_trip")
    return screen_studentized_elements(data)


def write_regression_calibration(clean: pd.DataFrame, dataset_path: Path,
                                 calibration_path: Path | None = None) -> Path:
    """Fit zero-intercept Th-K2O and U-K2O regressions and write regression_calibration.json."""

    calibration_path = calibration_path or CALIBRATION_PATH
    fits = {}
    for label, name in (("Th", "TH(PPM)"), ("U", "U(PPM)")):
        filtered = select_finite_pairs(clean, "K2O(WT%)", name)
        slope, se, p_value, r2 = zero_intercept_regression(filtered["K2O(WT%)"], filtered[name])
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError(f"Insufficient valid data for {label} calibration")
        fits[label] = dict(slope=slope, standard_error=se, p_value=p_value,
                           r_squared_uncentered=r2, n=int((filtered["K2O(WT%)"] != 0).sum()),
                           n_valid_pairs=int(clean[name].notna().sum()))
    calibration = dict(schema_version=1, k2o_slope_wt_per_km=0.0423, k2o_intercept_wt=0.17,
                       filter="47 <= SiO2 <= 72 wt%; single-pass external studentized residuals from log10 element vs log10 K2O OLS; two-sided Bonferroni alpha=0.05 per element; mask only the flagged element value; refit zero-intercept regression; no U concentration or Th/U ratio cutoff",
                       dataset=dataset_path.name,
                       dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                       sample_count=len(clean), rock_types=clean["ROCK TYPE"].value_counts().to_dict(),
                       fits=fits)
    calibration_path.write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return calibration_path


def calibrate_plot_data(dataset_path: Path = PLOT_DATA) -> Path:
    """Refresh the post-screening table and model calibration only."""
    clean, audit = load_analysis_dataset(dataset_path)
    exported = clean.copy()
    # Keep excluded measurements visible without returning them to the analysis.
    # Flags describe this statistical screen only, not pre-existing missing data.
    for label, column in (("Th", "TH(PPM)"), ("U", "U(PPM)")):
        excluded = audit.loc[(audit["element"] == label) & ~audit["keep"]]
        original_values = excluded.set_index("source_excel_row")[column]
        location = exported.columns.get_loc(column) + 1
        exported.insert(location, f"{label}_excluded_by_screening",
                        exported["source_excel_row"].isin(original_values.index))
        exported.insert(location + 1, f"{label}_excluded_value_ppm",
                        exported["source_excel_row"].map(original_values))
    exported.to_csv(SCREENED_DATA, index=False)
    path = write_regression_calibration(clean, dataset_path)
    print(f"Saved screening and calibration (n={len(clean)}): {path}")
    return path
