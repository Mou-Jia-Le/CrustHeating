# Crustal heating

Reproduce the geochemical and thermal-model figures for crustal differentiation
and radiogenic heating. This repository contains the two sample datasets,
analysis code, Figures 1–9 and S1–S2, and Table S1.

## Installation

Use Python 3.9 or newer from a checkout of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, use `python` instead of `python3` when creating the environment.

Keep the repository checkout: the package resolves data and outputs relative to
its root. The data are supplied separately from the Python package.

## Reproduce all figures

```bash
python run_all.py
```

This command applies the studentized-residual screening to the supplied
pre-screening data, refreshes the screened table and thermal calibration, and
writes 11 PNG figures to `figures/` and Table S1 to `tables/`.
The supplied figures and table are reference outputs and are overwritten on rerun.

## Reproduce a single figure

```bash
crustal-heating figure 5   # Figure 5 only
crustal-heating figure S1  # Figure S1 only
crustal-heating figure S2  # Figure S2 and its companion Table S1
```

Choose any number from `1` through `9`, or `S1` / `S2` (case-insensitive).
Only the requested PNG in `figures/` is overwritten. Main-figure commands also
refresh the screened dataset and thermal calibration. Figure S2 also refreshes
Table S1. You can equivalently use `python -m crustal_heating figure 5`.

Additional commands:

```bash
crustal-heating appendix   # Figures S1–S2 and Table S1
crustal-heating calibrate  # Screened data and model calibration
crustal-heating --help     # Individual figure commands
```

## Data and methods

- `data/quaternary_arc_magmatism.csv`: 29,811 eligible samples before statistical screening.
- `data/quaternary_arc_magmatism_screened.csv`: the same rows after screening.


For each element separately, the code fits log10(Th) or log10(U) against
log10(K2O), computes externally studentized residuals, and applies a two-sided
Bonferroni threshold with family alpha = 0.05. Screening is applied once and
masks only the flagged element values: 14 Th and 33 U measurements. The remaining
14,973 Th and 11,338 U measurements determine the through-origin regressions used
in the main figures and thermal calibration. No sample rows are removed by this
statistical screen. The screened CSV records each element's exclusion flag and
original excluded value next to its analysis column, distinguishing removed
measurements from pre-existing missing data. See [data details](data/README.md).

## Figure-to-code guide

All modules are under `src/crustal_heating/`.

| Output | Generator |
| --- | --- |
| Figure 1: element variation | `plots.plot_element_variation` |
| Figure 2: heat production and crustal thickness | `plots.plot_heat_generation_vs_thickness` |
| Figure 3: enriched upper-layer thickness | `upper_crust_figure.main` |
| Figure 4: heat-source profiles and geotherms | `plots.plot_temperature_profile` |
| Figure 5: present-day basal temperature | `plots.plot_current_basal_temperature` |
| Figure 6: lower-layer HPE retention | `retention_figure.main` |
| Figure 7: differentiated geotherms | `plots.plot_differentiated_geotherms` |
| Figure 8: basal temperature versus thickness by age | `plots.plot_basal_temperature_by_geologic_time` |
| Figure 9: basal temperature through time | `plots.plot_basal_temperature_vs_time` |
| Figure S1: linear and power-law fit comparison | `appendix.make_figure` |
| Figure S2 and Table S1: screening comparison | `appendix.make_screening_figure` |

`statistics.py` implements regressions and screening; `thermal.py` and
`hpe_retention.py` implement heat production and the conductive thermal models.
`config/regression_calibration.json` records the source-data hash and fitted
parameters. Figures S1–S2 use the same retained pairs as Figure 1.

Table S1 reports uncentered R² and concentration-space RMSE for through-origin
linear fits, and centered R² and log-space RMSE for log–log fits. Metrics in
different response spaces are not directly interchangeable.

## Reproduction environment

The workflow has been run with Python 3.9, NumPy 2.0.2,
pandas 2.3.3, SciPy 1.13.1, Matplotlib 3.9.4 and pyMelt 2.3. Rendering can vary
with plotting-library and font versions.

The data retain source and citation fields. Attribute GEOROC and the original
analytical studies when reusing the compilation.
