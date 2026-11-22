# Quaternary arc magmatism data

| File | Contents |
| --- | --- |
| `quaternary_arc_magmatism.csv` | 29,811 eligible samples before externally studentized-residual screening |
| `quaternary_arc_magmatism_screened.csv` | The same samples after masking 14 Th and 33 U measurements |

The pre-screening population already satisfies the adopted age, lithology,
major-element-total and silica criteria. Its positive K2O–Th and K2O–U pairs
number 14,987 and 11,371, respectively. Screening leaves 14,973 Th and 11,338 U
pairs without removing sample rows or changing unrelated measurements.

Concentration units are included in column names, for example `K2O(WT%)`,
`TH(PPM)` and `U(PPM)` contain analysis values; flagged measurements remain blank
in these columns. The screened file includes two annotation columns immediately
after each element:

| Annotation column | Meaning |
| --- | --- |
| `Th_excluded_by_screening` | `True` if this sample's Th measurement was excluded by the statistical screen |
| `Th_excluded_value_ppm` | Original excluded Th value in ppm; blank otherwise |
| `U_excluded_by_screening` | `True` if this sample's U measurement was excluded by the statistical screen |
| `U_excluded_value_ppm` | Original excluded U value in ppm; blank otherwise |

Filter either flag column to `True` to inspect the exclusions. There are 11
Th-only exclusions, 30 U-only exclusions and 3 samples with both values excluded
(44 distinct samples). `False` means the value was not excluded by this screen;
it does not imply a measurement was available. A blank analysis value with a
`False` flag is not a studentized-screening exclusion. Excluded-value columns are
for provenance only and are never used in regressions or plots.

CSV does not retain font or cell colors; these explicit columns remain readable
in both spreadsheet software and analysis code.

`source_excel_row` preserves the row identifier from the source compilation;
source and citation fields retain analytical provenance.

The selected ages are Pleistocene, Holocene or Quaternary labels. Some ages
refer to the activity range of the source volcano, rather than individual sample
dates. The repository reproduces statistical screening and figures from this
prepared population; it does not reconstruct upstream age assignments from raw
GEOROC exports.

`crustal-heating calibrate` regenerates the screened table and
`config/regression_calibration.json`. Screening diagnostics used in Figure S2
are calculated in memory rather than saved as additional sample datasets.
