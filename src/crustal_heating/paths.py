"""Repository paths for the two manuscript datasets and derived figures."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DATA = DATA_DIR / "quaternary_arc_magmatism.csv"
SCREENED_DATA = DATA_DIR / "quaternary_arc_magmatism_screened.csv"
CALIBRATION_PATH = PROJECT_ROOT / "config" / "regression_calibration.json"
FIGURES_DIR = PROJECT_ROOT / "figures"
