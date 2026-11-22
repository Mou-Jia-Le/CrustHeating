"""Reproduce manuscript Figures 1-9 from the curated dataset."""
import sys
from .paths import PROJECT_ROOT as ROOT, PLOT_DATA

def generate_main_figures(numbers=None):
    """Generate all main figures, or only the requested integer figure numbers."""
    selected = set(range(1, 10) if numbers is None else numbers)
    if not selected or not selected <= set(range(1, 10)):
        raise ValueError("Main figure numbers must be between 1 and 9.")
    import matplotlib
    matplotlib.use("Agg")

    # Refresh before importing thermal modules, which load calibration at import time.
    from crustal_heating.data_processing import calibrate_plot_data
    calibrate_plot_data()

    from crustal_heating.plots import (
        plot_element_variation,
        plot_heat_generation_vs_thickness,
        plot_temperature_profile,
        plot_current_basal_temperature,
        plot_differentiated_geotherms,
        plot_basal_temperature_by_geologic_time,
        plot_basal_temperature_vs_time,
    )
    from .retention_figure import main as plot_retention
    from .upper_crust_figure import main as plot_upper_crust_thickness

    jobs = [
        plot_element_variation,
        plot_heat_generation_vs_thickness,
        lambda: plot_upper_crust_thickness([]),
        plot_temperature_profile,
        plot_current_basal_temperature,
        lambda: plot_retention([]),
        plot_differentiated_geotherms,
        plot_basal_temperature_by_geologic_time,
        plot_basal_temperature_vs_time,
    ]
    print(f"Python: {sys.executable}", flush=True)
    print(f"Data: {PLOT_DATA}", flush=True)
    for number, job in enumerate(jobs, start=1):
        if number not in selected:
            continue
        print(f"Generating Figure {number} ...", flush=True)
        output = job()
        if output is not None:
            print(f"Saved: {output}", flush=True)
    print(f"Done. Requested figures saved as PNG in {ROOT / 'figures'}", flush=True)


def generate_figure(number):
    """Generate one main or supplementary figure by its manuscript label."""
    label = str(number).upper()
    if label in {str(n) for n in range(1, 10)}:
        generate_main_figures([int(label)])
        return
    if label not in {"S1", "S2"}:
        raise ValueError("Choose a figure from 1-9, S1 or S2.")

    from .appendix import (
        ELEMENTS, analyse_element, make_figure, make_screening_figure, TABLES_DIR,
    )
    from .data_processing import load_analysis_dataset
    data, audit = load_analysis_dataset()
    if label == "S1":
        results = {meta["name"]: analyse_element(data, column)
                   for column, meta in ELEMENTS.items()}
        output = make_figure(results, data)
    else:
        output = make_screening_figure(audit)
        print(f"Saved: {TABLES_DIR / 'TableS1_screening_statistics.csv'}")
    print(f"Saved: {output}")
