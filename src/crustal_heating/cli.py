"""Command-line interface for the crustal heating project."""

from __future__ import annotations

import argparse

from .data_processing import calibrate_plot_data



def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Crustal heating analysis commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("all", help="Generate Figures 1-9, S1-S2 and supplementary tables.")
    subparsers.add_parser("appendix", help="Generate Figures S1-S2 and supplementary tables.")
    subparsers.add_parser("heat-generation", help="Generate Figure 2.")
    subparsers.add_parser("calibrate", help="Recompute regression_calibration.json from the figure dataset.")
    subparsers.add_parser("element-variation", help="Generate Figure 1.")
    subparsers.add_parser("thermal-profile", help="Generate Figure 4.")
    subparsers.add_parser("thermal-models", help="Generate the main thermal-model plots.")
    figure = subparsers.add_parser("figure", help="Generate a single figure by number.")
    figure.add_argument("number", type=str.upper,
                        choices=[str(n) for n in range(1, 10)] + ["S1", "S2"])
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a project command."""

    args = build_parser().parse_args(argv)
    if args.command == "figure":
        from .workflow import generate_figure
        generate_figure(args.number)
        return 0
    if args.command == "appendix":
        from .appendix import main as appendix_main
        appendix_main()
        return 0
    if args.command == "calibrate":
        calibrate_plot_data()
        return 0
    if args.command == "all":
        from .workflow import generate_main_figures
        generate_main_figures()
        from .appendix import main as appendix_main
        appendix_main()
        return 0

    from .plots import (
        plot_all_thermal_models,
        plot_element_variation,
        plot_heat_generation_vs_thickness,
        plot_temperature_profile,
    )

    if args.command == "heat-generation":
        print(f"Saved: {plot_heat_generation_vs_thickness()}")
    elif args.command == "element-variation":
        print(f"Saved: {plot_element_variation()}")
    elif args.command == "thermal-profile":
        print(f"Saved: {plot_temperature_profile()}")
    elif args.command == "thermal-models":
        for path in plot_all_thermal_models():
            print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
