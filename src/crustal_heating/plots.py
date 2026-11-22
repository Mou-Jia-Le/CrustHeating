"""Main-text figure builders for the crustal heating manuscript."""

from .style import model_colormap, add_model_colorbar

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data_processing import load_analysis_dataset
from .statistics import bin_edges, select_finite_pairs, zero_intercept_regression

from .hpe_retention import geotherm
from .paths import PLOT_DATA, FIGURES_DIR
from .solidus import get_eclogite_solidus
from .style import (
    COLOR_DIFF,
    COLOR_EROSION,
    COLOR_K,
    COLOR_TH,
    COLOR_U,
    COLOR_UNDIFF,
    label_lines_or_legend,
    save_figure,
    style_axes,
)
from .thermal import (
    DEFAULT_CONSTANTS,
    TEN_KM_CRUST_HEAT_PRODUCTION_W_M3,
    basal_temperature_differentiated,
    basal_temperature_undifferentiated,
    build_heat_profiles,
    concentration_series,
    first_crossing_time,
    radiogenic_heat_through_time,
    temperature_from_heat_profile,
)


def plot_heat_generation_vs_thickness(
    output_path: Path = FIGURES_DIR / "Figure2_CrustalThickness_vs_HeatGeneration.png",
) -> Path:
    """Figure 2: Create crustal thickness, heat production, and enrichment-factor panels."""

    crustal_thickness_km = np.linspace(20, 70, 200)
    series = concentration_series(crustal_thickness_km)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    ax_left = axes[0]
    line_k, = ax_left.plot(crustal_thickness_km, series["k2o_wt"], color=COLOR_K, lw=2, label="K$_2$O (wt%)")
    line_th, = ax_left.plot(crustal_thickness_km, series["th_ppm"], color=COLOR_TH, lw=2, label="Th (ppm)")
    # U (ppm) and K2O (wt%) are numerically close; dashes keep both visible.
    line_u, = ax_left.plot(crustal_thickness_km, series["u_ppm"], color=COLOR_U, lw=2, linestyle="--", label="U (ppm)")
    ax_left.set_xlabel("Crustal thickness (km)", fontsize=12, fontweight="bold")
    ax_left.set_ylabel("Concentration", fontsize=12, fontweight="bold")
    style_axes(ax_left)
    ax_left.tick_params(axis="y", right=False)

    lines = [line_k, line_th, line_u]
    ax_left.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=9, frameon=False)
    ax_left.text(0.95, 0.05, "(a)", ha="right", va="bottom", fontsize=10, fontweight="bold", transform=ax_left.transAxes)

    ax_heat = axes[1]
    heat_color = "red"
    line_heat, = ax_heat.plot(
        crustal_thickness_km,
        series["heat_generation_uW_m3"],
        color=heat_color,
        lw=2,
        linestyle="--",
        label="Heat production",
    )
    ax_heat.set_xlabel("Crustal thickness (km)", fontsize=12, fontweight="bold")
    ax_heat.set_ylabel(r"Heat production ($\mu$W/m$^3$)", fontsize=12, fontweight="bold")
    ax_heat.set_ylim(0.3, max(2.0, float(series["heat_generation_uW_m3"].max()) * 1.05))
    style_axes(ax_heat)
    ax_heat.tick_params(axis="y", which="both", right=False, labelright=False)
    ax_heat.spines["right"].set_visible(False)

    baseline_heat_uW_m3 = TEN_KM_CRUST_HEAT_PRODUCTION_W_M3 * 1e6

    def heat_to_enrichment(heat_generation_uW_m3):
        return np.asarray(heat_generation_uW_m3) / baseline_heat_uW_m3

    def enrichment_to_heat(enrichment):
        return np.asarray(enrichment) * baseline_heat_uW_m3

    ax_enrichment = ax_heat.secondary_yaxis(
        "right",
        functions=(heat_to_enrichment, enrichment_to_heat),
    )
    ax_enrichment.set_ylabel(
        "Enrichment factor",
        fontsize=12,
        fontweight="bold",
        color=heat_color,
    )
    ax_enrichment.minorticks_on()
    ax_enrichment.tick_params(
        axis="y",
        colors=heat_color,
        which="both",
        direction="in",
        right=True,
        labelsize=12,
    )
    ax_enrichment.spines["right"].set_color(heat_color)
    ax_enrichment.spines["right"].set_linewidth(2)
    ax_heat.legend(
        [line_heat],
        [line_heat.get_label()],
        loc="upper left",
        fontsize=9,
        frameon=False,
    )
    ax_heat.text(0.95, 0.05, "(b)", ha="right", va="bottom", fontsize=10, fontweight="bold", transform=ax_heat.transAxes)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    return save_figure(fig, output_path, dpi=300, transparent=False)


def add_mg_number(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1 helper: Calculate Mg# from MgO and FeOT."""

    mg_molar_mass = 24.305 + 15.999
    feo_molar_mass = 55.845 + 15.999
    df = df.copy()
    df["Mg_mol"] = df["MGO(WT%)"] / mg_molar_mass
    df["Fe_mol"] = df["FEOT(WT%)"] / feo_molar_mass
    df["Mg_number"] = 100 * df["Mg_mol"] / (df["Mg_mol"] + df["Fe_mol"])
    return df


def get_bins_and_ticks(x_column: str) -> tuple[np.ndarray, tuple[int, ...]]:
    """Figure 1 helper: Return bin edges and x ticks for a panel."""

    if x_column == "SIO2(WT%)":
        return np.arange(47, 74, 2), (47, 52, 57, 62, 67, 72)
    if x_column == "Mg_number":
        return np.arange(0, 90, 5), (0, 20, 40, 60, 80)
    return np.arange(0, 7, 0.5), (0, 1, 2, 3, 4, 5, 6, 7)


def format_p_value(p_value: float) -> str:
    """Figure 1 helper: Format p-values for compact plot annotations."""

    if not np.isfinite(p_value):
        return "$p$ = n/a"
    if p_value < 0.001:
        return "$p$ < 0.001"
    return f"$p$ = {p_value:.3f}"


def add_binned_statistics(ax: plt.Axes, df: pd.DataFrame, x_column: str, y_column: str) -> None:
    """Overlay statistics only for bins containing at least 50 finite pairs."""

    df = select_finite_pairs(df, x_column, y_column)
    bins = bin_edges(x_column, df[x_column])
    panel_df = df.copy()
    panel_df["bin"] = pd.cut(panel_df[x_column], bins, include_lowest=True)
    intervals = list(panel_df["bin"].cat.categories)
    bin_data = [panel_df.loc[panel_df["bin"] == interval, y_column].values for interval in intervals]
    box_data = [values for values in bin_data if values.size >= 50]
    centers = [interval.mid for interval, values in zip(intervals, bin_data) if values.size >= 50]

    if not box_data:
        return

    width = np.min(np.diff(bins)) * 0.8
    boxplot = ax.boxplot(box_data, positions=centers, widths=width, patch_artist=True, whis=1.5, showfliers=False)
    for box in boxplot["boxes"]:
        box.set(facecolor="white", alpha=0.9)
    for median in boxplot["medians"]:
        median.set(linewidth=2)
    for artist in [*boxplot["whiskers"], *boxplot["caps"]]:
        artist.set_visible(False)

    for center, values in zip(centers, box_data):
        upper_error = values.mean() + 2 * values.std(ddof=1) / np.sqrt(values.size)
        label_y = max(float(np.percentile(values, 75)), float(upper_error))
        ax.annotate(
            f"{values.size}",
            xy=(center, label_y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
            zorder=6,
        )

    means = []
    stderr_2x = []
    mean_centers = []
    for interval in intervals:
        series = panel_df.loc[panel_df["bin"] == interval, y_column].dropna()
        mean_centers.append(interval.mid)
        if series.size < 50:
            # NaNs also break the line across sparse or empty bins.
            means.append(np.nan)
            stderr_2x.append(np.nan)
            continue
        means.append(series.mean())
        stderr_2x.append(2 * series.std(ddof=1) / np.sqrt(series.size) if series.size > 1 else 0.0)

    ax.scatter(mean_centers, means, marker="s", s=30, color="green", edgecolor="black", zorder=5)
    ax.plot(mean_centers, means, color="green", lw=2, zorder=4)
    ax.errorbar(mean_centers, means, yerr=stderr_2x, fmt="none", ecolor="black", capsize=4, zorder=3)



def plot_element_variation(
    input_path: Path = PLOT_DATA,
    output_path: Path = FIGURES_DIR / "Figure1_ElementVariation.png",
) -> Path:
    """Figure 1: Create the multi-panel element-variation figure."""

    df, _ = load_analysis_dataset(input_path)
    df = add_mg_number(df)

    panels = [
        ("Mg_number", "K2O(WT%)", "Mg#", "K$_2$O (wt%)", (10, 80), (0, 8), "(a)"),
        ("SIO2(WT%)", "TH(PPM)", "SiO$_2$ (wt%)", "Th (ppm)", (47, 73), (0, 40), "(c)"),
        ("Mg_number", "TH(PPM)", "Mg#", "Th (ppm)", (10, 80), (0, 40), "(e)"),
        ("K2O(WT%)", "TH(PPM)", "K$_2$O (wt%)", "Th (ppm)", (0, 8), (0, 40), "(g)"),
        ("SIO2(WT%)", "K2O(WT%)", "SiO$_2$ (wt%)", "K$_2$O (wt%)", (47, 73), (0, 8), "(b)"),
        ("SIO2(WT%)", "U(PPM)", "SiO$_2$ (wt%)", "U (ppm)", (47, 73), (0, 12.5), "(d)"),
        ("Mg_number", "U(PPM)", "Mg#", "U (ppm)", (10, 80), (0, 12.5), "(f)"),
        ("K2O(WT%)", "U(PPM)", "K$_2$O (wt%)", "U (ppm)", (0, 8), (0, 12.5), "(h)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for ax, (x_column, y_column, x_label, y_label, xlim, ylim, panel_label) in zip(axes, panels):
        subset = select_finite_pairs(df, x_column, y_column)
        ax.scatter(subset[x_column], subset[y_column], s=12, alpha=0.7, color="#5A5AFF", edgecolor="black", linewidths=0.5)
        add_binned_statistics(ax, subset, x_column, y_column)

        if x_column == "K2O(WT%)" and y_column in {"TH(PPM)", "U(PPM)"} and len(subset) > 1:
            slope, slope_stderr, p_value, r_squared = zero_intercept_regression(
                subset[x_column],
                subset[y_column],
            )
            x_fit = np.linspace(0, subset[x_column].max(), 200)
            ax.plot(x_fit, slope * x_fit, color="black", linestyle="--", lw=4, zorder=4)
            p_text = r"$\bf{p < 0.001}$" if p_value < 0.001 else rf"$\bf{{p = {p_value:.3f}}}$"

            ax.text(
                0.05,
                0.95,
                "\n".join(
                    [
                        rf"$\bf{{y = ({slope:.3f} \pm {slope_stderr:.3f})x}}$",
                        rf"$\bf{{R^2 = {r_squared:.3f}}}$",
                        p_text,
                    ]
                ),
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
            )
        _, xticks = get_bins_and_ticks(x_column)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(tick)) for tick in xticks])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
        ax.text(0.95, 0.95, panel_label, ha="right", va="top", fontsize=12, fontweight="bold", transform=ax.transAxes)
        style_axes(ax)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return save_figure(fig, output_path, dpi=300, transparent=False)


def plot_temperature_profile(
    output_path: Path = FIGURES_DIR / "Figure4_Temperature_Profile_and_A_vs_depth.png",
) -> Path:
    """Figure 4: Create the temperature-profile and heat-production depth figure."""

    profiles = build_heat_profiles()
    depth = profiles["depth_m"]
    eroded_for_temperature = np.nan_to_num(profiles["eroded"], nan=0.0)
    eroded_mask = np.isnan(profiles["eroded"])

    temperature_undiff = temperature_from_heat_profile(depth, profiles["undifferentiated"])
    temperature_diff = temperature_from_heat_profile(depth, profiles["differentiated"])
    temperature_eroded = temperature_from_heat_profile(depth, eroded_for_temperature)
    temperature_eroded[eroded_mask] = np.nan

    fig, (ax_temp, ax_heat) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    depth_km = depth / 1000.0
    crust_boundary_depth_km = profiles["h_u"] / 1000.0
    ax_temp.plot(temperature_undiff, depth_km, color=COLOR_UNDIFF, lw=2, label="Undifferentiated")
    ax_temp.plot(temperature_diff, depth_km, color=COLOR_DIFF, linestyle="--", lw=2, label="Differentiated")
    ax_temp.plot(temperature_eroded[~eroded_mask], depth_km[~eroded_mask], color=COLOR_EROSION, lw=2, linestyle="-.", label="Eroded")
    ax_temp.set_xlabel(r"Temperature ($^\circ$C)", fontweight="bold", fontsize=12)
    ax_temp.set_ylabel("Depth (km)", fontweight="bold", fontsize=12)
    ax_temp.set_ylim(0, 65)
    ax_temp.invert_yaxis()
    ax_temp.text(0.95, 0.05, "(a)", ha="right", va="bottom", fontsize=10, fontweight="bold", transform=ax_temp.transAxes)
    yaxis_transform = ax_temp.get_yaxis_transform()
    ax_temp.text(0.96, crust_boundary_depth_km - 0.8, f"E = {profiles['e']:.2f}", transform=yaxis_transform, ha="right", va="bottom", fontsize=9, fontweight="bold")
    # Align all three inline labels at the same depth.
    label_depth_km = 30.0
    label_temperatures = [
        float(np.interp(label_depth_km, depth_km, temperature))
        for temperature in (temperature_undiff, temperature_diff, temperature_eroded)
    ]
    label_lines_or_legend(ax_temp, lines=list(ax_temp.lines),
                          xvals=label_temperatures, fontsize=10, fixed=True)
    ax_heat.plot(profiles["differentiated"] / 1e-6, depth_km, color=COLOR_DIFF, lw=2, linestyle="--")
    ax_heat.plot(profiles["undifferentiated"] / 1e-6, depth_km, color=COLOR_UNDIFF, lw=2.0)
    # The eroded model has zero heat production and overlaps the deep
    # differentiated profile; a narrower dash-dot line shows both styles.
    ax_heat.plot(profiles["eroded"] / 1e-6, depth_km,
                 color=COLOR_EROSION, lw=1.5, linestyle="-.", zorder=4)
    ax_heat.set_xlabel(r"$A$ ($\mu$W/m$^3$)", fontweight="bold", fontsize=12)
    ax_heat.set_xlim(-0.05, max(2.5, np.nanmax(profiles["differentiated"] / 1e-6) * 1.1))
    ax_heat.text(0.95, 0.05, "(b)", ha="right", va="bottom", fontsize=10, fontweight="bold", transform=ax_heat.transAxes)

    for ax in [ax_temp, ax_heat]:
        ax.axhline(crust_boundary_depth_km, color="black", linestyle=":", lw=1.4, zorder=1)
        style_axes(ax)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.18)
    return save_figure(fig, output_path)

def plot_current_basal_temperature() -> Path:
    """Figure 5: Plot present-day basal temperatures."""
    thickness_m = np.linspace(10_000, 80_000, 300)
    thickness_km = thickness_m / 1000.0
    solidus_dry, solidus_wet = get_eclogite_solidus(thickness_km)
    enrichment_factors = np.arange(2.0, 7.0, 1.0)
    colormap = model_colormap()

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot(thickness_km, solidus_dry, lw=2, linestyle="--", color="black")
    ax.plot(thickness_km, solidus_wet, lw=2, linestyle="--", color="black")
    constant_enrichment_profiles = {}
    constant_enrichment_colors = {}
    for enrichment in enrichment_factors:
        temperature = basal_temperature_undifferentiated(
            thickness_m,
            DEFAULT_CONSTANTS.heat_present_w_m3 / enrichment,
        )
        constant_enrichment_profiles[enrichment] = temperature
        color = colormap((enrichment - 2.0) / 4.0)
        constant_enrichment_colors[enrichment] = color
        ax.plot(
            thickness_km,
            temperature,
            lw=2,
            linestyle=":",
            color=color,
            label=rf"$E = {enrichment:g}$",
        )

    undifferentiated_temperature = basal_temperature_undifferentiated(
        thickness_m,
        DEFAULT_CONSTANTS.heat_present_w_m3,
    )
    differentiated_temperature = basal_temperature_differentiated(
        thickness_m,
        DEFAULT_CONSTANTS.heat_present_w_m3 * DEFAULT_CONSTANTS.u238_fraction,
        DEFAULT_CONSTANTS.heat_present_w_m3 * DEFAULT_CONSTANTS.u235_fraction,
        DEFAULT_CONSTANTS.heat_present_w_m3 * DEFAULT_CONSTANTS.th_fraction,
        DEFAULT_CONSTANTS.heat_present_w_m3 * DEFAULT_CONSTANTS.k_fraction,
    )
    differentiated_line, = ax.plot(
        thickness_km,
        differentiated_temperature,
        lw=2,
        linestyle="--",
        color=COLOR_DIFF,
        label="Differentiated, E(H)",
    )
    undifferentiated_line, = ax.plot(
        thickness_km,
        undifferentiated_temperature,
        lw=2,
        linestyle="-",
        color=COLOR_UNDIFF,
        label="Undifferentiated",
    )


    ax.text(32, 770, "Water-saturated solidus", fontsize=10)
    ax.text(32, 1000, "Dry solidus", fontsize=10, rotation=7)
    ax.set_xlabel("Crustal thickness H (km)", fontweight="bold", fontsize=12)
    ax.set_ylabel(r"Basal temperature ($^\circ$C)", fontweight="bold", fontsize=12)
    ax.set_xlim(30, 80)
    ax.set_ylim(200, 1200)
    style_axes(ax)

    right_axis_transform = ax.get_yaxis_transform()
    label_y_offsets = {4.0: 12.0, 5.0: -2.0, 6.0: -16.0}
    for enrichment in enrichment_factors:
        y_right = float(constant_enrichment_profiles[enrichment][-1])
        y_right += label_y_offsets.get(float(enrichment), 0.0)
        ax.text(
            1.015,
            y_right,
            rf"$E = {enrichment:g}$",
            color=constant_enrichment_colors[enrichment],
            fontsize=8,
            ha="left",
            va="center",
            transform=right_axis_transform,
            clip_on=False,
        )

    ax.legend(
        [undifferentiated_line, differentiated_line],
        [undifferentiated_line.get_label(), differentiated_line.get_label()],
        loc="upper right",
        bbox_to_anchor=(0.79, 0.9),
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    return save_figure(fig, FIGURES_DIR / "Figure5_basalT_vs_thickness.png")


def plot_differentiated_geotherms() -> Path:
    """Figure 7: Plot present-day geotherms for differentiated crust."""
    fig, ax_geotherm = plt.subplots(figsize=(5.4, 4.8))
    retention = 0.3
    thicknesses_km = np.arange(40, 81, 10)
    colors = model_colormap()(np.linspace(0, 0.9, thicknesses_km.size))
    reference_depth_km = 40
    ax_geotherm.axhline(reference_depth_km, color="black", lw=1, zorder=1)
    style_axes(ax_geotherm)

    for thickness, color in zip(thicknesses_km, colors):
        thickness_m = thickness * 1000
        depth_km = np.unique(np.r_[np.linspace(0, thickness, 301), reference_depth_km])
        temperature = geotherm(depth_km * 1000, thickness_m, retention)
        shallow = depth_km <= reference_depth_km
        ax_geotherm.plot(temperature[shallow], depth_km[shallow], color=color, lw=1.6)
        if thickness > reference_depth_km:
            deep = depth_km >= reference_depth_km
            ax_geotherm.plot(
                temperature[deep], depth_km[deep], color=color, lw=1.6, linestyle="--",
            )

        moho = (temperature[-1], thickness)
        ax_geotherm.plot(*moho, color=color, marker="o", ms=3.5, zorder=3)
        ax_geotherm.annotate(
            f"{thickness} km", xy=moho,
            xytext=(-14, 6) if thickness == reference_depth_km else (-7, 0),
            textcoords="offset points", ha="right", va="center", fontsize=9,
        )

    ax_geotherm.set_ylabel("Depth (km)", fontweight="bold", fontsize=12)
    ax_geotherm.set_ylim(85, 0)
    ax_geotherm.set_xlabel(r"Temperature ($^\circ$C)", fontweight="bold", fontsize=12)
    ax_geotherm.set_xlim(0, 850)
    ax_geotherm.set_xticks(np.arange(0, 801, 100))

    ax_geotherm.set_title(
        rf"Differentiated crust ($R = {retention:g}$)",
        loc="left", fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return save_figure(fig, FIGURES_DIR / "Figure7_differentiated_geotherms.png")


def plot_basal_temperature_by_geologic_time() -> Path:
    """Figure 8: Plot basal temperature against crustal thickness for selected geologic times."""

    times = np.linspace(0, 4.5e9, 10)
    heat_u238, heat_u235, heat_th, heat_k, heat_total = radiogenic_heat_through_time(times)
    thickness_m = np.linspace(10_000, 80_000, 300)
    thickness_km = thickness_m / 1000.0
    solidus_dry, solidus_wet = get_eclogite_solidus(thickness_km)
    colormap = model_colormap()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, title in zip(axes, ["Undifferentiated", "Differentiated"]):
        ax.plot(thickness_km, solidus_dry, lw=2, linestyle="--", color="black")
        ax.plot(thickness_km, solidus_wet, lw=2, linestyle="--", color="black")
        ax.text(32, 780, "Water-saturated solidus", fontsize=10)
        ax.text(32, 1000, "Dry solidus", fontsize=10, rotation=7)
        style_axes(ax)
        ax.set_xlabel("Crustal thickness H (km)", fontweight="bold", fontsize=12)
        ax.set_xlim(30, 80)
        ax.set_ylim(300, 1200)
        ax.set_title(title, fontsize=10, fontweight="bold")

    axes[0].set_ylabel(r"Basal temperature ($^\circ$C)", fontweight="bold", fontsize=12)
    for time, heat in zip(times, heat_total):
        label = "Present" if time == 0 else f"{time / 1e9:.1f}"
        axes[0].plot(thickness_km, basal_temperature_undifferentiated(thickness_m, heat), lw=2, color=colormap(time / 4.5e9), label=label)

    for time, u238, u235, th, k in zip(times, heat_u238, heat_u235, heat_th, heat_k):
        label = "Present" if time == 0 else f"{time / 1e9:.1f}"
        axes[1].plot(thickness_km, basal_temperature_differentiated(thickness_m, u238, u235, th, k), lw=2, color=colormap(time / 4.5e9), label=label)

    # Arrange age labels in two alternating rows across the fan.
    age_lines = axes[0].get_lines()[-len(times):]
    label_x = [np.interp(930 + 90*(i % 2), line.get_ydata(), line.get_xdata())
               for i, line in enumerate(age_lines)]
    label_lines_or_legend(axes[0], xvals=label_x, lines=age_lines, fontsize=7)
    label_lines_or_legend(axes[1], lines=axes[1].get_lines()[-5:])
    add_model_colorbar(fig, axes, colormap, (0, 4.5), np.arange(0, 4.6, 0.5), "Age (Ga)")
    for ax, panel in zip(axes, ("(a)", "(b)")):
        ax.text(0.96, 0.04, panel, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, fontweight="bold")
    return save_figure(fig, FIGURES_DIR / "Figure8_basalT_vs_thickness_GeologicTimes.png")


def plot_basal_temperature_vs_time() -> Path:
    """Figure 9: Plot basal temperature through geologic time for selected crustal thicknesses."""

    time = np.linspace(0, 4.5e9, 400)
    time_gyr = time / 1e9
    heat_u238, heat_u235, heat_th, heat_k, heat_total = radiogenic_heat_through_time(time)
    thicknesses_m = np.linspace(30_000, 80_000, 6)
    solidus_thicknesses_m = np.linspace(10_000, 100_000, 100)
    colormap = model_colormap()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, mode in zip(axes, ["Undifferentiated", "Differentiated"]):
        for thickness in thicknesses_m:
            color = colormap((thickness / 1000 - 30) / 50)
            if mode == "Undifferentiated":
                temperature = basal_temperature_undifferentiated(thickness, heat_total)
            else:
                temperature = basal_temperature_differentiated(thickness, heat_u238, heat_u235, heat_th, heat_k)
            ax.plot(time_gyr, temperature, lw=2, color=color, label=f"{thickness / 1000:.0f}")

        wet_crossings = []
        wet_temperatures = []
        dry_crossings = []
        dry_temperatures = []
        for thickness in solidus_thicknesses_m:
            if mode == "Undifferentiated":
                temperature = basal_temperature_undifferentiated(thickness, heat_total)
            else:
                temperature = basal_temperature_differentiated(thickness, heat_u238, heat_u235, heat_th, heat_k)
            solidus_dry, solidus_wet = get_eclogite_solidus(thickness / 1000)
            wet_crossings.append(first_crossing_time(time, temperature - solidus_wet) / 1e9)
            wet_temperatures.append(solidus_wet)
            dry_crossings.append(first_crossing_time(time, temperature - solidus_dry) / 1e9)
            dry_temperatures.append(solidus_dry)

        ax.plot(wet_crossings, wet_temperatures, "--", color="black", zorder=1)
        ax.plot(dry_crossings, dry_temperatures, "--", color="black", zorder=1)
        for ages, temperatures, label in (
            (wet_crossings, wet_temperatures, "Water-saturated solidus"),
            (dry_crossings, dry_temperatures, "Dry solidus"),
        ):
            visible = [(float(a), float(t)) for a, t in zip(ages, temperatures)
                       if np.isfinite(a) and 0 <= a <= 4.5 and 200 <= t <= 1400]
            if visible:
                age, temp = min(visible, key=lambda point: abs(point[0] - 4.0))
                if label == "Water-saturated solidus":
                    # Move only the label down by 20 degrees on the temperature axis.
                    ax.annotate(label, (age, temp), xytext=(age, temp - 20),
                                textcoords="data", fontsize=9, va="top")
                else:
                    ax.annotate(label, (age, temp), xytext=(0, 6),
                                textcoords="offset points", fontsize=9, va="bottom")

        ax.set_xlabel("Age (Ga)", fontweight="bold", fontsize=12)
        ax.set_xlim(4.5, 0)
        ax.set_ylim(200, 1400)
        ax.set_title(mode, fontsize=10, fontweight="bold")
        style_axes(ax)








    axes[0].set_ylabel(r"Basal temperature ($^\circ$C)", fontweight="bold", fontsize=12)
    label_lines_or_legend(axes[0], xvals=(1.4,) * 6, lines=axes[0].get_lines()[: len(thicknesses_m)], fixed=True)
    label_lines_or_legend(axes[1], xvals=(1.4,) * 6, lines=axes[1].get_lines()[: len(thicknesses_m)], fixed=True)

    add_model_colorbar(fig, axes, colormap, (30, 80), np.arange(30, 81, 10), "Crustal thickness (km)")
    for ax, panel in zip(axes, ("(a)", "(b)")):
        ax.text(0.96, 0.04, panel, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, fontweight="bold")
    return save_figure(fig, FIGURES_DIR / "Figure9_basalT_vs_geologic_time.png")


def plot_all_thermal_models() -> list[Path]:
    """Generate Figures 5 and 7–9."""

    return [
        plot_current_basal_temperature(),
        plot_differentiated_geotherms(),
        plot_basal_temperature_by_geologic_time(),
        plot_basal_temperature_vs_time(),
    ]
