from __future__ import annotations

import argparse
from pathlib import Path

from .paths import PROJECT_ROOT as ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from crustal_heating.hpe_retention import basal_temperature, geotherm, layer_properties
from crustal_heating.solidus import get_eclogite_solidus
from crustal_heating.style import COLOR_UNDIFF, style_axes, save_figure, label_lines_or_legend, model_colormap, continuous_colorbar
from crustal_heating.thermal import enrichment_factor



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-thickness-km", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args(argv)
    H = args.profile_thickness_km * 1000
    layer_properties(H, 0)  # validate the requested thickness
    args.output_dir.mkdir(parents=True, exist_ok=True)
    R_values = np.arange(6) / 10
    cmap = model_colormap()
    norm = Normalize(0, 0.5)
    colors = cmap(norm(R_values))
    depth = np.linspace(0, H, 601)
    thickness = np.linspace(30e3, 80e3, 501)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.8))
    rows = []
    for R, color in zip(R_values, colors):
        h, upper, lower = layer_properties(H, R)
        z = np.unique(np.r_[depth, h])
        left.plot(geotherm(z, H, R), z / 1000, color=color, lw=2)
        interface_T = float(geotherm(h, H, R))
        left.plot(interface_T, h / 1000, "o", ms=4, color=color, zorder=5)
        left.hlines(h / 1000, 0, interface_T, color=color, lw=1.0, ls=":", alpha=0.8)
        Tb = basal_temperature(thickness, R)
        right.plot(thickness / 1000, Tb, color=color, lw=2, ls="-")
        rows.append([R, H / 1000, float(enrichment_factor(H)), float(h / 1000),
                     float(upper * 1e6), float(lower * 1e6), float(basal_temperature(H, R))])

    # The R=1 limit reproduces a uniform, undifferentiated crust.
    left.plot(geotherm(depth, H, 1), depth / 1000, color=COLOR_UNDIFF, lw=2)
    right.plot(thickness / 1000, basal_temperature(thickness, 1), color=COLOR_UNDIFF, lw=2)
    dry, wet = get_eclogite_solidus(thickness / 1000)
    for temp, label in ((wet, "Water-saturated solidus"), (dry, "Dry solidus")):
        right.plot(thickness / 1000, temp, color="black", lw=1.5, ls="--")
        right.text(32, temp[20] + 20, label, fontsize=9, va="bottom")

    left.set(xlim=(0, 800), ylim=(H / 1000 + 3, 0))
    left.set_xlabel(r"Temperature ($^\circ$C)", fontweight="bold", fontsize=12)
    left.set_ylabel("Depth (km)", fontweight="bold", fontsize=12)
    right.set(xlim=(30, 80), ylim=(200, 1200))
    right.set_xlabel("Crustal thickness H (km)", fontweight="bold", fontsize=12)
    right.set_ylabel(r"Basal temperature ($^\circ$C)", fontweight="bold", fontsize=12)
    for ax, panel in zip((left, right), ("(a)", "(b)")):
        ax.set_box_aspect(1)
        style_axes(ax)
        ax.text(0.96, 0.04, panel, transform=ax.transAxes, ha="right", fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.14, top=0.91, wspace=0.28)
    left.legend(handles=[
        Line2D([], [], color=COLOR_UNDIFF, lw=2, label="Undifferentiated"),
        Line2D([], [], color="0.35", marker="o", ls=":", ms=4,
               label="Upper-crust thickness"),
    ], loc="upper right", frameon=False, fontsize=9)
    # Draw first so curve-label angles and the shared colorbar use final square axes.
    fig.canvas.draw()
    pos = right.get_position()
    colorbar_ax = fig.add_axes([0.90, pos.y0, 0.024, pos.height])
    colorbar = continuous_colorbar(fig, colorbar_ax, cmap, (0, 0.5), R_values)
    colorbar.set_label(r"Lower-crust HPE mass fraction, $R$", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)

    for ax in (left, right):
        curves = [line for line in ax.lines if len(line.get_xdata()) > 10][:6]
        for line, R in zip(curves, R_values):
            line.set_label(f"{R:.1f}")
        label_lines_or_legend(ax, lines=curves)
    stem = args.output_dir / "Figure6_HPE_retention"
    fig.set_facecolor("white")
    save_figure(fig, stem.with_suffix(".png"), dpi=400, transparent=False)
    print(stem.with_suffix(".png"))
    for row in rows:
        print(f"R={row[0]:.1f}: upper crust={row[3]:.3f} km; basal T={row[6]:.2f} C")


if __name__ == "__main__":
    main()
