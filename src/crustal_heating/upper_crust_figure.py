"""Effective upper-crust thickness vs. lower-crust HPE retention R, contoured by crustal thickness H.

h(R, H) = (1 - R) * H / E(H), from crustal_heating.hpe_retention.layer_properties.
Each curve is a constant-H contour (30-80 km); colour encodes H.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .paths import PROJECT_ROOT as ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from crustal_heating.hpe_retention import layer_properties
from crustal_heating.style import style_axes, save_figure, label_lines_or_legend, continuous_colorbar, model_colormap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--h-min-km", type=float, default=30.0)
    parser.add_argument("--h-max-km", type=float, default=80.0)
    # 70 km is omitted by default because its inline label overlaps the 60 and 80 km labels.
    parser.add_argument("--h-values-km", type=float, nargs="+", default=[30, 40, 50, 80])
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    R = np.linspace(0, 0.5, 201)
    H_km = np.asarray(args.h_values_km, dtype=float)
    cmap = model_colormap()
    norm = Normalize(args.h_min_km, args.h_max_km)

    fig, ax = plt.subplots(figsize=(6, 5))
    lines = []
    h_all = []
    for H in H_km:
        h, _, _ = layer_properties(H * 1000, R)
        (line,) = ax.plot(R, h / 1000, color=cmap(norm(H)), lw=2, label=f"{H:.0f}")
        lines.append(line)
        h_all.append(h / 1000)

    h_lo, h_hi = float(np.min(h_all)), float(np.max(h_all))
    pad = 0.05 * (h_hi - h_lo)
    y_min = max(0.0, np.floor((h_lo - pad) * 2) / 2)
    y_max = np.ceil((h_hi + pad) * 2) / 2
    ax.set(xlim=(0, 0.5), ylim=(y_min, y_max))
    ax.set_xlabel(r"Lower-layer HPE mass fraction, $R$", fontweight="bold", fontsize=12)
    ax.set_ylabel("Effective thickness of\nHPE-enriched upper layer (km)", fontweight="bold", fontsize=12)
    style_axes(ax)
    fig.subplots_adjust(left=0.20, right=0.92, bottom=0.16, top=0.95)

    fig.canvas.draw()
    pos = ax.get_position()

    label_lines_or_legend(ax, lines=lines)
    out = args.output_dir / "Figure3_HPE_upper_crust_thickness_vs_R.png"
    save_figure(fig, out, dpi=400)
    print(out)


if __name__ == "__main__":
    main()
