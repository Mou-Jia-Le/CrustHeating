"""Shared plotting styles."""

from pathlib import Path
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np


COLOR_U = "#7A3E9D"
COLOR_TH = "#008C8C"
COLOR_K = "#4C7D00"
COLOR_UNDIFF = "#F54900"
COLOR_DIFF = "#FF8000"
COLOR_EROSION = "#3333FF"


def style_axes(ax: plt.Axes, grid: bool = False) -> None:
    """Apply a consistent publication-style axis format."""

    ax.set_box_aspect(1)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.tick_params(axis="both", labelsize=12)
    if grid:
        ax.grid(color="grey", which="major", linestyle="-.", linewidth=0.6, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = 400,
    transparent: bool = False,
) -> Path:
    """Save a figure and close it."""

    fig.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    _place_inline_labels(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Render to a fresh file before replacing the destination. This avoids
    # truncating an existing PNG during a failed write in a synced folder.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".figure-", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        fig.savefig(temporary_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        plt.close(fig)
    return output_path


def label_lines_or_legend(ax: plt.Axes, xvals=None, lines=None, fontsize: int = 8, fixed: bool = False) -> None:
    """Defer inline labels until the final axes geometry is available."""
    ax._inline_label_request = (list(lines if lines is not None else ax.get_lines()), fontsize, xvals, fixed)


def _place_inline_labels(fig):
    """Choose visible positions with clearance from other labels and curves."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        request = getattr(ax, "_inline_label_request", None)
        if request is None:
            continue
        lines, fontsize, xvals, fixed = request
        occupied = [t.get_window_extent(renderer).expanded(1.15, 1.3)
                    for t in ax.texts if t.get_visible()]
        legend = ax.get_legend()
        if legend is not None:
            occupied.append(legend.get_window_extent(renderer))
        bounds = ax.get_window_extent(renderer)
        all_points = {line: ax.transData.transform(np.column_stack(line.get_data()))
                      for line in ax.lines}
        for line_index, line in enumerate(lines):
            points = all_points[line]
            if fixed:
                data_x, data_y = map(np.asarray, line.get_data())
                order = np.argsort(data_x)
                x = xvals[line_index]
                y = np.interp(x, data_x[order], data_y[order])
                step = (data_x.max() - data_x.min()) * 0.001
                tangent = ax.transData.transform([(x-step, np.interp(x-step, data_x[order], data_y[order])),
                                                 (x+step, np.interp(x+step, data_x[order], data_y[order]))])
                dx, dy = tangent[1] - tangent[0]
                angle = (np.degrees(np.arctan2(dy, dx)) + 90) % 180 - 90
                ax.text(x, y, line.get_label(), color="black", fontsize=fontsize,
                        ha="center", va="center", rotation=angle, rotation_mode="anchor",
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.25), zorder=8)
                continue
            candidates = []
            for i in np.linspace(1, len(points)-2, 100).astype(int):
                x, y = points[i]
                if not (bounds.x0+24 < x < bounds.x1-24 and bounds.y0+18 < y < bounds.y1-18):
                    continue
                delta = points[i+1]-points[i-1]
                angle = np.degrees(np.arctan2(delta[1], delta[0]))
                angle = (angle+90) % 180-90
                xy = ax.transData.inverted().transform(points[i])
                text = ax.text(*xy, line.get_label(), color="black", fontsize=fontsize,
                               ha="center", va="center", rotation=angle, rotation_mode="anchor",
                               bbox=dict(facecolor="white", edgecolor="none", pad=0.25), zorder=8)
                box = text.get_window_extent(renderer).expanded(1.2, 1.4)
                overlaps = sum(box.overlaps(other) for other in occupied)
                clearance = min((np.nanmin(np.linalg.norm(p-points[i], axis=1))
                                 for other, p in all_points.items() if other is not line and len(p)), default=100)
                text.update_bbox_position_size(renderer)
                patch = text.get_bbox_patch()
                label_path = patch.get_path().transformed(patch.get_transform())
                curve_overlaps = sum(other.get_path().transformed(other.get_transform()).intersects_path(label_path, filled=False)
                                     for other in ax.lines if other is not line)
                if xvals is None:
                    preference = clearance
                else:
                    target_x = ax.transData.transform((xvals[line_index], 0))[0]
                    preference = -abs(x - target_x) * 10
                score = preference - (overlaps + curve_overlaps)*10000
                if not bounds.contains(box.x0, box.y0) or not bounds.contains(box.x1, box.y1):
                    score -= 10000
                candidates.append((score, xy, angle))
                text.remove()
            if not candidates:
                continue
            _, xy, angle = max(candidates, key=lambda item: item[0])
            text = ax.text(*xy, line.get_label(), color="black", fontsize=fontsize,
                           ha="center", va="center", rotation=angle, rotation_mode="anchor",
                           bbox=dict(facecolor="white", edgecolor="none", pad=0.25), zorder=8)
            occupied.append(text.get_window_extent(renderer).expanded(1.2, 1.4))


def model_colormap():
    """End at the former 4.0 Ga color on the full 0-4.5 Ga scale."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("model_viridis", plt.cm.viridis(np.linspace(0, 4.0 / 4.5, 256)))


def add_model_colorbar(fig, axes, cmap, limits, ticks, label):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.90, wspace=0.25)
    fig.canvas.draw()
    pos = axes[-1].get_position()
    cax = fig.add_axes([0.89, pos.y0, 0.024, pos.height])
    bar = continuous_colorbar(fig, cax, cmap, limits, ticks)
    bar.set_label(label, fontsize=11)
    bar.ax.tick_params(labelsize=10)


def continuous_colorbar(fig, cax, cmap, limits, values):
    """Continuous scale shared with the model curves."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    bar = fig.colorbar(ScalarMappable(norm=Normalize(*limits), cmap=cmap),
                       cax=cax, ticks=values)
    bar.outline.set_linewidth(0.8)
    return bar
