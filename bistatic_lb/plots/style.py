"""One visual style for the app, the notebooks and the report.

Colors follow fixed categorical slots, so the same entity keeps its color across figures.
Line style and direct labels also carry identity, so charts survive grey-scale printing.
"""

from __future__ import annotations

import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # the app and scripts draw off-screen; notebooks keep their inline backend
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 3))]
SEQUENTIAL = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"


def new_figure(width: float = 7.2, height: float = 3.9):
    """A figure with one axes, carrying the shared style."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=130, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    return fig, ax


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11, color=INK, fontweight="semibold")


def legend(ax, **kwargs) -> None:
    kwargs.setdefault("frameon", False)
    ax.legend(fontsize=9, labelcolor=INK, **kwargs)


def marker_note(ax, x: float, y: float, text: str, offset=(6, 6)) -> None:
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=9, color=INK)
