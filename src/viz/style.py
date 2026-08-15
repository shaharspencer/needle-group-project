"""Shared chart styling: palette, typography, grid.

Figures render on a light surface for PDF embedding.
"""

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed categorical order. Never cycled, never reassigned by rank.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",          # never dashed
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    })


def strip_spines(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def caption(fig, text: str) -> None:
    """One line under the chart saying what to take from it."""
    fig.text(0.0, -0.02, text, ha="left", va="top",
             fontsize=8, color=INK_SECONDARY, wrap=True)
