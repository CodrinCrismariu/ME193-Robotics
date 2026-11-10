"""Shared helpers for the step-1 scripts: load Anna's V data, scale it, and style the plots."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "annacode-v.tsv"
FIG_DIR = HERE / "figures"

# Plot colors (validated categorical palette: blue, orange, aqua, yellow)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DATA_GRAY = "#b5b3ab"


def load_v():
    """Anna's V example, saved from the AnnaCode builder page.

    Columns in the file: Time (s), x, output 1.  The page made it with
    y = 2|x - 60| + noise (noise std 2), 120 points, x uniform on [20, 100].
    """
    d = np.loadtxt(DATA_FILE, skiprows=1, delimiter="\t")
    return d[:, 1], d[:, 2]


def standardize(v):
    """Shift to mean 0 and scale to spread 1. Returns (scaled, mean, std).

    np.std is the population std (divide by N), the same as Excel's STDEVP
    and the same scheme Anna's code uses before it trains.
    """
    mean, std = v.mean(), v.std()
    return (v - mean) / std, mean, std


def relu(z):
    return np.maximum(0.0, z)


def relu_prime(z):
    # Slope of ReLU: 1 where z > 0, else 0 (same rule as AnnaCode's activations.py)
    return (z > 0).astype(float)


def style():
    matplotlib.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": "white",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path.relative_to(HERE)}")
