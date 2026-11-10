"""Loss landscape of ONE neuron + ReLU on Anna's V, with gradient-descent paths from four starts.

The model is y_hat = ReLU(m*u + b), with u = x scaled to mean 0 and spread 1 (as in v_fit.py).
For every (m, b) on a grid we compute the loss, draw its contours, and then run gradient
descent from four different starting guesses to see where each one ends up.

Run:  python relu_landscape.py      ->  figures/relu_landscape.png
"""
import matplotlib.pyplot as plt
import numpy as np

from common import AQUA, BLUE, GRID, INK, INK_2, MUTED, ORANGE, load_v, relu, save, standardize, style
from v_fit import LR_NEURON, train_neuron

EPOCHS = 3000
STARTS = [   # (m, b), color, label, label offset in points
    ((1.0, 0.0), BLUE, "Excel start (1, 0)", (8, 6)),
    ((1.0, -1.0), ORANGE, "start (1, -1)", (8, 6)),
    ((-5.0, -2.0), AQUA, "start (-5, -2)", (8, 6)),
    ((1.0, -3.0), INK, "start (1, -3): dead, never moves", (10, -26)),
]


def main():
    x, y = load_v()
    u, _, _ = standardize(x)

    ms = np.linspace(-60, 60, 361)
    bs = np.linspace(-45, 55, 301)
    M, B = np.meshgrid(ms, bs)
    Z = M[..., None] * u + B[..., None]                       # (b, m, point)
    L = np.mean((y - relu(Z)) ** 2, axis=-1)                  # loss at every (m, b)
    dead = (Z <= 0).all(axis=-1)                              # ReLU off for every point

    style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(False)
    ax.contourf(M, B, dead.astype(float), levels=[0.5, 1.5], colors=[GRID])
    levels = np.geomspace(L.min() * 1.01, L.max(), 14)
    cs = ax.contour(M, B, L, levels=levels, colors=MUTED, linewidths=0.7)
    ax.clabel(cs, levels[::3], fmt="%.0f", fontsize=7, colors=MUTED)
    ax.text(-8, -40, "dead zone: ReLU is off for every point,\nso every gradient is 0 and nothing moves",
            color=INK_2, fontsize=9, ha="center")

    print(f"{'start':>14} -> {'m':>8} {'b':>8} {'MSE':>8}  ReLU on for")
    for (m0, b0), color, label, offset in STARTS:
        h = np.array(train_neuron(u, y, m0, b0, LR_NEURON, EPOCHS, use_relu=True))
        m_end, b_end, loss_end = h[-1, :3]
        on = int((m_end * u + b_end > 0).sum())
        print(f"{str((m0, b0)):>14} -> {m_end:8.2f} {b_end:8.2f} {loss_end:8.1f}  {on}/{len(u)} points")
        ax.plot(h[:, 0], h[:, 1], color=color, lw=2, zorder=3)
        ax.plot(h[:6, 0], h[:6, 1], "o", color=color, ms=4, zorder=4)     # first 5 steps
        ax.plot([m0], [b0], marker="s", color=color, ms=7, mec="white", zorder=5)
        ax.plot([m_end], [b_end], marker="o", color=color, ms=10, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{label}\n-> MSE {loss_end:.0f}", (m_end, b_end), textcoords="offset points",
                    xytext=offset, fontsize=8.5, color=INK, zorder=6)

    ax.set_xlabel("m (weight)")
    ax.set_ylabel("b (bias)")
    ax.set_title("One neuron + ReLU: where gradient descent ends depends on where it starts", loc="left")
    ax.text(0.0, -0.13, "Squares = starting guesses, dots = first 5 steps, big circles = where each run "
            "ends after 3000 epochs.  Contours = MSE.", transform=ax.transAxes, color=INK_2, fontsize=8.5)
    save(fig, "relu_landscape.png")


if __name__ == "__main__":
    main()
