"""Compare the baseline and the background-randomized classifier on the three shortcut tests.

Run after both trainings:  ..\\.venv\\Scripts\\python compare_runs.py      ->  figures/shortcut_test.png
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = [("baseline", "usual augmentation (40 epochs)", "#9aa4ae"), ("randomized_200", "+ background randomization (200 epochs)", "#e8743b")]
TESTS = [("test_real", "real test photos"), ("test_swapped_background", "hands pasted onto\nother letters' scenes"),
         ("hand_removed_still_right", "hand painted out:\nstill names the letter")]


def main():
    res = {k: json.loads((HERE / "runs" / k / "metrics.json").read_text()) for k, _, _ in RUNS}
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False,
                                "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    xs = np.arange(len(TESTS))
    for i, (k, label, color) in enumerate(RUNS):
        vals = [res[k][t] for t, _ in TESTS]
        bars = ax.bar(xs + (i - 0.5) * 0.36, vals, 0.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], padding=2, fontsize=9)
    ax.axhline(1 / 7, color="#52514e", lw=1, ls=(0, (4, 3)))
    ax.text(-0.55, 1 / 7 + 0.015, "chance (1 in 7)", ha="left", fontsize=9, color="#52514e")
    ax.set_xticks(xs, [t for _, t in TESTS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of held-out test photos")
    ax.legend(frameon=False, loc="upper right")
    n = res["randomized_200"]["n_test"]
    ax.set_title(f"Does the letter classifier read the hand or the background?  ({n} photos from held-out bursts)",
                 loc="left", fontsize=11, fontweight="bold")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "shortcut_test.png", dpi=150, bbox_inches="tight")
    for k, label, _ in RUNS:
        r = res[k]
        print(f"{k:11s} real {r['test_real']:.1%}   swapped background {r['test_swapped_background']:.1%}   "
              f"hand removed: still right {r['hand_removed_still_right']:.1%}, says none {r['hand_removed_says_none']:.1%}"
              f"   (best epoch {r['best_epoch']})")
    print("saved figures/shortcut_test.png")


if __name__ == "__main__":
    main()
