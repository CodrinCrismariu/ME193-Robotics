"""Live training plots: reads the training logs (runs_*.log) every few seconds and redraws the curves.

Run from this folder while training runs:   ..\\.venv\\Scripts\\python watch_training.py
                                             ..\\.venv\\Scripts\\python watch_training.py --every 10
Close the window to stop.
"""
import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CLF = re.compile(r"epoch\s+(\d+)\s+loss ([\d.]+)\s+train ([\d.]+)\s+val real ([\d.]+)\s+val swapped bg ([\d.]+)")
DET = re.compile(r"epoch\s+(\d+)\s+loss ([\d.]+)\s+val: hand/no hand ([\d.]+)\s+(?:false hand on empty ([\d.]+)\s+)?"
                 r"mean IoU ([\d.]+)\s+IoU>0.5 ([\d.]+)")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#008300"]


def read_logs():
    clf, det = {}, {}
    for f in sorted(HERE.glob("runs_*.log")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        name = f.stem.replace("runs_", "")
        done = "\nexit" in text or "saved runs" in text
        c = [tuple(float(v) for v in m.groups()) for m in CLF.finditer(text)]
        d = [tuple(float(v) if v is not None else float("nan") for v in m.groups()) for m in DET.finditer(text)]
        if c:
            clf[name] = (c, done)
        if d:
            det[name] = (d, done)
    return clf, det


def draw(fig, axes):
    clf, det = read_logs()
    for ax in axes.flat:
        ax.clear()
        ax.grid(True, color="#e1e0d9")
        ax.set_xlabel("epoch")
    a1, a2, a3, a4 = axes.flat
    for k, (name, (rows, done)) in enumerate(clf.items()):
        e = [r[0] for r in rows]
        col = COLORS[k % len(COLORS)]
        tag = f"{name} ({'done' if done else 'running'}, epoch {int(e[-1])})"
        a1.plot(e, [r[3] for r in rows], color=col, lw=2, label=tag)
        a1.plot(e, [r[4] for r in rows], color=col, lw=1.5, ls="--")
        a2.plot(e, [r[1] for r in rows], color=col, lw=2, label=name)
    a1.set_title("Classifier: validation accuracy (solid = real photos, dashed = swapped backgrounds)", loc="left", fontsize=10)
    a1.set_ylim(0, 1.03)
    a2.set_title("Classifier: training loss", loc="left", fontsize=10)
    for k, (name, (rows, done)) in enumerate(det.items()):
        e = [r[0] for r in rows]
        a3.plot(e, [r[4] for r in rows], color=COLORS[0], lw=2, label="mean IoU")
        a3.plot(e, [r[5] for r in rows], color=COLORS[1], lw=2, label="IoU > 0.5")
        a3.plot(e, [r[2] for r in rows], color=COLORS[2], lw=2, label="hand / no hand right")
        a4.plot(e, [r[3] for r in rows], color=COLORS[1], lw=2, label="sees a hand in empty crops")
        a4.plot(e, [r[1] for r in rows], color=COLORS[0], lw=2, label="training loss")
        a3.set_title(f"Detector ({'done' if done else 'running'}, epoch {int(e[-1])}): validation", loc="left", fontsize=10)
    if not det:
        a3.set_title("Detector: no log yet", loc="left", fontsize=10)
    a3.set_ylim(0, 1.03)
    a4.set_title("Detector: false hands on empty crops, and loss", loc="left", fontsize=10)
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False)
    fig.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--every", type=float, default=5, help="seconds between refreshes")
    a = p.parse_args()
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
                                "axes.spines.top": False, "axes.spines.right": False})
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    fig.canvas.manager.set_window_title("Finger Lock: live training")
    while plt.fignum_exists(fig.number):
        draw(fig, axes)
        fig.tight_layout()
        plt.pause(a.every)


if __name__ == "__main__":
    main()
