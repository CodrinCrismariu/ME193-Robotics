"""End to end, the way the lock will run: camera frame -> hand detector -> crop -> letter classifier.

For every frame of the held-out test bursts, the detector looks for the hand; if it finds none the answer is
"none", otherwise the crop around its box goes to the classifier. This is compared with the classifier given the
labeled boxes (its best case), so the gap shows what detector mistakes cost.

Run from this folder (after train_detector.py and train_classifier.py):
    ..\\.venv\\Scripts\\python pipeline_eval.py                     (classifier from runs/randomized)
    ..\\.venv\\Scripts\\python pipeline_eval.py --classifier randomized_200
Writes runs/pipeline_<classifier>.json and figures/pipeline_confusion.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import augment
from data import DATA, LABELS, context_crop, load_rows, split_by_burst
from detector import IN_H, IN_W, HandDetector, decode
from models import LetterNet
from train_detector import SRC_H, SRC_W, view

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def classify_with_boxes(clf, frames_pil, boxes):
    crops = torch.from_numpy(np.stack([np.asarray(context_crop(im, b)) for im, b in zip(frames_pil, boxes)]))
    return clf(augment.eval_batch(crops.permute(0, 3, 1, 2).contiguous())).softmax(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", default="randomized_200")
    p.add_argument("--detector", default="detector")
    p.add_argument("--min-hand", type=float, default=0.5, help="detector probability needed to count as a hand")
    a = p.parse_args()
    rows = load_rows()
    part = split_by_burst(rows)
    test = [r for r in rows if part[r["id"]] == "test"]
    pil = [Image.open(DATA / "images" / f"{r['id']}.jpg").convert("RGB") for r in test]
    x = torch.from_numpy(np.stack([np.asarray(im) for im in pil])).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor([LABELS.index(r["label"]) for r in test])

    det = HandDetector()
    det.load_state_dict(torch.load(HERE / "runs" / a.detector / "model.pt", weights_only=False)["state_dict"])
    det.eval()
    clf = LetterNet()
    clf.load_state_dict(torch.load(HERE / "runs" / a.classifier / "model.pt", weights_only=False)["state_dict"])
    clf.eval()

    with torch.no_grad():
        img, _ = view(x, torch.zeros(len(x), 4), train=False)
        boxes, prob = decode(det(augment.normalize(img)))
    boxes = boxes * torch.tensor([SRC_W / IN_W, SRC_H / IN_H] * 2)
    found = prob > a.min_hand
    pred = torch.full((len(test),), LABELS.index("none"))
    fi = torch.nonzero(found).flatten().tolist()
    if fi:
        safe = [[max(0.0, b[0]), max(0.0, b[1]), min(SRC_W, b[2]), min(SRC_H, b[3])] for b in boxes[fi].tolist()]
        pred[fi] = classify_with_boxes(clf, [pil[i] for i in fi], safe).argmax(1)

    has = [i for i, r in enumerate(test) if r.get("box")]
    oracle = torch.full((len(test),), LABELS.index("none"))
    oracle[has] = classify_with_boxes(clf, [pil[i] for i in has], [test[i]["box"] for i in has]).argmax(1)

    acc = (pred == y).float().mean().item()
    acc_oracle = (oracle == y).float().mean().item()
    per = {l: (pred[y == k] == k).float().mean().item() for k, l in enumerate(LABELS) if (y == k).any()}
    res = {"classifier": a.classifier, "detector": a.detector, "n_test_frames": len(test), "end_to_end": acc,
           "with_labeled_boxes": acc_oracle, "per_label_end_to_end": per,
           "frames_detector_said_no_hand": int((~found).sum())}
    (HERE / "runs" / f"pipeline_{a.classifier}.json").write_text(json.dumps(res, indent=1))
    print(f"{len(test)} test frames from held-out bursts")
    print(f"  end to end (detector box -> classifier): {acc:.1%} right")
    print(f"  classifier with the labeled boxes:       {acc_oracle:.1%} right")
    print("  per label (end to end): " + ", ".join(f"{l} {v:.0%}" for l, v in per.items()))

    k = len(LABELS)
    cm = np.zeros((k, k), int)
    for t, q in zip(y.tolist(), pred.tolist()):
        cm[t, q] += 1
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.imshow(cm, cmap="Oranges")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(k), LABELS)
    ax.set_yticks(range(k), LABELS)
    ax.set_xlabel("predicted (detector -> classifier)")
    ax.set_ylabel("true")
    ax.set_title(f"End to end on held-out bursts: {acc:.0%} right", loc="left", fontsize=10)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "pipeline_confusion.png", dpi=140, bbox_inches="tight")
    print("saved figures/pipeline_confusion.png")


if __name__ == "__main__":
    main()
