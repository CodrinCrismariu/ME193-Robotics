"""Make a hand outline (mask) for every photo with a box. Used only during training, to paste the hand onto
random backgrounds (domain randomization) and to test whether a model reads the hand or the background.

How each mask is made:
  1. A pretrained hand tracker (MediaPipe Hands) finds the 21 hand landmarks inside the labeled box.
  2. The landmarks are drawn as a skeleton (thick lines along the finger bones, the palm filled in):
     those pixels are certainly hand.
  3. GrabCut (a classic graph-cut segmentation) grows the true outline from that skeleton, using everything
     outside the box as certain background.
  4. A skin-color check drops pixels whose color is far from the hand's own skin (measured on the skeleton),
     which clears background left inside the openings of C, O and D.
  If the tracker finds no hand in the box, GrabCut starts from the box alone (no skin check).

Run from this folder:   ..\\.venv\\Scripts\\python make_hand_masks.py          (skips masks that are up to date)
                        ..\\.venv\\Scripts\\python make_hand_masks.py --redo
Writes dataset/masks/<id>.png (white = hand), dataset/masks/index.json, figures/hand_masks_check.png
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from prelabel_boxes import landmarker

HERE = Path(__file__).resolve().parent
LABELS = ["C", "O", "D", "R", "I", "N", "none"]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PALM = [0, 1, 5, 9, 13, 17]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def landmarks_in_box(det, rgb, box):
    """Pixel landmarks (21x2) of the tracked hand that best matches the labeled box, or None."""
    import mediapipe as mp
    h, w = rgb.shape[:2]
    side = max(w, h)
    ox, oy = (side - w) // 2, (side - h) // 2
    sq = np.pad(rgb, ((oy, side - h - oy), (ox, side - w - ox), (0, 0)), mode="edge")
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(sq)))
    best, best_iou = None, 0.3
    for lms in res.hand_landmarks:
        pts = np.array([[p.x * side - ox, p.y * side - oy] for p in lms])
        lb = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        o = iou(lb, box)
        if o > best_iou:
            best, best_iou = pts, o
    return best


def mask_for(rgb, box, pts):
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    m = 15                                                           # crop with a margin of certain background
    cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    crop = np.ascontiguousarray(cv2.cvtColor(rgb[cy0:cy1, cx0:cx1], cv2.COLOR_RGB2BGR))
    gc = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    gc[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = cv2.GC_PR_BGD          # inside the box: maybe hand
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    if pts is not None:
        p = (pts - [cx0, cy0]).astype(np.int32)
        palm = np.linalg.norm(pts[9] - pts[0]) + 1e-6
        thick = max(3, int(0.22 * palm))
        sure, maybe = np.zeros_like(gc), np.zeros_like(gc)
        for a, b in BONES:
            cv2.line(sure, tuple(p[a]), tuple(p[b]), 1, max(2, thick // 2))
            cv2.line(maybe, tuple(p[a]), tuple(p[b]), 1, int(thick * 1.8))
        cv2.fillPoly(sure, [p[PALM]], 1)
        cv2.fillPoly(maybe, [p[PALM]], 1)
        inside = gc == cv2.GC_PR_BGD
        gc[(maybe == 1) & inside] = cv2.GC_PR_FGD
        gc[(sure == 1) & inside] = cv2.GC_FGD
        cv2.grabCut(crop, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        seed = sure
    else:
        rect = (x0 - cx0 + 2, y0 - cy0 + 2, max(4, x1 - x0 - 4), max(4, y1 - y0 - 4))
        gc[:] = 0
        cv2.grabCut(crop, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        seed = None
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 4)
    if n > 1:                                                        # keep the hand blob(s): touching the skeleton, or the biggest
        keep = set(np.unique(lab[seed == 1])) - {0} if seed is not None else set()
        if not keep:
            keep = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
        fg = np.isin(lab, list(keep)).astype(np.uint8)
    out = np.zeros((h, w), np.uint8)
    out[cy0:cy1, cx0:cx1] = fg * 255
    return skin_filter(rgb, out, pts) if pts is not None else out


def skin_filter(rgb, mask, pts, thresh=3.2):
    """Drop mask pixels whose color is far from this hand's own skin color (measured on the landmark skeleton,
    in Lab color space, as a Mahalanobis distance). This clears background that GrabCut kept inside the
    openings of C, O and D."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    palm = np.linalg.norm(pts[9] - pts[0]) + 1e-6
    sure = np.zeros(mask.shape, np.uint8)
    p = pts.astype(np.int32)
    for a, b in BONES:
        cv2.line(sure, tuple(p[a]), tuple(p[b]), 1, max(2, int(0.11 * palm)))
    cv2.fillPoly(sure, [p[PALM]], 1)
    sure &= (mask > 0).astype(np.uint8)
    s = lab[sure == 1]
    if len(s) < 30:
        return mask
    mu, cov = s.mean(0), np.cov(s.T) + np.eye(3) * 4
    d = lab - mu
    dist = np.sqrt(np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d))
    keep = ((mask > 0) & ((dist < thresh) | (sure == 1))).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab_cc, _, _ = cv2.connectedComponentsWithStats(keep, 4)
    touch = set(np.unique(lab_cc[sure == 1])) - {0}
    return np.isin(lab_cc, list(touch)).astype(np.uint8) * 255 if touch else keep * 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "dataset"))
    p.add_argument("--redo", action="store_true")
    a = p.parse_args()
    data = Path(a.data)
    rows = [json.loads(l) for l in (data / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("box")]
    (data / "masks").mkdir(exist_ok=True)
    idx_path = data / "masks" / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    det = landmarker(0.3)
    made = {"skeleton + GrabCut": 0, "box + GrabCut": 0, "up to date": 0}
    for i, r in enumerate(rows):
        if not a.redo and index.get(r["id"], {}).get("box") == r["box"] and (data / "masks" / f"{r['id']}.png").exists():
            made["up to date"] += 1
            continue
        rgb = np.asarray(Image.open(data / "images" / f"{r['id']}.jpg").convert("RGB"))
        pts = landmarks_in_box(det, rgb, r["box"])
        Image.fromarray(mask_for(rgb, r["box"], pts)).save(data / "masks" / f"{r['id']}.png")
        how = "skeleton + GrabCut" if pts is not None else "box + GrabCut"
        index[r["id"]] = {"box": r["box"], "method": how}
        made[how] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}")
    idx_path.write_text(json.dumps(index))
    print(", ".join(f"{v} {k}" for k, v in made.items()))

    # a quick look: crops around the hand with the mask tinted
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    rng = random.Random(1)
    fig, axes = plt.subplots(len(LABELS), 8, figsize=(13, 1.75 * len(LABELS)))
    for i, l in enumerate(LABELS):
        rl = [r for r in rows if r["label"] == l]
        for j, r in enumerate(rng.sample(rl, min(8, len(rl)))):
            img = np.asarray(Image.open(data / "images" / f"{r['id']}.jpg").convert("RGB"), np.float32) / 255
            m = np.asarray(Image.open(data / "masks" / f"{r['id']}.png"), np.float32)[..., None] / 255
            x0, y0, x1, y1 = r["box"]
            tinted = img * (1 - 0.45 * m) + np.array([0.91, 0.45, 0.23]) * 0.45 * m
            axes[i, j].imshow(tinted[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8])
        for j in range(8):
            axes[i, j].axis("off")
        axes[i, 0].set_title(l, loc="left", fontweight="bold")
    fig.suptitle("Hand masks (orange) on random photos", x=0.01, ha="left", fontweight="bold")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "hand_masks_check.png", dpi=100, bbox_inches="tight")
    print("saved figures/hand_masks_check.png")


if __name__ == "__main__":
    main()
