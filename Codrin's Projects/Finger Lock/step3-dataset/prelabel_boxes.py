"""Pre-fill the hand boxes with a pretrained hand tracker (MediaPipe Hands), before checking them by hand.

Every photo whose box has not been checked yet gets a box around the 21 hand landmarks the tracker finds,
with a margin so the fingertips and the edge of the palm are inside. Checked boxes are never touched.
Afterwards, check and fix every box in the labeler (http://localhost:8000/label.html).

Run from this folder, while you are not capturing photos:
    ..\\.venv\\Scripts\\python prelabel_boxes.py
    ..\\.venv\\Scripts\\python prelabel_boxes.py --redo     (also redo unchecked boxes that already have one)

The tracker's model file (hand_landmarker.task, 7.8 MB) is downloaded next to this script on the first run.
"""
import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "dataset"
MODEL = HERE / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def landmarker(min_conf):
    if not MODEL.exists():
        print(f"downloading the hand tracker model to {MODEL.name}")
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf)
    return vision.HandLandmarker.create_from_options(opts)


def box_from_landmarks(lms, w, h, margin):
    xs = np.array([p.x for p in lms]) * w
    ys = np.array([p.y for p in lms]) * h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = margin * (x1 - x0) + 3, margin * (y1 - y0) + 3     # landmarks sit at joint centers: pad to the skin
    return [int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx + 1)), int(min(h, y1 + my + 1))]


def find_box(det, rgb, margin):
    """Biggest hand's box in pixels of rgb, or None. The tracker is most accurate on square images,
    so the photo is padded to a square first and the box shifted back."""
    h, w = rgb.shape[:2]
    side = max(w, h)
    ox, oy = (side - w) // 2, (side - h) // 2
    sq = np.pad(rgb, ((oy, side - h - oy), (ox, side - w - ox), (0, 0)), mode="edge")
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(sq)))
    boxes = []
    for lms in res.hand_landmarks:
        b = box_from_landmarks(lms, side, side, margin)
        boxes.append([max(0, b[0] - ox), max(0, b[1] - oy), min(w, b[2] - ox), min(h, b[3] - oy)])
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--redo", action="store_true", help="also replace unchecked boxes that already exist")
    p.add_argument("--min-conf", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.12)
    a = p.parse_args()
    data = Path(a.data)
    index = data / "samples.jsonl"
    if not index.exists():
        print(f"No photos yet in {data}. Collect some with the collector (http://localhost:8000) first, then run this again.")
        return
    rows = [json.loads(l) for l in index.read_text(encoding="utf-8").splitlines() if l.strip()]
    det = landmarker(a.min_conf)
    stats = Counter()
    for r in rows:
        if r.get("checked"):
            stats["already checked (left alone)"] += 1
            continue
        if r.get("box") and not a.redo:
            stats["not checked, already had a box (left alone)"] += 1
            continue
        path = data / "images" / f"{r['id']}.jpg"
        if not path.exists():
            stats["image missing"] += 1
            continue
        r["box"] = find_box(det, np.asarray(Image.open(path).convert("RGB")), a.margin)
        stats[f"{'box found' if r['box'] else 'no hand found'} ({r['label']})"] += 1
    tmp = index.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(index)
    for k, v in sorted(stats.items()):
        print(f"{v:5d}  {k}")
    print("Now check every box in http://localhost:8000/label.html (Which photos: Not checked yet).")


if __name__ == "__main__":
    main()
