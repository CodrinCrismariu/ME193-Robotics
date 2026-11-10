"""Download the Sign Language Digits dataset and turn digits 1-5 into a small training file.

Dataset: Arda Mavi, "Sign Language Digits Dataset" (Apache-2.0), 100x100 RGB photos of hands
from 218 students, digits 0-9. Digits 1-5 are made by holding up 1-5 fingers, so they double
as a finger-count dataset. (0 is an "O" shape and 6-9 are three-finger combinations, so we skip them.)
https://github.com/ardamavi/Sign-Language-Digits-Dataset

Output: data/fingers_64.npz with 64x64 grayscale images split 70% train / 15% validation / 15% test.

Run:  python prepare_data.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RAW_ZIP = HERE / "data" / "raw" / "sign-digits.zip"
OUT = HERE / "data" / "fingers_64.npz"
URL = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip"

CLASSES = [1, 2, 3, 4, 5]          # number of fingers; the folder names in the dataset
IMG_SIZE = 64                      # images are shrunk from 100x100 to 64x64
SPLIT = (0.70, 0.15, 0.15)         # train / validation / test
SEED = 0


def load_images():
    if not RAW_ZIP.exists():
        RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, RAW_ZIP)
    images, labels = [], []
    with zipfile.ZipFile(RAW_ZIP) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if len(parts) == 4 and parts[1] == "Dataset" and name.lower().endswith(".jpg"):
                fingers = int(parts[2])
                if fingers in CLASSES:
                    img = Image.open(io.BytesIO(z.read(name))).convert("L")       # grayscale
                    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
                    images.append(np.asarray(img, dtype=np.uint8))
                    labels.append(CLASSES.index(fingers))                       # 0..4
    return np.stack(images), np.array(labels, dtype=np.int64)


def split(images, labels):
    """Stratified split: every class is divided 70/15/15, so all three sets have the same class mix."""
    rng = np.random.default_rng(SEED)
    parts = {"train": [], "val": [], "test": []}
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_train = round(SPLIT[0] * len(idx))
        n_val = round(SPLIT[1] * len(idx))
        parts["train"] += list(idx[:n_train])
        parts["val"] += list(idx[n_train:n_train + n_val])
        parts["test"] += list(idx[n_train + n_val:])
    return {k: rng.permutation(v) for k, v in parts.items()}


def main():
    images, labels = load_images()
    parts = split(images, labels)
    arrays = {}
    for name, idx in parts.items():
        arrays[f"x_{name}"] = images[idx]
        arrays[f"y_{name}"] = labels[idx]
        counts = np.bincount(labels[idx], minlength=len(CLASSES))
        print(f"{name:5s}: {len(idx):4d} images   per class {dict(zip(CLASSES, counts.tolist()))}")
    np.savez_compressed(OUT, classes=np.array(CLASSES), **arrays)
    print(f"saved {OUT.relative_to(HERE)}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
