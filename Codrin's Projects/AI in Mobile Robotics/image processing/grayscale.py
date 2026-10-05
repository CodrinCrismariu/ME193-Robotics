"""Convert an image to grayscale with OpenCV.

Usage:
    python grayscale.py [INPUT] [OUTPUT]

Defaults to "Codrin Crismariu.png" in this folder, writing "<name>_gray.png"
next to it. Reading and writing go through numpy buffers rather than
cv2.imread/imwrite so that non-ASCII paths work on Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "Codrin Crismariu.png"


def read_image(path: Path) -> np.ndarray:
    """Load an image as BGR, tolerating unicode paths."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path} is not an image OpenCV can decode")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Encode by the output suffix and write, tolerating unicode paths."""
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"OpenCV cannot encode '{path.suffix}' images")
    buffer.tofile(path)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR -> single-channel luminance (ITU-R BT.601 weights)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def main(argv: list[str]) -> int:
    source = Path(argv[0]) if argv else DEFAULT_INPUT
    if len(argv) > 1:
        target = Path(argv[1])
    else:
        target = source.with_name(f"{source.stem}_gray{source.suffix or '.png'}")

    if not source.is_file():
        print(f"No such image: {source}", file=sys.stderr)
        return 1

    gray = to_grayscale(read_image(source))
    write_image(target, gray)
    print(f"{source.name} -> {target.name}  ({gray.shape[1]}x{gray.shape[0]}, 1 channel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
