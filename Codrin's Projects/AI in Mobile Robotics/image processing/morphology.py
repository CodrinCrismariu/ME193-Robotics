"""Binary morphology operations for the threshold UI.

Every function takes and returns a uint8 image whose pixels are 0 or 255,
with white (255) as the foreground -- the convention OpenCV's morphology
assumes. Flip the UI's "invert" slider to treat dark regions as foreground
instead.

Erosion, dilation, opening, closing and hit-or-miss delegate to OpenCV.
Thinning and pruning are implemented here: they need a topology-preserving
iterative algorithm that lives in opencv-contrib (cv2.ximgproc.thinning),
which the plain opencv-python wheel does not ship.
"""

from __future__ import annotations

import cv2
import numpy as np

SHAPES = {
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
    "ellipse": cv2.MORPH_ELLIPSE,
}
SHAPE_NAMES = list(SHAPES)

# Hit-or-miss structuring elements: 1 = must be foreground, -1 = must be
# background, 0 = don't care.
HITMISS_PATTERNS: dict[str, np.ndarray] = {
    "isolated point": np.array(
        [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8
    ),
    "line endpoint": np.array([[-1, -1, -1], [-1, 1, -1], [0, 1, 0]], dtype=np.int8),
    "top-left corner": np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, 0]], dtype=np.int8),
    "interior point": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8),
}
HITMISS_NAMES = list(HITMISS_PATTERNS)

# Offsets of the 8 neighbours, as (row shift, column shift).
_NEIGHBOURS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def kernel(shape: str, size: int) -> np.ndarray:
    """Structuring element of the given shape, forced to an odd size."""
    size = max(1, size | 1)
    return cv2.getStructuringElement(SHAPES[shape], (size, size))


def erode(image, shape="rect", size=3, iterations=1):
    """Shrink foreground regions; drops small bright specks, thins bridges."""
    return cv2.erode(image, kernel(shape, size), iterations=iterations)


def dilate(image, shape="rect", size=3, iterations=1):
    """Grow foreground regions; fills small gaps, joins nearby components."""
    return cv2.dilate(image, kernel(shape, size), iterations=iterations)


def opening(image, shape="rect", size=3, iterations=1):
    """Erode then dilate: removes small objects, keeps larger shapes' size."""
    return cv2.morphologyEx(
        image, cv2.MORPH_OPEN, kernel(shape, size), iterations=iterations
    )


def closing(image, shape="rect", size=3, iterations=1):
    """Dilate then erode: fills small holes and gaps, smooths boundaries."""
    return cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, kernel(shape, size), iterations=iterations
    )


def hit_or_miss(image, pattern="isolated point"):
    """Mark pixels whose neighbourhood matches a foreground/background pattern.

    The single int8 kernel encodes both structuring elements of the classic
    definition: the +1 entries are the "hit" element, the -1 entries the
    "miss" element applied to the complement.
    """
    return cv2.morphologyEx(image, cv2.MORPH_HITMISS, HITMISS_PATTERNS[pattern])


def boundary(image, shape="rect", size=3, iterations=1):
    """Inner boundary: A - erosion(A), a one-pixel outline of each region."""
    return cv2.subtract(image, erode(image, shape, size, iterations))


def outer_boundary(image, shape="rect", size=3, iterations=1):
    """Outer boundary: dilation(A) - A, an outline just outside each region."""
    return cv2.subtract(dilate(image, shape, size, iterations), image)


def gradient(image, shape="rect", size=3, iterations=1):
    """Morphological gradient: dilation(A) - erosion(A), a thicker outline."""
    return cv2.subtract(
        dilate(image, shape, size, iterations), erode(image, shape, size, iterations)
    )


def _neighbour_stack(padded: np.ndarray) -> list[np.ndarray]:
    """The 8 neighbours of every pixel, in clockwise order from north."""
    height, width = padded.shape[0] - 2, padded.shape[1] - 2
    return [
        padded[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
        for dr, dc in _NEIGHBOURS
    ]


def _thinning_pass(mask: np.ndarray, second: bool) -> np.ndarray:
    """One Zhang-Suen sub-iteration; returns the pixels to delete."""
    padded = np.pad(mask, 1)
    p = _neighbour_stack(padded)  # p[0] = north, clockwise to p[7] = north-west
    neighbours = sum(n.astype(np.uint8) for n in p)
    # Transitions from background to foreground around the ring.
    transitions = sum(
        (~p[i] & p[(i + 1) % 8]).astype(np.uint8) for i in range(8)
    )
    if second:
        corners = (~(p[0] & p[2] & p[6])) & (~(p[0] & p[4] & p[6]))
    else:
        corners = (~(p[0] & p[2] & p[4])) & (~(p[2] & p[4] & p[6]))
    return mask & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & corners


def thin(image, max_iterations: int = 100) -> np.ndarray:
    """Zhang-Suen skeletonization: 1-pixel-wide centrelines, topology kept.

    Iterates two complementary deletion passes until nothing changes, so
    `max_iterations` is only a safety stop for pathological inputs.
    """
    mask = image > 0
    for _ in range(max_iterations):
        changed = False
        for second in (False, True):
            removable = _thinning_pass(mask, second)
            if removable.any():
                mask = mask & ~removable
                changed = True
        if not changed:
            break
    return (mask * 255).astype(np.uint8)


def _endpoints(mask: np.ndarray) -> np.ndarray:
    """Foreground pixels with exactly one foreground neighbour.

    Equivalent to the union of the eight endpoint hit-or-miss elements, but
    a single neighbour count instead of eight passes.
    """
    padded = np.pad(mask, 1)
    neighbours = sum(n.astype(np.uint8) for n in _neighbour_stack(padded))
    return mask & (neighbours == 1)


def prune(image, iterations: int = 3, regrow: bool = True) -> np.ndarray:
    """Remove spurs up to `iterations` pixels long from a skeleton.

    Gonzalez's four-step pruning: strip endpoints `iterations` times, then
    regrow the surviving branches by the same amount (dilating the stripped
    skeleton's endpoints inside the original) so that only the short spurs
    are actually lost, not a fixed length off every branch.

    That regrowth is a dilation constrained to the original, so it also
    creeps a little way into any branch within `iterations` pixels of a
    surviving endpoint -- a spur that close can come partly back. Pass
    regrow=False for the blunt version: strip endpoints and stop, which
    always removes the spurs but shortens every branch by `iterations`.
    """
    original = image > 0
    stripped = original.copy()
    for _ in range(iterations):
        stripped = stripped & ~_endpoints(stripped)
    if not regrow:
        return (stripped * 255).astype(np.uint8)

    regrown = _endpoints(stripped)
    cross = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        grown = cv2.dilate((regrown * 255).astype(np.uint8), cross) > 0
        regrown = grown & original

    return ((stripped | regrown) * 255).astype(np.uint8)


def skeleton_ops_note() -> str:
    """One line on why thinning is hand-rolled -- printed by the UI."""
    return "thin/prune run in numpy (cv2.ximgproc is contrib-only), so they are slower"
