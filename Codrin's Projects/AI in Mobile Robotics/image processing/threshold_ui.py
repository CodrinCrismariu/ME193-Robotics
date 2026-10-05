"""Interactive binarization plus binary morphology, driven by sliders.

Usage:
    python threshold_ui.py [INPUT]

The pipeline is: grayscale -> threshold -> (optional) morphology operation.
Pixels darker than the threshold become 0 (black), the rest 255 (white);
morphology treats white as the foreground, so flip "invert" to operate on
the dark regions instead. Sliders:

    threshold   0-255 cut between the two levels (starts at Otsu's value)
    invert      swap black and white after thresholding
    operation   which morphology operation to apply (see OPERATIONS)
    shape       structuring element: rect, cross, ellipse
    size        structuring element size, forced odd (1..21)
    iterations  how many times to apply it / spur length for prune
    hitmiss     which pattern the hit-or-miss operation looks for

A toolbar across the top of the window carries a Save button and the
current settings. Keys do the same and a little more:

    s      save the image on screen, at the resolution it was computed at
    r      reset the threshold to Otsu's automatic value
    q/Esc  quit

Results are cached, so only a slider change costs a recomputation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

import morphology as morph
from grayscale import DEFAULT_INPUT, read_image, to_grayscale, write_image

WINDOW = "binarize + morphology - s: save, r: otsu, q: quit"
MAX_DISPLAY_WIDTH = 1100
# Thinning and pruning are iterative numpy passes over the whole frame, and
# the iteration count grows with the image, so full resolution costs ~30s per
# change -- unusable on a slider. Downscale first for those two only.
THIN_MAX_WIDTH = 600

# name -> callable(binary image, **params). Every entry accepts the same
# keywords so the dispatch below stays a single call.
OPERATIONS = {
    "none": lambda img, **_: img,
    "erosion": lambda img, shape, size, iters, **_: morph.erode(img, shape, size, iters),
    "dilation": lambda img, shape, size, iters, **_: morph.dilate(img, shape, size, iters),
    "opening": lambda img, shape, size, iters, **_: morph.opening(img, shape, size, iters),
    "closing": lambda img, shape, size, iters, **_: morph.closing(img, shape, size, iters),
    "hit-or-miss": lambda img, pattern, **_: morph.hit_or_miss(img, pattern),
    "boundary (A - erode)": lambda img, shape, size, iters, **_: morph.boundary(
        img, shape, size, iters
    ),
    "outer boundary (dilate - A)": lambda img, shape, size, iters, **_: (
        morph.outer_boundary(img, shape, size, iters)
    ),
    "gradient (dilate - erode)": lambda img, shape, size, iters, **_: morph.gradient(
        img, shape, size, iters
    ),
    "skeleton (thinning)": lambda img, **_: morph.thin(img),
    "prune (thin then despur)": lambda img, iters, **_: morph.prune(
        morph.thin(img), iters
    ),
}
OPERATION_NAMES = list(OPERATIONS)
SLOW_OPERATIONS = {"skeleton (thinning)", "prune (thin then despur)"}

# Toolbar geometry, in pixels: a strip drawn above the image carrying the
# Save button. cv2.createButton needs a Qt build of OpenCV and this wheel is
# Win32UI, so the button is drawn into the frame and clicked via a mouse
# callback instead.
TOOLBAR_HEIGHT = 44
SAVE_BUTTON = (12, 8, 120, 28)  # x, y, width, height
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONFIRM_SECONDS = 1.5

TRACKBARS = {
    "threshold": 255,
    "invert": 1,
    "operation": len(OPERATION_NAMES) - 1,
    "shape": len(morph.SHAPE_NAMES) - 1,
    "size": 21,
    "iterations": 10,
    "hitmiss": len(morph.HITMISS_NAMES) - 1,
}


def binarize(gray: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
    """Two-level image: below `threshold` -> 0, at or above -> 255."""
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    # OpenCV compares strictly greater than `thresh`, so step back by one to
    # make `threshold` itself land in the bright band.
    _, binary = cv2.threshold(gray, threshold - 1, 255, mode)
    return binary


def otsu_threshold(gray: np.ndarray) -> int:
    """The threshold Otsu's method picks for this image."""
    value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return int(value)


def scale_to_width(image: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink to `max_width` if wider, nearest-neighbour to stay two-level."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(
        image, (max_width, round(height * scale)), interpolation=cv2.INTER_NEAREST
    )


def apply_pipeline(gray: np.ndarray, params: dict) -> np.ndarray:
    """Threshold, then run the selected morphology operation."""
    binary = binarize(gray, params["threshold"], bool(params["invert"]))
    name = OPERATION_NAMES[params["operation"]]
    if name in SLOW_OPERATIONS:
        binary = scale_to_width(binary, THIN_MAX_WIDTH)
    return OPERATIONS[name](
        binary,
        shape=morph.SHAPE_NAMES[params["shape"]],
        size=max(1, params["size"]),
        iters=max(1, params["iterations"]),
        pattern=morph.HITMISS_NAMES[params["hitmiss"]],
    )


def read_params() -> dict:
    """Current position of every trackbar."""
    return {name: cv2.getTrackbarPos(name, WINDOW) for name in TRACKBARS}


def describe(params: dict, result: np.ndarray, elapsed: float) -> str:
    """One status line, printed whenever the parameters change."""
    name = OPERATION_NAMES[params["operation"]]
    size = max(1, params["size"]) | 1
    iterations = max(1, params["iterations"])
    detail = f"threshold {params['threshold']}"
    if params["invert"]:
        detail += " inverted"
    if name != "none":
        detail += f" | {name}"
    if name == "hit-or-miss":
        pattern = morph.HITMISS_NAMES[params["hitmiss"]]
        detail += f" [{pattern}]: {(result > 0).sum()} matches"
    elif name in SLOW_OPERATIONS:
        detail += f" @ {result.shape[1]}x{result.shape[0]}"
        if name.startswith("prune"):
            detail += f", spurs <= {iterations}px"
    elif name != "none":
        detail += f" {morph.SHAPE_NAMES[params['shape']]} {size}px x{iterations}"
    return f"{detail}  [{elapsed * 1000:.0f} ms]"


def save(source: Path, params: dict, result: np.ndarray) -> str:
    """Write the current frame beside the input, named after the settings."""
    name = OPERATION_NAMES[params["operation"]].split()[0].replace("-", "")
    target = source.with_name(
        f"{source.stem}_{name}{params['threshold']}{source.suffix or '.png'}"
    )
    write_image(target, result)
    print(f"saved {target.name} ({result.shape[1]}x{result.shape[0]})")
    return f"saved {target.name}"


def hit_save_button(x: int, y: int) -> bool:
    """Whether a click at window coordinates (x, y) landed on Save."""
    left, top, width, height = SAVE_BUTTON
    return left <= x < left + width and top <= y < top + height


def compose_frame(result: np.ndarray, status: str, pressed: bool) -> np.ndarray:
    """Stack the toolbar above the image; the result itself stays untouched.

    Only this composite is displayed -- saving writes `result`, so the
    toolbar never ends up in a saved file.
    """
    image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    toolbar = np.full((TOOLBAR_HEIGHT, image.shape[1], 3), 32, np.uint8)

    left, top, width, height = SAVE_BUTTON
    fill = (90, 140, 90) if pressed else (70, 70, 70)
    cv2.rectangle(toolbar, (left, top), (left + width, top + height), fill, -1)
    cv2.rectangle(toolbar, (left, top), (left + width, top + height), (190, 190, 190), 1)
    cv2.putText(
        toolbar, "Save (s)", (left + 16, top + 19), FONT, 0.5, (240, 240, 240), 1,
        cv2.LINE_AA,
    )

    # Trim the status to whatever fits beside the button.
    available = image.shape[1] - (left + width) - 24
    text = status[: max(0, available // 8)]
    cv2.putText(
        toolbar, text, (left + width + 16, top + 19), FONT, 0.45, (200, 200, 200), 1,
        cv2.LINE_AA,
    )
    return np.vstack([toolbar, image])


def main(argv: list[str]) -> int:
    source = Path(argv[0]) if argv else DEFAULT_INPUT
    if not source.is_file():
        print(f"No such image: {source}", file=sys.stderr)
        return 1

    gray = to_grayscale(read_image(source))
    initial = otsu_threshold(gray)
    print(f"{source.name}: {gray.shape[1]}x{gray.shape[0]}, Otsu threshold {initial}")
    print(morph.skeleton_ops_note() + f"; they run at <= {THIN_MAX_WIDTH}px wide")
    for index, name in enumerate(OPERATION_NAMES):
        print(f"  operation {index}: {name}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    # The trackbars hold the state; the loop reads them and recomputes on change.
    for name, maximum in TRACKBARS.items():
        cv2.createTrackbar(name, WINDOW, 0, maximum, lambda _value: None)
    cv2.setTrackbarPos("threshold", WINDOW, initial)
    cv2.setTrackbarPos("size", WINDOW, 3)
    cv2.setTrackbarPos("iterations", WINDOW, 1)

    # Clicks arrive on the callback thread, so it only records the request
    # and the loop below does the saving.
    clicks = {"save": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and hit_save_button(x, y):
            clicks["save"] = True

    cv2.setMouseCallback(WINDOW, on_mouse)

    cached = None
    status = ""
    confirmed_until = 0.0
    try:
        while True:
            # A closed window (title-bar X) leaves the trackbars unreadable.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return 0

            params = read_params()
            if cached is None or cached[0] != params:
                started = time.perf_counter()
                result = apply_pipeline(gray, params)
                cached = (params, result)
                status = describe(params, result, time.perf_counter() - started)
                print(status)
                confirmed_until = 0.0

            pressed = time.monotonic() < confirmed_until
            frame = compose_frame(
                scale_to_width(cached[1], MAX_DISPLAY_WIDTH), status, pressed
            )
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("r"):
                cv2.setTrackbarPos("threshold", WINDOW, initial)
            if key == ord("s") or clicks["save"]:
                clicks["save"] = False
                status = save(source, params, cached[1])
                confirmed_until = time.monotonic() + CONFIRM_SECONDS
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
