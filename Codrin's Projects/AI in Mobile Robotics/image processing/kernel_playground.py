"""Draw your own structuring element and watch it sweep across the image.

Usage:
    python kernel_playground.py [INPUT]

Where threshold_ui.py applies morphology all at once, this shows the
mechanism: the image is reduced to a coarse grid of visible cells, and the
kernel crawls over it one position at a time, filling in the output as it
goes. At each stop the kernel's cells are tinted green where they agree
with the pixels underneath and red where they don't -- which is the whole
of what erosion, dilation and hit-or-miss are doing.

Kernel cells have three states, matching OpenCV's hit-or-miss convention:

    white  (+1)  must be foreground
    blue   (-1)  must be background
    grey   ( 0)  don't care

Erosion and dilation read only the +1 cells; hit-or-miss reads both.

Mouse:
    left click on the kernel     cycle a cell:  0 -> +1 -> -1 -> 0
    left click on the input      move the kernel there and pause
    left click Play/Step/Reset   the buttons on the toolbar

Keys:
    space  play / pause          n  single step        r  restart the sweep
    c      clear kernel          f  fill kernel (+1)   o  cross / plus shape
    s      save the output       q / Esc  quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from grayscale import DEFAULT_INPUT, read_image, to_grayscale, write_image

WINDOW = "kernel playground - space: play, n: step, r: reset, q: quit"

# Working resolutions, in cells across. Coarse enough to see the kernel
# straddle individual pixels; the last few are past that and just run fast.
RESOLUTIONS = (56, 84, 120, 170, 240, 340)
PANEL_WIDTH = 620  # target on-screen width of the input and output panels
# Below this on-screen cell size the separator lines cost more than they
# show, so they are dropped.
MIN_SEPARATOR_CELL = 6
KERNEL_CELL = 34  # on-screen size of one kernel-editor cell
MARGIN = 14
TOOLBAR_HEIGHT = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX

BACKGROUND = (24, 24, 28)
INK = (225, 225, 225)
MUTED = (150, 150, 150)
# Kernel cell states -> fill colour (BGR).
STATE_COLORS = {0: (70, 70, 74), 1: (235, 235, 235), -1: (170, 90, 60)}
AGREE = (90, 200, 110)
DISAGREE = (80, 80, 235)
UNVISITED = (55, 55, 58)

OPERATIONS = ("erosion", "dilation", "hit-or-miss")
BUTTONS = ("play", "step", "reset", "save")

# Filled in by layout(); maps a panel name to its (x, y, width, height).
PANELS: dict[str, tuple[int, int, int, int]] = {}


def build_grid(gray: np.ndarray, threshold: int, columns: int) -> np.ndarray:
    """Coarse boolean grid: threshold, then shrink to `columns` wide."""
    _, binary = cv2.threshold(gray, threshold - 1, 255, cv2.THRESH_BINARY)
    rows = max(1, round(columns * binary.shape[0] / binary.shape[1]))
    small = cv2.resize(binary, (columns, rows), interpolation=cv2.INTER_AREA)
    return small > 127


def cross_kernel(size: int) -> np.ndarray:
    """A plus sign of +1 cells -- the usual starting shape."""
    kernel = np.zeros((size, size), np.int8)
    kernel[size // 2, :] = 1
    kernel[:, size // 2] = 1
    return kernel


def evaluate(patch: np.ndarray, kernel: np.ndarray, operation: str):
    """Decide one output cell, and say which kernel cells drove the decision.

    Returns (output value, agreement array) where agreement is +1 where a
    kernel cell is satisfied, -1 where it is violated and 0 where the cell
    is don't-care and had no say.
    """
    agreement = np.zeros(kernel.shape, np.int8)
    wants_foreground = kernel == 1
    wants_background = kernel == -1

    if operation == "dilation":
        # Any overlap between the +1 cells and the foreground fires.
        hits = wants_foreground & patch
        agreement[hits] = 1
        agreement[wants_foreground & ~patch] = -1
        return bool(hits.any()), agreement

    # Erosion needs every +1 cell over foreground; hit-or-miss adds the -1
    # cells over background. Erosion simply has no -1 cells to check.
    if operation == "hit-or-miss":
        satisfied = (wants_foreground & patch) | (wants_background & ~patch)
        violated = (wants_foreground & ~patch) | (wants_background & patch)
    else:
        satisfied = wants_foreground & patch
        violated = wants_foreground & ~patch
    agreement[satisfied] = 1
    agreement[violated] = -1
    return not violated.any() and satisfied.any(), agreement


def neighbourhood(grid: np.ndarray, row: int, column: int, size: int) -> np.ndarray:
    """The size x size patch centred on (row, column), padded with background.

    Everything outside the image counts as background, which is the obvious
    reading but not quite cv2.MORPH_HITMISS's: OpenCV erodes both the image
    and its complement against the same constant border, so for it the
    outside is foreground and background at once. The two therefore differ
    on edge cells and agree everywhere inside.
    """
    return patch_from(np.pad(grid, size // 2), row, column, size)


def patch_from(padded: np.ndarray, row: int, column: int, size: int) -> np.ndarray:
    """Same patch, from a grid padded once up front.

    The sweep runs this per cell, and padding the whole grid each time cost
    more than every other part of a step put together.
    """
    return padded[row : row + size, column : column + size]


def draw_grid_panel(
    grid: np.ndarray, cell: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Render a boolean grid as blocks; `mask` cells are drawn as unvisited."""
    image = np.where(grid[..., None], np.uint8(240), np.uint8(15))
    image = np.repeat(np.repeat(image, cell, 0), cell, 1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if mask is not None:
        blocks = np.repeat(np.repeat(mask, cell, 0), cell, 1)
        image[blocks] = UNVISITED
    # Faint cell separators, so the reader can count pixels -- but at high
    # resolutions the cells are too small for them to mean anything.
    if cell >= MIN_SEPARATOR_CELL:
        image[::cell, :] = (45, 45, 48)
        image[:, ::cell] = (45, 45, 48)
    return image


def draw_kernel_editor(kernel: np.ndarray, agreement: np.ndarray | None) -> np.ndarray:
    """The clickable kernel grid, tinted by agreement when the sweep is live."""
    size = kernel.shape[0]
    panel = np.full((size * KERNEL_CELL, size * KERNEL_CELL, 3), BACKGROUND, np.uint8)
    for row in range(size):
        for column in range(size):
            state = int(kernel[row, column])
            colour = STATE_COLORS[state]
            if agreement is not None and agreement[row, column] != 0:
                colour = AGREE if agreement[row, column] > 0 else DISAGREE
            top_left = (column * KERNEL_CELL, row * KERNEL_CELL)
            bottom_right = (top_left[0] + KERNEL_CELL, top_left[1] + KERNEL_CELL)
            cv2.rectangle(panel, top_left, bottom_right, colour, -1)
            cv2.rectangle(panel, top_left, bottom_right, (20, 20, 22), 1)
            if state == -1:
                cv2.putText(
                    panel, "0", (top_left[0] + 12, top_left[1] + 23), FONT, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA,
                )
    # Mark the origin -- the cell that lands on the output pixel.
    centre = size // 2
    cv2.circle(
        panel,
        (centre * KERNEL_CELL + KERNEL_CELL // 2, centre * KERNEL_CELL + KERNEL_CELL // 2),
        4, (60, 160, 240), -1, cv2.LINE_AA,
    )
    return panel


def label(canvas: np.ndarray, text: str, x: int, y: int, colour=MUTED, scale=0.45):
    cv2.putText(canvas, text, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)


def draw_toolbar(canvas: np.ndarray, width: int, playing: bool, status: str):
    """Buttons across the top; records their rectangles in PANELS."""
    x = MARGIN
    for name in BUTTONS:
        text = "pause" if (name == "play" and playing) else name
        button_width = 78
        rectangle = (x, 8, button_width, TOOLBAR_HEIGHT - 16)
        PANELS[f"button:{name}"] = rectangle
        cv2.rectangle(
            canvas, (x, 8), (x + button_width, TOOLBAR_HEIGHT - 8), (70, 70, 76), -1
        )
        cv2.rectangle(
            canvas, (x, 8), (x + button_width, TOOLBAR_HEIGHT - 8), (120, 120, 126), 1
        )
        label(canvas, text, x + 12, TOOLBAR_HEIGHT - 16, INK)
        x += button_width + 8
    label(canvas, status, x + 12, TOOLBAR_HEIGHT - 16, MUTED)


def compose(state: dict) -> np.ndarray:
    """Lay the whole window out and return the frame to show."""
    # Never below 2px, or the kernel outline has nothing to sit on.
    cell = max(2, PANEL_WIDTH // state["grid"].shape[1])
    input_panel = draw_grid_panel(state["grid"], cell)
    output_panel = draw_grid_panel(state["output"], cell, mask=~state["visited"])
    kernel_panel = draw_kernel_editor(state["kernel"], state["agreement"])

    size = state["kernel"].shape[0]
    radius = size // 2
    row, column = state["row"], state["column"]
    # Outline the kernel's footprint on the input, and its origin on both.
    top_left = ((column - radius) * cell, (row - radius) * cell)
    cv2.rectangle(
        input_panel, top_left,
        (top_left[0] + size * cell, top_left[1] + size * cell),
        (60, 160, 240), 2,
    )
    for panel in (input_panel, output_panel):
        cv2.rectangle(
            panel, (column * cell, row * cell),
            ((column + 1) * cell, (row + 1) * cell), (60, 160, 240), 1,
        )

    grid_height = input_panel.shape[0]
    grid_width = input_panel.shape[1]
    top = TOOLBAR_HEIGHT + 22
    width = MARGIN * 3 + grid_width * 2
    width = max(width, MARGIN * 2 + kernel_panel.shape[1] + 260)
    height = top + grid_height + kernel_panel.shape[0] + 70
    canvas = np.full((height, width, 3), BACKGROUND, np.uint8)

    PANELS["input"] = (MARGIN, top, grid_width, grid_height)
    canvas[top : top + grid_height, MARGIN : MARGIN + grid_width] = input_panel
    output_x = MARGIN * 2 + grid_width
    PANELS["output"] = (output_x, top, grid_width, grid_height)
    canvas[top : top + grid_height, output_x : output_x + grid_width] = output_panel
    label(canvas, "input (thresholded)", MARGIN, top - 8)
    label(canvas, f"output: {state['operation']}", output_x, top - 8)

    kernel_top = top + grid_height + 30
    PANELS["kernel"] = (MARGIN, kernel_top, kernel_panel.shape[1], kernel_panel.shape[0])
    canvas[
        kernel_top : kernel_top + kernel_panel.shape[0],
        MARGIN : MARGIN + kernel_panel.shape[1],
    ] = kernel_panel
    label(canvas, "kernel - click to cycle", MARGIN, kernel_top - 8)

    # Legend and the current decision, beside the kernel editor.
    text_x = MARGIN + kernel_panel.shape[1] + 20
    lines = [
        "white +1  must be foreground",
        "blue  -1  must be background",
        "grey   0  don't care",
        f"position: row {row}, col {column} of {state['grid'].shape}",
        f"result here: {'FOREGROUND' if state['result'] else 'background'}",
        f"visited {int(state['visited'].sum())} / {state['visited'].size}",
    ]
    for index, line in enumerate(lines):
        colour = INK if index >= 3 else MUTED
        label(canvas, line, text_x, kernel_top + 18 + index * 22, colour)

    draw_toolbar(canvas, width, state["playing"], state["status"])
    return canvas


def inside(rectangle, x: int, y: int) -> bool:
    left, top, width, height = rectangle
    return left <= x < left + width and top <= y < top + height


def advance(state: dict, steps: int):
    """Compute the next `steps` output cells."""
    rows, columns = state["grid"].shape
    padded, kernel, operation = state["padded"], state["kernel"], state["operation"]
    size = kernel.shape[0]
    for _ in range(steps):
        row, column = state["row"], state["column"]
        result, agreement = evaluate(patch_from(padded, row, column, size), kernel, operation)
        state["output"][row, column] = result
        state["visited"][row, column] = True
        state["result"], state["agreement"] = result, agreement

        column += 1
        if column >= columns:
            column, row = 0, row + 1
        if row >= rows:
            row, state["playing"], state["status"] = 0, False, "sweep complete"
        state["row"], state["column"] = row, column


def reset_sweep(state: dict, status: str = "reset"):
    # Pad once here; the sweep reads slices of it a cell at a time.
    state["padded"] = np.pad(state["grid"], state["kernel"].shape[0] // 2)
    state["output"] = np.zeros_like(state["grid"])
    state["visited"] = np.zeros_like(state["grid"])
    state["row"] = state["column"] = 0
    state["agreement"] = None
    state["result"] = False
    state["status"] = status


def recompute_here(state: dict):
    """Refresh the tinting for the current position without advancing."""
    patch = neighbourhood(
        state["grid"], state["row"], state["column"], state["kernel"].shape[0]
    )
    state["result"], state["agreement"] = evaluate(
        patch, state["kernel"], state["operation"]
    )


def main(argv: list[str]) -> int:
    source = Path(argv[0]) if argv else DEFAULT_INPUT
    if not source.is_file():
        print(f"No such image: {source}", file=sys.stderr)
        return 1

    gray = to_grayscale(read_image(source))
    otsu = int(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])
    state = {
        "gray": gray,
        "grid": build_grid(gray, otsu, RESOLUTIONS[0]),
        "kernel": cross_kernel(3),
        "operation": "erosion",
        "playing": False,
        "status": "click Play",
        "source": source,
    }
    reset_sweep(state, "ready")
    recompute_here(state)
    print(f"{source.name}: grid {state['grid'].shape[1]}x{state['grid'].shape[0]} cells")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("threshold", WINDOW, otsu, 255, lambda _v: None)
    cv2.createTrackbar("operation", WINDOW, 0, len(OPERATIONS) - 1, lambda _v: None)
    cv2.createTrackbar("kernel size", WINDOW, 1, 4, lambda _v: None)  # -> 3,5,7,9,11
    cv2.createTrackbar("resolution", WINDOW, 0, len(RESOLUTIONS) - 1, lambda _v: None)
    cv2.createTrackbar("speed", WINDOW, 12, 60, lambda _v: None)

    pending: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))

    cv2.setMouseCallback(WINDOW, on_mouse)

    settings = None
    try:
        while True:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return 0

            threshold = cv2.getTrackbarPos("threshold", WINDOW)
            operation = OPERATIONS[cv2.getTrackbarPos("operation", WINDOW)]
            size = cv2.getTrackbarPos("kernel size", WINDOW) * 2 + 3
            columns = RESOLUTIONS[cv2.getTrackbarPos("resolution", WINDOW)]
            # Squared, so the low end of the slider still steps slowly enough
            # to follow while the top end can finish a 340-wide sweep at once.
            speed = max(1, cv2.getTrackbarPos("speed", WINDOW) ** 2 // 2)

            # A trackbar change invalidates the sweep so far.
            if settings != (threshold, operation, size, columns):
                if settings is None or settings[0] != threshold or settings[3] != columns:
                    state["grid"] = build_grid(gray, threshold, columns)
                if settings is not None and settings[2] != size:
                    state["kernel"] = cross_kernel(size)
                state["operation"] = operation
                reset_sweep(state, f"{operation}, {size}x{size} kernel, {columns} wide")
                recompute_here(state)
                settings = (threshold, operation, size, columns)

            while pending:
                x, y = pending.pop(0)
                handle_click(state, x, y, speed)

            if state["playing"]:
                advance(state, max(1, speed))

            cv2.imshow(WINDOW, compose(state))

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord(" "):
                state["playing"] = not state["playing"]
                state["status"] = "playing" if state["playing"] else "paused"
            elif key == ord("n"):
                state["playing"] = False
                advance(state, 1)
                state["status"] = "stepped"
            elif key == ord("r"):
                reset_sweep(state)
                recompute_here(state)
            elif key in (ord("c"), ord("f"), ord("o")):
                shape = state["kernel"].shape[0]
                if key == ord("c"):
                    state["kernel"] = np.zeros((shape, shape), np.int8)
                elif key == ord("f"):
                    state["kernel"] = np.ones((shape, shape), np.int8)
                else:
                    state["kernel"] = cross_kernel(shape)
                reset_sweep(state, "kernel changed")
                recompute_here(state)
            elif key == ord("s"):
                save_output(state)
    finally:
        cv2.destroyAllWindows()


def handle_click(state: dict, x: int, y: int, speed: int):
    """Route a click to a button, a kernel cell, or a position on the input."""
    for name in BUTTONS:
        if inside(PANELS.get(f"button:{name}", (0, 0, 0, 0)), x, y):
            if name == "play":
                state["playing"] = not state["playing"]
                state["status"] = "playing" if state["playing"] else "paused"
            elif name == "step":
                state["playing"] = False
                advance(state, 1)
                state["status"] = "stepped"
            elif name == "reset":
                reset_sweep(state)
                recompute_here(state)
            elif name == "save":
                save_output(state)
            return

    if "kernel" in PANELS and inside(PANELS["kernel"], x, y):
        left, top, _, _ = PANELS["kernel"]
        row = (y - top) // KERNEL_CELL
        column = (x - left) // KERNEL_CELL
        if row < state["kernel"].shape[0] and column < state["kernel"].shape[1]:
            # 0 -> +1 -> -1 -> 0
            state["kernel"][row, column] = {0: 1, 1: -1, -1: 0}[
                int(state["kernel"][row, column])
            ]
            reset_sweep(state, "kernel edited")
            recompute_here(state)
        return

    if "input" in PANELS and inside(PANELS["input"], x, y):
        left, top, width, _ = PANELS["input"]
        cell = width // state["grid"].shape[1]
        state["row"] = min((y - top) // cell, state["grid"].shape[0] - 1)
        state["column"] = min((x - left) // cell, state["grid"].shape[1] - 1)
        state["playing"] = False
        recompute_here(state)
        state["status"] = "moved here (paused)"


def save_output(state: dict):
    """Write the swept output at grid resolution, scaled up to stay readable."""
    source: Path = state["source"]
    image = (state["output"] * 255).astype(np.uint8)
    # Scale up so a coarse sweep still saves as a reasonably sized file.
    scale = max(1, 1200 // image.shape[1])
    image = np.repeat(np.repeat(image, scale, 0), scale, 1)
    target = source.with_name(f"{source.stem}_{state['operation']}_grid.png")
    write_image(target, image)
    state["status"] = f"saved {target.name}"
    print(state["status"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
