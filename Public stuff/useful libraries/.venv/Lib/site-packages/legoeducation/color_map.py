"""App-aligned LEGO color constants and firmware↔App translation.

The RPC firmware uses one numbering scheme for colors; the LEGO Education
Block App uses a different scheme. This module provides App-aligned constants
and bidirectional translation so the Python API matches the App.
"""

# Import firmware constants and notification types for internal use only
from .rpc_message import (
    LEGO_COLOR_NONE as _FW_NONE,
    LEGO_COLOR_BLACK as _FW_BLACK,
    LEGO_COLOR_MAGENTA as _FW_MAGENTA,
    LEGO_COLOR_PURPLE as _FW_PURPLE,
    LEGO_COLOR_BLUE as _FW_BLUE,
    LEGO_COLOR_AZURE as _FW_AZURE,
    LEGO_COLOR_TURQUOISE as _FW_TURQUOISE,
    LEGO_COLOR_GREEN as _FW_GREEN,
    LEGO_COLOR_YELLOW as _FW_YELLOW,
    LEGO_COLOR_ORANGE as _FW_ORANGE,
    LEGO_COLOR_RED as _FW_RED,
    LEGO_COLOR_WHITE as _FW_WHITE,
    ColorSensorNotification as _ColorSensorNotification,
    CardNotification as _CardNotification,
)

# ── App-aligned color constants ──
LEGO_COLOR_NOCOLOR = 0
LEGO_COLOR_RED = 1
LEGO_COLOR_YELLOW = 2
LEGO_COLOR_BLUE = 3
LEGO_COLOR_TEAL = 4
LEGO_COLOR_GREEN = 5
LEGO_COLOR_PURPLE = 6
LEGO_COLOR_WHITE = 7
LEGO_COLOR_MAGENTA = 8
LEGO_COLOR_ORANGE = 9
LEGO_COLOR_AZURE = 10

# ── Firmware → App translation ──
_FIRMWARE_TO_APP = {
    _FW_NONE: LEGO_COLOR_NOCOLOR,
    _FW_BLACK: LEGO_COLOR_NOCOLOR,  # Black not in App → NoColor
    _FW_MAGENTA: LEGO_COLOR_MAGENTA,
    _FW_PURPLE: LEGO_COLOR_PURPLE,
    _FW_BLUE: LEGO_COLOR_BLUE,
    _FW_AZURE: LEGO_COLOR_AZURE,
    _FW_TURQUOISE: LEGO_COLOR_TEAL,
    _FW_GREEN: LEGO_COLOR_GREEN,
    _FW_YELLOW: LEGO_COLOR_YELLOW,
    _FW_ORANGE: LEGO_COLOR_ORANGE,
    _FW_RED: LEGO_COLOR_RED,
    _FW_WHITE: LEGO_COLOR_WHITE,
}

# ── App → Firmware translation ──
_APP_TO_FIRMWARE = {
    LEGO_COLOR_NOCOLOR: _FW_NONE,
    LEGO_COLOR_RED: _FW_RED,
    LEGO_COLOR_YELLOW: _FW_YELLOW,
    LEGO_COLOR_BLUE: _FW_BLUE,
    LEGO_COLOR_TEAL: _FW_TURQUOISE,
    LEGO_COLOR_GREEN: _FW_GREEN,
    LEGO_COLOR_PURPLE: _FW_PURPLE,
    LEGO_COLOR_WHITE: _FW_WHITE,
    LEGO_COLOR_MAGENTA: _FW_MAGENTA,
    LEGO_COLOR_ORANGE: _FW_ORANGE,
    LEGO_COLOR_AZURE: _FW_AZURE,
}

# ── Hex values for colors that have them ──
LEGO_COLOR_HEX_MAP = {
    LEGO_COLOR_RED: "#de1a21",
    LEGO_COLOR_YELLOW: "#ffd400",
    LEGO_COLOR_BLUE: "#006cb8",
    LEGO_COLOR_GREEN: "#61a836",
    LEGO_COLOR_PURPLE: "#4b2f91",
    LEGO_COLOR_MAGENTA: "#e4599e",
    LEGO_COLOR_ORANGE: "#f57d20",
    LEGO_COLOR_AZURE: "#78bfea",
}

# ── Validation sets ──
VALID_LEGO_COLOR = {
    LEGO_COLOR_NOCOLOR, LEGO_COLOR_RED, LEGO_COLOR_YELLOW, LEGO_COLOR_BLUE,
    LEGO_COLOR_TEAL, LEGO_COLOR_GREEN, LEGO_COLOR_PURPLE, LEGO_COLOR_WHITE,
    LEGO_COLOR_MAGENTA, LEGO_COLOR_ORANGE, LEGO_COLOR_AZURE,
}

VALID_LEGO_COLOR_STRING_FORMAT = {
    "LEGO_COLOR_NOCOLOR", "LEGO_COLOR_RED", "LEGO_COLOR_YELLOW", "LEGO_COLOR_BLUE",
    "LEGO_COLOR_TEAL", "LEGO_COLOR_GREEN", "LEGO_COLOR_PURPLE", "LEGO_COLOR_WHITE",
    "LEGO_COLOR_MAGENTA", "LEGO_COLOR_ORANGE", "LEGO_COLOR_AZURE",
}

# Colors the sensor can detect
SENSOR_DETECTABLE_COLORS = {
    LEGO_COLOR_RED, LEGO_COLOR_YELLOW, LEGO_COLOR_BLUE,
    LEGO_COLOR_TEAL, LEGO_COLOR_GREEN, LEGO_COLOR_PURPLE,
    LEGO_COLOR_WHITE,
}

# Colors available on connection cards
CARD_COLORS = {
    LEGO_COLOR_RED, LEGO_COLOR_YELLOW, LEGO_COLOR_BLUE,
    LEGO_COLOR_GREEN, LEGO_COLOR_PURPLE, LEGO_COLOR_MAGENTA,
    LEGO_COLOR_AZURE, LEGO_COLOR_ORANGE,
}

# ── Color name lookup (App-aligned values → constant names) ──
LEGO_COLOR_NAME_MAP = {
    LEGO_COLOR_NOCOLOR: "LEGO_COLOR_NOCOLOR",
    LEGO_COLOR_RED: "LEGO_COLOR_RED",
    LEGO_COLOR_YELLOW: "LEGO_COLOR_YELLOW",
    LEGO_COLOR_BLUE: "LEGO_COLOR_BLUE",
    LEGO_COLOR_TEAL: "LEGO_COLOR_TEAL",
    LEGO_COLOR_GREEN: "LEGO_COLOR_GREEN",
    LEGO_COLOR_PURPLE: "LEGO_COLOR_PURPLE",
    LEGO_COLOR_WHITE: "LEGO_COLOR_WHITE",
    LEGO_COLOR_MAGENTA: "LEGO_COLOR_MAGENTA",
    LEGO_COLOR_ORANGE: "LEGO_COLOR_ORANGE",
    LEGO_COLOR_AZURE: "LEGO_COLOR_AZURE",
}

# Public API — only constants and helper sets are user-visible.
# Translation functions are internal implementation details.
__all__ = [
    "LEGO_COLOR_NOCOLOR", "LEGO_COLOR_RED", "LEGO_COLOR_YELLOW",
    "LEGO_COLOR_BLUE", "LEGO_COLOR_TEAL", "LEGO_COLOR_GREEN",
    "LEGO_COLOR_PURPLE", "LEGO_COLOR_WHITE", "LEGO_COLOR_MAGENTA",
    "LEGO_COLOR_ORANGE", "LEGO_COLOR_AZURE",
    "LEGO_COLOR_NAME_MAP", "LEGO_COLOR_HEX_MAP",
    "VALID_LEGO_COLOR", "VALID_LEGO_COLOR_STRING_FORMAT",
    "SENSOR_DETECTABLE_COLORS", "CARD_COLORS",
]


def _firmware_to_app(firmware_color: int) -> int:
    """Translate a firmware color value to the App-aligned value (internal)."""
    return _FIRMWARE_TO_APP.get(firmware_color, LEGO_COLOR_NOCOLOR)


def _app_to_firmware(app_color: int) -> int:
    """Translate an App-aligned color value to the firmware value (internal)."""
    return _APP_TO_FIRMWARE.get(app_color, _FW_NONE)


def _translate_notification_colors(parsed_items):
    """Translate firmware color values to App-aligned values in-place (internal).

    Mutates 'ColorSensorNotification.color' and 'CardNotification.color'
    fields on any matching items in *parsed_items*. This is the single
    translation point used by both 'basic_device._handle_device_notification'
    and the public 'device_notification_parser' wrapper.
    """
    for item in parsed_items:
        if isinstance(item, _ColorSensorNotification):
            item.color = _firmware_to_app(item.color)
        if isinstance(item, _CardNotification):
            item.color = _firmware_to_app(item.color)
