"""
Compatibility shim -- the macOS fixes now live in hand_control.py.

This file used to be a separate macOS variant, because hand_control.py opened
the camera with cv2.CAP_DSHOW (DirectShow, Windows-only) and assumed camera
index 0 was the built-in one. Both of those are now handled per-platform in
hand_control.py itself, so there is only one script to keep up to date.

Kept so that existing instructions and habits keep working:

    python hand_control_mac.py                # same as hand_control.py
    python hand_control_mac.py --no-robot
    python hand_control_mac.py --camera 1

Prefer hand_control.py in anything new.
"""

from hand_control import main

if __name__ == "__main__":
    main()
