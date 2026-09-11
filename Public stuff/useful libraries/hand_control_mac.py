"""
Drive the tricycle robot with your hand in front of the webcam -- macOS variant.

Same controls as hand_control.py; it differs only where that script assumes
Windows:

  * the capture backend is chosen per platform (CAP_DSHOW is Windows-only and
    never opens on macOS),
  * the built-in camera is picked by device type, since Continuity Camera adds
    a nearby iPhone as an extra device and shifts the index order,
  * a camera that fails to open disconnects the robot instead of leaving the
    motors live.

    python hand_control_mac.py                # drive the robot
    python hand_control_mac.py --no-robot     # camera only, for trying it out

Hold one hand up to the camera. Relative to the centre of the frame:

    up    -> forward          left  -> steer left
    down  -> backward         right -> steer right

Control is CONTINUOUS: the further from centre, the more speed or steering,
ramping smoothly from zero at the edge of the deadzone to full at the outer
ring. Both axes work at once, so up-and-left is a fast left turn.

    SPACE   arm / disarm (starts DISARMED -- the robot will not move until
            you arm it, so you can position your hand first)
    Q / Esc quit

Safety: losing the hand for a few frames stops the robot, as does disarming,
quitting, or any error.
"""

import argparse
import math
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions,
                                           RunningMode)

MODEL = "hand_landmarker.task"

# --- Control mapping ------------------------------------------------------
DEADZONE = 0.07     # fraction of the frame around centre where nothing happens
FULL_AT = 0.34      # distance from centre giving full output
SMOOTHING = 0.35    # EMA factor: lower = smoother but laggier
MAX_SPEED = 100.0
MAX_STEER = 45.0    # matches the UI's practical steering limit
LOST_HAND_FRAMES = 5
SEND_HZ = 25.0      # BLE drops the link much above ~30 Hz

# Landmarks averaged for a stable palm centre: wrist and middle-finger base.
WRIST, MIDDLE_MCP = 0, 9


def axis(value, dead=DEADZONE, full=FULL_AT):
    """Map a signed offset to -1..1 with a deadzone and a smooth ramp."""
    a = abs(value)
    if a <= dead:
        return 0.0
    return math.copysign(min(1.0, (a - dead) / (full - dead)), value)


def draw_hud(frame, cx, cy, hand_xy, speed, steer, armed, connected):
    h, w = frame.shape[:2]
    dead_px = int(DEADZONE * min(w, h) * 2)
    full_px = int(FULL_AT * min(w, h) * 2)

    cv2.circle(frame, (cx, cy), full_px // 2, (70, 70, 70), 1)
    cv2.circle(frame, (cx, cy), dead_px // 2, (0, 180, 255), 2)
    cv2.drawMarker(frame, (cx, cy), (120, 120, 120), cv2.MARKER_CROSS, 18, 1)

    if hand_xy is not None:
        hx, hy = hand_xy
        cv2.line(frame, (cx, cy), (hx, hy), (0, 220, 120), 2)
        cv2.circle(frame, (hx, hy), 11, (0, 220, 120), -1)
        cv2.circle(frame, (hx, hy), 13, (255, 255, 255), 2)

    # Bars: speed up the left edge, steering along the bottom.
    bar_x, bar_top, bar_h = 24, 90, 220
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 18, bar_top + bar_h), (60, 60, 60), 1)
    mid = bar_top + bar_h // 2
    cv2.line(frame, (bar_x, mid), (bar_x + 18, mid), (110, 110, 110), 1)
    fill = int((speed / MAX_SPEED) * (bar_h // 2))
    if fill:
        colour = (0, 220, 120) if fill > 0 else (0, 140, 255)
        cv2.rectangle(frame, (bar_x + 1, mid), (bar_x + 17, mid - fill), colour, -1)

    sb_y, sb_w = h - 40, 260
    sb_x = (w - sb_w) // 2
    cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_w, sb_y + 16), (60, 60, 60), 1)
    smid = sb_x + sb_w // 2
    cv2.line(frame, (smid, sb_y), (smid, sb_y + 16), (110, 110, 110), 1)
    sfill = int((steer / MAX_STEER) * (sb_w // 2))
    if sfill:
        cv2.rectangle(frame, (smid, sb_y + 1), (smid + sfill, sb_y + 15),
                      (255, 180, 0), -1)

    state = "ARMED" if armed else "DISARMED - press SPACE"
    colour = (0, 220, 120) if armed else (0, 165, 255)
    cv2.putText(frame, state, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    link = "robot: connected" if connected else "robot: offline"
    cv2.putText(frame, link, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    cv2.putText(frame, "speed {:+6.1f}   steer {:+6.1f}".format(speed, steer),
                (24, h - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)


def _builtin_camera_index():
    """OpenCV index of the Mac's built-in camera, or None if it can't be found.

    Continuity Camera makes a nearby iPhone appear as an extra capture device,
    and the index order is not guaranteed, so pick the built-in camera by device
    type instead of trusting index 0. Returns None off macOS or when the
    optional pyobjc AVFoundation bridge is absent -- callers fall back to 0.
    """
    if sys.platform != "darwin":
        return None
    try:
        import AVFoundation as AV
    except ImportError:
        return None
    names = ("AVCaptureDeviceTypeBuiltInWideAngleCamera",
             "AVCaptureDeviceTypeExternal",
             "AVCaptureDeviceTypeContinuityCamera",
             "AVCaptureDeviceTypeDeskViewCamera")
    types = [t for t in (getattr(AV, n, None) for n in names) if t]
    try:
        session = (AV.AVCaptureDeviceDiscoverySession
                   .discoverySessionWithDeviceTypes_mediaType_position_(
                       types, AV.AVMediaTypeVideo,
                       AV.AVCaptureDevicePositionUnspecified))
        for i, dev in enumerate(session.devices()):
            if dev.deviceType() == AV.AVCaptureDeviceTypeBuiltInWideAngleCamera:
                return i
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-robot", action="store_true", help="camera only")
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index (default: the built-in camera)")
    args = ap.parse_args()

    bot = None
    if not args.no_robot:
        import trike
        # flush=True: stdout is buffered when redirected to a file, so without
        # it these stage messages never appear in a log.
        say = lambda m: print(m, flush=True)
        say("connecting to robot ...")
        try:
            bot = trike.Trike().connect(center=False, progress=say)
        except Exception as exc:
            say("ROBOT NOT CONNECTED: {}".format(exc))
            say("continuing with the camera only -- the HUD will say 'offline'.")
            bot = None

    # CAP_DSHOW is DirectShow -- Windows only. On macOS/Linux it never opens, so
    # use OpenCV's default backend there (AVFoundation / V4L2).
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    index = args.camera
    if index is None:
        index = _builtin_camera_index()
        if index is None:
            index = 0
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        # The robot is already connected by this point; without this the motors
        # are left live and the dropped link surfaces as a stray
        # "Device disconnected unexpectedly" future.
        if bot is not None:
            try:
                bot.stop()
                bot.disconnect()
            except Exception:
                pass
        raise SystemExit("cannot open camera {}".format(index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5))

    armed = False
    speed = steer = 0.0
    missing = 0
    last_send = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)          # mirror: move right, dot right
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int((time.perf_counter() - t0) * 1000)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            hand_xy = None
            if res.hand_landmarks:
                lm = res.hand_landmarks[0]
                nx = (lm[WRIST].x + lm[MIDDLE_MCP].x) / 2.0
                ny = (lm[WRIST].y + lm[MIDDLE_MCP].y) / 2.0
                hand_xy = (int(nx * w), int(ny * h))
                missing = 0

                # Up is forward, so invert y. Steering is +left in trike's
                # convention, and the mirrored image already matches the
                # user's own left, so a dot LEFT of centre gives +angle.
                tgt_speed = axis(0.5 - ny) * MAX_SPEED
                tgt_steer = axis(0.5 - nx) * MAX_STEER
            else:
                missing += 1
                tgt_speed = tgt_steer = 0.0
                if missing >= LOST_HAND_FRAMES:
                    speed = steer = 0.0

            speed += SMOOTHING * (tgt_speed - speed)
            steer += SMOOTHING * (tgt_steer - steer)
            if abs(speed) < 0.5:
                speed = 0.0
            if abs(steer) < 0.5:
                steer = 0.0

            now = time.perf_counter()
            if bot is not None and now - last_send >= 1.0 / SEND_HZ:
                last_send = now
                try:
                    bot.drive_at(speed if armed else 0.0,
                                 steer if armed else 0.0, settle=False)
                except Exception as exc:
                    print("lost link: {}".format(exc))
                    bot = None

            draw_hud(frame, cx, cy, hand_xy, speed, steer, armed, bot is not None)
            cv2.imshow("Hand Control  [SPACE arm/disarm]  [Q quit]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                armed = not armed
                if not armed and bot is not None:
                    bot.drive.movement_move_tank(0, 0, blocking=False)
    finally:
        if bot is not None:
            try:
                bot.stop()
                bot.disconnect()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("stopped")


if __name__ == "__main__":
    main()
