"""Pose Race: steer a LEGO Education robot with your right wrist.

Drivebase: Double Motor block (driving wheels).
Steering: Single Motor block (steering wheel), driven to an absolute angle.

Control scheme, measured from the nose to the right wrist in the camera image:
  - Within STOP_RADIUS of the nose: stop driving and center the steering.
  - Wrist above the nose: drive forward, power scales with vertical distance.
  - Wrist below the nose: drive backward, power scales with vertical distance.
  - Wrist right of the nose: steer right, angle scales with horizontal distance.
  - Wrist left of the nose: steer left, angle scales with horizontal distance.
"""

import argparse
import math
import os
import time
import urllib.request

import cv2
import numpy as np
import legoeducation as le
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Dead zone: wrist within this many pixels of the nose stops the robot.
STOP_RADIUS = 40

# Pixel distance (beyond the dead zone) that maps to full speed / full steering angle.
SPEED_MAX_DISTANCE = 200
STEER_MAX_DISTANCE = 200

MAX_SPEED = 100
MAX_STEER_ANGLE = 90
STEERING_MOTOR_SPEED = 100

COMMAND_INTERVAL = 0.1

NOSE = vision.PoseLandmark.NOSE
RIGHT_WRIST = vision.PoseLandmark.RIGHT_WRIST


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def scale_to_range(distance, dead_zone, max_distance, max_output):
    """Map a distance beyond dead_zone linearly to [0, max_output]."""
    span = max(1e-6, max_distance - dead_zone)
    return clamp((distance - dead_zone) / span, 0.0, 1.0) * max_output


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def connect_drivebase():
    drivebase = le.DoubleMotor()
    print("Searching for a LEGO Education Double Motor drivebase over Bluetooth...")
    drivebase.connect()
    print("Connected to drivebase.")
    return drivebase


def connect_steering():
    steering = le.SingleMotor()
    print("Searching for a LEGO Education Single Motor steering block over Bluetooth...")
    steering.connect()
    steering.motor_reset_relative_position()
    print("Connected to steering.")
    return steering


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        cv2.line(
            frame,
            (int(start.x * w), int(start.y * h)),
            (int(end.x * w), int(end.y * h)),
            (0, 255, 0),
            2,
        )
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)


DASHBOARD_SIZE = (320, 420)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_dashboard(speed, direction, angle, connected):
    w, h = DASHBOARD_SIZE
    dash = np.full((h, w, 3), (32, 32, 32), dtype=np.uint8)

    cv2.putText(dash, "Pose Race Debug", (12, 30), FONT, 0.7, (255, 255, 255), 2)
    conn_text, conn_color = ("HARDWARE", (0, 200, 0)) if connected else ("SIM (no hardware)", (0, 165, 255))
    cv2.putText(dash, conn_text, (12, 55), FONT, 0.5, conn_color, 1)

    # --- Steering dial ---
    cx, cy, r = w // 2, 165, 90
    cv2.circle(dash, (cx, cy), r, (90, 90, 90), 2)
    for mark_angle in range(-MAX_STEER_ANGLE, MAX_STEER_ANGLE + 1, 30):
        rad = math.radians(mark_angle - 90)
        x1 = int(cx + (r - 10) * math.cos(rad))
        y1 = int(cy + (r - 10) * math.sin(rad))
        x2 = int(cx + r * math.cos(rad))
        y2 = int(cy + r * math.sin(rad))
        cv2.line(dash, (x1, y1), (x2, y2), (110, 110, 110), 1)

    needle_rad = math.radians(clamp(angle, -MAX_STEER_ANGLE, MAX_STEER_ANGLE) - 90)
    nx = int(cx + (r - 12) * math.cos(needle_rad))
    ny = int(cy + (r - 12) * math.sin(needle_rad))
    cv2.line(dash, (cx, cy), (nx, ny), (0, 255, 255), 3)
    cv2.circle(dash, (cx, cy), 6, (0, 255, 255), -1)
    cv2.putText(dash, f"Steer {angle:+d} deg", (cx - 75, cy + r + 30), FONT, 0.6, (255, 255, 255), 1)

    # --- Speed bar (signed, forward = right/green, backward = left/blue) ---
    bar_x, bar_y, bar_w, bar_h = 30, cy + r + 55, w - 60, 36
    mid_x = bar_x + bar_w // 2
    cv2.rectangle(dash, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 1)
    cv2.line(dash, (mid_x, bar_y), (mid_x, bar_y + bar_h), (90, 90, 90), 1)

    signed_speed = 0
    if direction == le.MOVEMENT_DIRECTION_FORWARD:
        signed_speed = speed
    elif direction == le.MOVEMENT_DIRECTION_BACKWARD:
        signed_speed = -speed

    fill_w = int((bar_w // 2) * (abs(signed_speed) / MAX_SPEED))
    color = (0, 200, 0) if signed_speed >= 0 else (255, 120, 0)
    if signed_speed >= 0:
        cv2.rectangle(dash, (mid_x, bar_y), (mid_x + fill_w, bar_y + bar_h), color, -1)
    else:
        cv2.rectangle(dash, (mid_x - fill_w, bar_y), (mid_x, bar_y + bar_h), color, -1)

    label = "STOP" if direction is None else f"{'FWD' if direction == le.MOVEMENT_DIRECTION_FORWARD else 'BACK'} {speed}%"
    cv2.putText(dash, label, (bar_x, bar_y + bar_h + 28), FONT, 0.6, (255, 255, 255), 1)
    cv2.putText(dash, "-100", (bar_x - 4, bar_y - 8), FONT, 0.4, (150, 150, 150), 1)
    cv2.putText(dash, "+100", (bar_x + bar_w - 34, bar_y - 8), FONT, 0.4, (150, 150, 150), 1)

    return dash


def main():
    parser = argparse.ArgumentParser(description="Drive a LEGO robot by pointing with your right wrist.")
    parser.add_argument("--sim", action="store_true", help="Run the vision pipeline without connecting to a robot.")
    args = parser.parse_args()

    ensure_model()

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    drivebase = None
    steering = None
    if not args.sim:
        drivebase = connect_drivebase()
        steering = connect_steering()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_command_time = 0.0
    last_speed = None
    last_direction = None
    last_angle = None
    start_time = time.time()

    print("Move your right wrist above/below/left/right of your nose to drive. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((now - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            speed = 0
            direction = None
            angle = 0
            radius = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_pose(frame, landmarks)

                nose = landmarks[NOSE]
                wrist = landmarks[RIGHT_WRIST]

                dx = (wrist.x - nose.x) * w
                dy = (wrist.y - nose.y) * h
                radius = (dx ** 2 + dy ** 2) ** 0.5

                if radius >= STOP_RADIUS:
                    if dy < 0:
                        direction = le.MOVEMENT_DIRECTION_FORWARD
                        speed = int(scale_to_range(abs(dy), 0, SPEED_MAX_DISTANCE, MAX_SPEED))
                    elif dy > 0:
                        direction = le.MOVEMENT_DIRECTION_BACKWARD
                        speed = int(scale_to_range(abs(dy), 0, SPEED_MAX_DISTANCE, MAX_SPEED))

                    steer_magnitude = scale_to_range(abs(dx), 0, STEER_MAX_DISTANCE, MAX_STEER_ANGLE)
                    angle = int(steer_magnitude if dx >= 0 else -steer_magnitude)

                nose_px = (int(nose.x * w), int(nose.y * h))
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                cv2.circle(frame, nose_px, STOP_RADIUS, (255, 255, 0), 1)
                cv2.line(frame, nose_px, wrist_px, (255, 0, 255), 2)

            if drivebase is not None and steering is not None and now - last_command_time >= COMMAND_INTERVAL:
                if direction is None:
                    if last_speed != 0:
                        drivebase.movement_stop(blocking=False)
                else:
                    if speed != last_speed or direction != last_direction:
                        drivebase.movement_move(direction=direction, speed=speed, blocking=False)

                if angle != last_angle:
                    steering.motor_run_to_absolute_position(angle, speed=STEERING_MOTOR_SPEED, blocking=False)

                last_speed = speed if direction is not None else 0
                last_direction = direction
                last_angle = angle
                last_command_time = now

            status = "STOP" if direction is None else f"{'FWD' if direction == le.MOVEMENT_DIRECTION_FORWARD else 'BACK'} {speed}%"
            cv2.putText(
                frame,
                f"{status}  steer:{angle}deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Pose Race", frame)
            cv2.imshow("Dashboard", draw_dashboard(speed, direction, angle, connected=drivebase is not None))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if drivebase is not None:
            drivebase.movement_stop()
            drivebase.disconnect()
        if steering is not None:
            steering.motor_run_to_absolute_position(0, speed=STEERING_MOTOR_SPEED)
            steering.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
