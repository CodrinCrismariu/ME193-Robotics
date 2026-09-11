"""
Small pop-up control panel for the tricycle robot.

    python robot_ui.py

Keys (hold them -- releasing stops, like a dead-man switch):
    W / Up      drive forward          A / Left    steer left
    S / Down    drive backward         D / Right   steer right
    Space       stop now               Esc         quit

The slider sets top speed. Steering sweeps toward the limit while held and
returns to centre when released, like a car's wheel.

Why it is built this way
------------------------
* Everything talks to the robot from the Tk main loop on a fixed ~12 Hz tick.
  Writing faster than ~30 Hz floods the BLE link and the hub drops the
  connection, so the tick rate is the throttle.
* Steering is commanded non-blocking, and each tick recomputes the wheel
  speeds from the angle the steered wheel has ACTUALLY reached. So while the
  wheel is still sweeping, the speeds track it instead of assuming it has
  arrived -- which is when a tricycle scrubs.
* Connecting happens on a background thread so the window appears instantly
  instead of freezing on a BLE scan.
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk

import trike

TICK_MS = 40              # 25 Hz -- about as fast as BLE takes before the hub
                          # starts dropping the link. Do not go much below this.
STEER_RATE = 400.0        # degrees of steering per second while a key is held
STEER_RETURN = 500.0      # degrees per second returning to centre when released
PRACTICAL_MAX_STEER = 45.0


class RobotUI:
    def __init__(self, root):
        self.root = root
        root.title("Robot Control")
        root.resizable(False, False)

        self.bot = None
        self.status_q = queue.Queue()
        self.pressed = set()
        self.steer = 0.0
        self.last_sent = None

        pad = dict(padx=10, pady=4)

        self.status = tk.StringVar(value="connecting ...")
        ttk.Label(root, textvariable=self.status, font=("Segoe UI", 10, "bold")
                  ).grid(row=0, column=0, columnspan=3, **pad)

        ttk.Label(root, text="Top speed").grid(row=1, column=0, sticky="w", **pad)
        self.speed_var = tk.IntVar(value=100)
        ttk.Scale(root, from_=10, to=100, variable=self.speed_var,
                  orient="horizontal", length=180).grid(row=1, column=1, **pad)
        self.speed_lbl = ttk.Label(root, text="100")
        self.speed_lbl.grid(row=1, column=2, **pad)

        # On-screen buttons, for driving with the mouse.
        grid = ttk.Frame(root)
        grid.grid(row=2, column=0, columnspan=3, pady=(6, 2))
        self._mk(grid, "▲", 0, 1, "fwd")
        self._mk(grid, "◀", 1, 0, "left")
        self._mk(grid, "STOP", 1, 1, None)
        self._mk(grid, "▶", 1, 2, "right")
        self._mk(grid, "▼", 2, 1, "back")

        self.readout = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.readout, font=("Consolas", 9),
                  foreground="#555").grid(row=3, column=0, columnspan=3, **pad)

        ttk.Label(root, text="hold W A S D or the arrow keys · Space = stop",
                  foreground="#777").grid(row=4, column=0, columnspan=3, pady=(0, 8))

        for seq, key in (("<KeyPress>", True), ("<KeyRelease>", False)):
            root.bind(seq, lambda e, d=key: self._key(e, d))
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.focus_force()

        threading.Thread(target=self._connect, daemon=True).start()
        root.after(TICK_MS, self.tick)

    def _mk(self, parent, text, r, c, action):
        b = ttk.Button(parent, text=text, width=6)
        b.grid(row=r, column=c, padx=3, pady=3)
        if action is None:
            b.configure(command=self.stop_now)
        else:
            b.bind("<ButtonPress-1>", lambda e: self.pressed.add(action))
            b.bind("<ButtonRelease-1>", lambda e: self.pressed.discard(action))
        return b

    # --- connection ------------------------------------------------------
    def _connect(self):
        try:
            # center=False: centring uses BLOCKING motor moves, which can hang
            # this thread forever if the steered wheel cannot reach its target.
            bot = trike.Trike().connect(
                center=False, progress=lambda m: self.status_q.put(("ok", m)))
            self.bot = bot
            self.status_q.put(("ok", "connected"))
        except Exception as exc:
            self.status_q.put(("err", f"connect failed: {exc}"))

    # --- input -----------------------------------------------------------
    KEYMAP = {"w": "fwd", "up": "fwd", "s": "back", "down": "back",
              "a": "left", "left": "left", "d": "right", "right": "right"}

    def _key(self, event, down):
        k = event.keysym.lower()
        if k == "escape" and down:
            return self.quit()
        if k == "space" and down:
            return self.stop_now()
        action = self.KEYMAP.get(k)
        if action:
            self.pressed.add(action) if down else self.pressed.discard(action)

    def stop_now(self):
        self.pressed.clear()
        self.steer = 0.0
        if self.bot and not getattr(self.bot.drive, "connected", True):
            self.status.set("link lost -- robot powered off?")
            self.bot = None

        if self.bot:
            try:
                self.bot.drive.movement_move_tank(0, 0, blocking=False)
                self.bot.set_steer(0.0, wait=False)
            except Exception:
                pass
        self.last_sent = None

    # --- main loop -------------------------------------------------------
    def tick(self):
        while not self.status_q.empty():
            kind, msg = self.status_q.get()
            self.status.set(msg)
        self.speed_lbl.configure(text=str(self.speed_var.get()))

        dt = TICK_MS / 1000.0
        top = float(self.speed_var.get())

        speed = 0.0
        if "fwd" in self.pressed:
            speed += top
        if "back" in self.pressed:
            speed -= top

        # Steering sweeps while held, springs back to centre when released.
        want_left = "left" in self.pressed
        want_right = "right" in self.pressed
        if want_left and not want_right:
            self.steer = min(PRACTICAL_MAX_STEER, self.steer + STEER_RATE * dt)
        elif want_right and not want_left:
            self.steer = max(-PRACTICAL_MAX_STEER, self.steer - STEER_RATE * dt)
        else:
            step = STEER_RETURN * dt
            self.steer = 0.0 if abs(self.steer) < step else self.steer - step * (
                1 if self.steer > 0 else -1)

        if self.bot:
            try:
                # settle=False: never block the UI waiting for the steering
                # motor. drive_at reads the real angle and matches speeds to it.
                vl, vr = self.bot.drive_at(speed, self.steer, settle=False)
                actual = self.bot.steer_position()
                self.readout.set(f"steer {actual:+6.1f}°   L {vl:+6.1f}   R {vr:+6.1f}")
            except Exception as exc:
                self.status.set(f"lost link: {exc}")
                self.bot = None

        self.root.after(TICK_MS, self.tick)

    def quit(self):
        if self.bot:
            try:
                self.bot.stop()
                self.bot.disconnect()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    RobotUI(root)
    root.mainloop()
