"""Tkinter MQTT controller for BigLegoRobot. Requires paho-mqtt >= 2."""
import json
import queue
import time
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import paho.mqtt.client as mqtt
except ImportError:
    raise SystemExit('Install dependencies first: python -m pip install -r requirements.txt')


class RobotControl:
    def __init__(self, root):
        self.root = root
        root.title('🛰️ MQTT Robot Control')
        root.geometry('760x720')
        self.client = None
        self.connected = False
        self.events = queue.Queue()
        self.active = None
        self.deadline = None
        self.next_send = 0
        self.host = tk.StringVar(value='broker.hivemq.com')
        self.port = tk.StringVar(value='1883')
        self.topic = tk.StringVar(value='BigLegoRobot')
        self.speed = tk.DoubleVar(value=0.5)
        self.left = tk.StringVar(value='0.3')
        self.right = tk.StringVar(value='-0.3')
        self.duration = tk.StringVar(value='1.0')
        self.state = tk.StringVar(value='Disconnected')
        self.left_invert = tk.BooleanVar(value=False)
        self.right_invert = tk.BooleanVar(value=True)
        self.raw = tk.StringVar(value='{"cmd":"config"}')
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='MQTT Robot Control', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='Public broker • Commands are visible to others • Confirm robot is clear before driving.').pack(anchor='w', pady=(4, 12))
        network = ttk.Frame(frame)
        network.pack(fill='x')
        ttk.Combobox(network, textvariable=self.host, values=['broker.hivemq.com', 'broker.emqx.io', 'test.mosquitto.org'], width=27).pack(side='left')
        ttk.Entry(network, textvariable=self.port, width=7).pack(side='left', padx=5)
        ttk.Entry(network, textvariable=self.topic, width=21).pack(side='left')
        ttk.Button(network, text='Connect', command=self.connect).pack(side='left', padx=5)
        ttk.Button(network, text='Disconnect', command=self.disconnect).pack(side='left')
        ttk.Label(frame, textvariable=self.state).pack(anchor='w', pady=8)
        ttk.Label(frame, text='Speed fraction (0–1)').pack(anchor='w')
        ttk.Scale(frame, from_=0, to=1, variable=self.speed).pack(fill='x')
        drive = ttk.LabelFrame(frame, text='Hold a button to drive; release to stop', padding=8)
        drive.pack(fill='x', pady=10)
        for label, command, row, col in [('Forward', 'forward', 0, 1), ('Left', 'left', 1, 0), ('Right', 'right', 1, 2), ('Backward', 'backward', 2, 1)]:
            button = ttk.Button(drive, text=label)
            button.grid(row=row, column=col, padx=5, pady=3, sticky='ew')
            button.bind('<ButtonPress-1>', lambda e, c=command: self.direction(c))
            button.bind('<ButtonRelease-1>', lambda e: self.stop())
        ttk.Button(drive, text='STOP (Space)', command=self.stop).grid(row=1, column=1, padx=5)
        for i in range(3):
            drive.columnconfigure(i, weight=1)
        manual = ttk.LabelFrame(frame, text='Independent motors: fractions −1…1 or PWM −255…255', padding=8)
        manual.pack(fill='x')
        for label, variable in [('Left', self.left), ('Right', self.right), ('Seconds', self.duration)]:
            ttk.Label(manual, text=label).pack(side='left', padx=4)
            ttk.Entry(manual, textvariable=variable, width=9).pack(side='left')
        ttk.Button(manual, text='Run timed', command=self.timed).pack(side='left', padx=8)
        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=12)
        ttk.Button(controls, text='Arm', command=lambda: self.control('arm')).pack(side='left')
        ttk.Button(controls, text='Disarm', command=lambda: self.control('disarm')).pack(side='left', padx=5)
        ttk.Checkbutton(controls, text='Invert left', variable=self.left_invert).pack(side='left')
        ttk.Checkbutton(controls, text='Invert right', variable=self.right_invert).pack(side='left')
        ttk.Button(controls, text='Apply inversion', command=self.invert).pack(side='left', padx=5)
        ttk.Label(frame, text='Raw command (JSON or plain text): config, min_pwm, throttle/steer, etc.').pack(anchor='w')
        raw_row = ttk.Frame(frame)
        raw_row.pack(fill='x', pady=5)
        ttk.Entry(raw_row, textvariable=self.raw).pack(side='left', fill='x', expand=True)
        ttk.Button(raw_row, text='Send once', command=self.send_raw).pack(side='left', padx=5)
        ttk.Label(frame, text='Broker / robot status').pack(anchor='w', pady=(8, 3))
        self.log = tk.Text(frame, height=10, state='disabled', wrap='word')
        self.log.pack(fill='both', expand=True)
        root.bind('<space>', lambda e: self.stop())
        root.bind('<Escape>', lambda e: self.stop())
        root.bind('<FocusOut>', self.focus_lost)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(50, self.tick)

    def record(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', time.strftime('%H:%M:%S ') + text + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def connect(self):
        try:
            port = int(self.port.get())
            if not 1 <= port <= 65535 or not self.host.get().strip() or not self.topic.get().strip():
                raise ValueError('Enter a host, a valid port, and a topic.')
            if '+' in self.topic.get() or '#' in self.topic.get():
                raise ValueError('Command topic cannot contain MQTT wildcards.')
        except ValueError as exc:
            messagebox.showerror('Connection settings', str(exc))
            return
        self.disconnect()
        self.command_topic = self.topic.get().strip()
        self.endpoint = f'{self.host.get().strip()}:{port}'
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client = client
        client.on_connect = lambda c, u, f, reason, p: self.events.put(('connect', c, reason))
        client.on_disconnect = lambda c, u, f, reason, p: self.events.put(('disconnect', c, reason))
        client.on_message = lambda c, u, m: self.events.put(('message', c, m.payload.decode('utf-8', errors='replace')))
        client.on_connect_fail = lambda c, u: self.events.put(('failure', c, 'Connection failed; retrying. Choose another broker if needed.'))
        self.state.set(f'Connecting to {self.endpoint}…')
        try:
            client.connect_async(self.host.get().strip(), port, keepalive=10)
            client.loop_start()
        except Exception as exc:
            self.record(str(exc))
            self.disconnect()

    def publish(self, payload):
        if not self.connected or not self.client:
            return False
        text = payload if isinstance(payload, str) else json.dumps(payload)
        result = self.client.publish(self.command_topic, text, qos=0, retain=False)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            self.active = None
            self.record('Publish failed; motion cleared.')
            return False
        return True

    def start(self, payload, duration=None):
        if not self.connected:
            messagebox.showinfo('Disconnected', 'Connect to the same broker as the robot first.')
            return
        if self.publish(payload):
            self.active = payload
            self.deadline = time.monotonic() + duration if duration is not None else None
            self.next_send = time.monotonic() + 0.25
            self.record('Drive: ' + json.dumps(payload))

    def direction(self, direction):
        speed = round(self.speed.get(), 3)
        left, right = {'forward': (speed, speed), 'backward': (-speed, -speed), 'left': (-speed, speed), 'right': (speed, -speed)}[direction]
        self.start({'left': left, 'right': right})

    def timed(self):
        try:
            left, right, seconds = float(self.left.get()), float(self.right.get()), float(self.duration.get())
            if not (-255 <= left <= 255 and -255 <= right <= 255 and 0 < seconds <= 60):
                raise ValueError('Motor values must be −255…255; duration must be greater than 0 and at most 60 seconds.')
            self.start({'left': left, 'right': right, 'duration': seconds}, seconds)
        except ValueError as exc:
            messagebox.showerror('Invalid drive values', str(exc))

    def stop(self):
        self.active = None
        self.deadline = None
        self.publish('stop')

    def control(self, command):
        self.stop()
        if self.publish({'cmd': command}):
            self.record('Control: ' + command)

    def invert(self):
        self.stop()
        self.publish({'cmd': 'invert', 'left': self.left_invert.get(), 'right': self.right_invert.get()})

    def send_raw(self):
        self.stop()
        value = self.raw.get().strip()
        if value.startswith('{'):
            try:
                json.loads(value)
            except ValueError as exc:
                messagebox.showerror('Invalid JSON', str(exc))
                return
        if value and self.publish(value):
            self.record('Sent once: ' + value)

    def focus_lost(self, event):
        self.root.after(20, lambda: self.stop() if self.root.focus_displayof() is None else None)

    def disconnect(self):
        self.stop()
        self.connected = False
        if self.client:
            self.client.disconnect()
            self.client.loop_stop()
            self.client = None
        self.state.set('Disconnected')

    def tick(self):
        while not self.events.empty():
            kind, client, value = self.events.get_nowait()
            if client is not self.client:
                continue
            if kind == 'connect':
                self.connected = not value.is_failure
                self.active = None
                if self.connected:
                    client.subscribe(self.command_topic + '/status')
                    self.publish('stop')
                    self.state.set(f'Connected: {self.endpoint} • {self.command_topic}')
                else:
                    self.state.set(f'Connection rejected: {value}')
                self.record(self.state.get())
            elif kind == 'disconnect':
                self.connected = False
                self.active = None
                self.state.set('Connection lost; motion cleared. Reconnecting…')
                self.record(self.state.get())
            else:
                self.record(str(value))
        now = time.monotonic()
        if self.active is not None:
            if self.deadline is not None and now >= self.deadline:
                self.stop()
            elif now >= self.next_send:
                payload = dict(self.active)
                if self.deadline is not None:
                    payload['duration'] = max(0.01, round(self.deadline - now, 3))
                self.publish(payload)
                self.next_send = now + 0.25
        self.root.after(50, self.tick)

    def close(self):
        self.disconnect()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    RobotControl(root)
    root.mainloop()
