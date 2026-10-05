# MQTT Robot Control

Double-click **Start Robot Control.bat**. First launch creates a local Python environment and installs paho-mqtt. Requires Python 3 with Tkinter and internet access.

1. Confirm which broker the robot reports. Default: broker.hivemq.com:1883. Both ends must use the same broker. Select broker.emqx.io or test.mosquitto.org if appropriate; the controller does not automatically switch brokers because this could leave it talking to a different server than the robot.
2. Topic defaults to BigLegoRobot; status subscription is BigLegoRobot/status.
3. Connect, arm if required by the firmware, then hold a direction button. Release, Space, Escape, leaving the app, disconnecting, or closing sends stop. Direction controls use explicit left/right values.
4. Independent motor fields accept fractions −1…1 or raw PWM −255…255 as interpreted by the robot. Run timed sends immediately and repeats every 250 ms, with remaining duration, then stops locally.
5. Arm/disarm and inversion are available directly. Raw commands send once and are not repeated. Use raw JSON for config, min_pwm, throttle/steer, or other firmware commands. The supplied instructions do not define the min_pwm/config argument schema; consult the robot firmware before sending settings.

Active drive commands repeat every 250 ms. Reconnection clears motion and sends stop; driving never resumes automatically. The robot's firmware must enforce its stated 800 ms command timeout. A GUI stop is best effort; physically accessible power cutoff remains useful. Public brokers provide no exclusive control: anyone using this topic can send commands. This connection uses unauthenticated, unencrypted MQTT on port 1883.

Right inversion defaults to checked, matching the reported firmware default; nothing is changed until Apply inversion is pressed. If forward spins in place, try both inversion boxes unchecked and apply while stopped.

This controller was validated locally; physical motor direction and the robot connection must be verified with the actual board. The previous end-to-end results supplied in the request are not a test of this new controller.
