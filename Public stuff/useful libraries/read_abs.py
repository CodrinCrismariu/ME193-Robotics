"""Read the steering motor's absolute encoder position (wheel is straight now)."""
import time
import legoeducation as le

m = le.SingleMotor()
m.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994)
time.sleep(1.0)
for _ in range(5):
    n = m.motor                      # re-read: notifications rebind this
    print(f"absolutePosition={n.absolutePosition}  position={n.position}  speed={n.speed}")
    time.sleep(0.3)
m.disconnect()
