"""Turn the steered wheel 60 degrees LEFT from where it is now, and stop."""
import legoeducation as le
from trike import Trike, STEER_GEAR_RATIO

t = Trike()
t.steer.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994)
t.steer.motor_set_speed(40)
print("connected -- treating the CURRENT wheel position as straight ahead")
t.center_steering()

t.set_steer(60, wait=True)
print(f"commanded +60 deg LEFT (motor moved {60 * STEER_GEAR_RATIO:.0f} motor-degrees)")
t.steer.disconnect()
