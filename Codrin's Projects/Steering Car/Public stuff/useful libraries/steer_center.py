"""Return the steered wheel to the zero set at the start of steer_left60.py."""
import legoeducation as le
from trike import Trike

t = Trike()
t.steer.connect(card_color=le.LEGO_COLOR_GREEN, card_serial=994)
t.steer.motor_set_speed(40)
t.steer.motor_run_to_relative_position(0, blocking=True)
print("steering returned to zero")
t.steer.disconnect()
