"""List every LEGO Education device in Bluetooth range with its connection-card color/serial."""
import asyncio
import sys

# LEGO advertises names containing coloured-square emoji, which the default
# Windows console codepage (cp1252) cannot encode -- printing one would abort
# the whole scan.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from bleak import BleakScanner
from legoeducation.color_map import _firmware_to_app

SERVICE_UUID = "0000fd02-0000-1000-8000-00805f9b34fb"
LEGO_COMPANY_ID = 0x0397
COLORS = {1: "RED", 2: "YELLOW", 3: "BLUE", 4: "GREEN", 5: "GREEN?",
          6: "PURPLE", 7: "MAGENTA", 8: "AZURE", 9: "ORANGE"}

seen = {}

def on_scan(device, adv):
    uuids = [u.lower() for u in (adv.service_uuids or [])]
    if SERVICE_UUID not in uuids:
        return
    data = (adv.manufacturer_data or {}).get(LEGO_COMPANY_ID)
    if not data or len(data) < 5:
        return
    product_id = (data[0] << 8) | data[1]
    color = _firmware_to_app(data[2])
    serial = data[3] | (data[4] << 8)
    seen[device.address] = (device.name, product_id, color, serial, adv.rssi)

async def main():
    scanner = BleakScanner(detection_callback=on_scan, service_uuids=[SERVICE_UUID])
    await scanner.start()
    await asyncio.sleep(10)
    await scanner.stop()
    print(f"\n{len(seen)} LEGO device(s) in range:")
    for addr, (name, pid, color, serial, rssi) in sorted(seen.items(), key=lambda kv: -kv[1][4]):
        print(f"  {name!r:28} product_id=0x{pid:04X}  card_color={color} ({COLORS.get(color,'?')})  "
              f"card_serial={serial:04d}  rssi={rssi}")

try:
    asyncio.run(main())
except Exception as exc:
    # A switched-off radio otherwise dumps a bleak traceback that buries the
    # one line that matters.
    from trike import bluetooth_error
    problem = bluetooth_error()
    raise SystemExit(problem if problem else "scan failed: {}".format(exc))
