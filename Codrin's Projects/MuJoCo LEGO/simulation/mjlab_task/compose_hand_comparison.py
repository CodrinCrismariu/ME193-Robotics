"""Make a 20-second, real-time comparison from a real clip and a simulated replay.

Usage: python compose_hand_comparison.py real.MOV simulation.mp4 hand_schedule.json output_dir
Requires imageio, imageio-ffmpeg and Pillow; no GPU needed.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def font(size):
  return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("real")
  parser.add_argument("simulation")
  parser.add_argument("schedule")
  parser.add_argument("output_dir", type=Path)
  args = parser.parse_args()
  schedule = json.loads(Path(args.schedule).read_text())
  args.output_dir.mkdir(parents=True, exist_ok=True)
  fps, duration = 30, schedule["duration_s"]
  real = imageio.get_reader(args.real, format="ffmpeg", size=(404, 720), input_params=["-threads", "2"])
  sim = imageio.get_reader(args.simulation, format="ffmpeg", size=(700, 700), input_params=["-threads", "2"])
  real_fps, sim_fps = real.get_meta_data()["fps"], sim.get_meta_data()["fps"]
  title_font, text_font, small_font = font(26), font(20), font(16)
  writer_args = dict(fps=fps, codec="libx264", quality=None, macro_block_size=16, ffmpeg_log_level="error",
                     output_params=["-preset", "veryfast", "-crf", "20", "-threads", "2"])
  with imageio.get_writer(args.output_dir / "real-vs-simulation.mp4", **writer_args) as comparison, \
       imageio.get_writer(args.output_dir / "simulation-timed.mp4", **writer_args) as simulation:
    import numpy as np
    for frame in range(round(duration * fps)):
      t = frame / fps
      real_im = Image.fromarray(real.get_data(round(t * real_fps)))
      sim_im = Image.fromarray(sim.get_data(round(t * sim_fps)))
      active = [c["id"] for c in schedule["contacts"] if c["start_s"] <= t < c["end_s"]]
      status = " / ".join(active) if active else ("Waiting for policy start" if t < schedule["policy_start_s"] else "Policy running")
      color = "#ff9a77" if active else "#a3c6d4"
      canvas = Image.new("RGB", (1280, 864), "#101923")
      draw = ImageDraw.Draw(canvas)
      draw.text((36, 14), "REAL ROBOT", font=title_font, fill="white")
      draw.text((500, 14), "SIMULATION", font=title_font, fill="white")
      draw.text((1055, 16), f"{t:05.2f} / 20 s", font=text_font, fill="white")
      draw.text((36, 49), "Original clip, first 20 seconds", font=small_font, fill="#a3c6d4")
      draw.text((500, 49), "Same policy; contact timing matched; forces estimated", font=small_font, fill="#a3c6d4")
      real_im.thumbnail((420, 700))
      sim_im.thumbnail((700, 700))
      canvas.paste(real_im, (36 + (420 - real_im.width) // 2, 80))
      canvas.paste(sim_im, (500, 80))
      draw.text((36, 797), status, font=text_font, fill=color)
      draw.text((500, 800), "Timing replay, not a measured force reconstruction", font=small_font, fill="#a3c6d4")
      draw.rectangle((36, 838, 1230, 848), fill="#334454")
      for event in schedule["contacts"]:
        x0 = 36 + 1194 * event["start_s"] / duration
        x1 = 36 + 1194 * event["end_s"] / duration
        draw.rectangle((x0, 838, x1, 848), fill="#db725c")
      cursor = 36 + 1194 * t / duration
      draw.line((cursor, 833, cursor, 854), fill="white", width=3)
      comparison.append_data(np.asarray(canvas))
      # Standalone simulation keeps the same time origin and playback speed.
      single = Image.new("RGB", (720, 800), "#101923")
      single.paste(sim_im.resize((720, 720)), (0, 48))
      d = ImageDraw.Draw(single)
      d.text((12, 12), f"SIMULATION  |  {t:05.2f} s  |  {status}", font=small_font, fill=color)
      d.text((12, 775), "Hand-contact times matched; forces are approximate", font=small_font, fill="#a3c6d4")
      simulation.append_data(np.asarray(single))
      if frame % 60 == 0:
        print(f"Composed {t:.0f}/20 seconds", flush=True)
  real.close()
  sim.close()
  print("Created real-vs-simulation.mp4 and simulation-timed.mp4 (20 seconds, 30 fps).")


if __name__ == "__main__":
  main()
