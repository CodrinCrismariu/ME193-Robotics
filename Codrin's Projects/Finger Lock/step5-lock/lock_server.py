"""The sign lock on the laptop: serves the lock page and runs the two trained networks on webcam frames.

Run from this folder:   ..\\.venv\\Scripts\\python lock_server.py               then open http://localhost:8001
                        ..\\.venv\\Scripts\\python lock_server.py --classifier randomized_200

The page sends each webcam frame (320 x 240 JPEG) to /api/predict. The server
  1. runs the hand detector on the whole frame -> hand probability and box,
  2. if there is a hand, cuts the crop around the box and runs the letter classifier -> 7 probabilities,
and sends both back. The page runs the lock policy (see web/lock.js). Preprocessing uses the exact functions
from training (step4-models), so the networks see frames just as they did in training.
"""
import argparse
import base64
import io
import json
import re
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "step4-models"
sys.path.insert(0, str(MODELS))
import augment                                                    # noqa: E402  (training code, reused as is)
from data import LABELS, context_crop                             # noqa: E402
from detector import HandDetector, decode                         # noqa: E402
from models import LetterNet                                      # noqa: E402
from train_detector import SRC_H, SRC_W, view                     # noqa: E402

WEB = HERE / "web"
LOCK = threading.Lock()
NET = {}


def load(args):
    missing = [r for r in (args.detector, args.classifier) if not (MODELS / "runs" / r / "model.pt").exists()]
    if missing:
        have = sorted(p.parent.name for p in (MODELS / "runs").glob("*/model.pt"))
        raise SystemExit(f"No trained model yet for: {', '.join(missing)} (looked in step4-models/runs/<name>/model.pt).\n"
                         f"If training is still running, wait for it to finish. Trained models available now: {', '.join(have) or 'none'}.")
    det = HandDetector()
    det.load_state_dict(torch.load(MODELS / "runs" / args.detector / "model.pt", weights_only=False)["state_dict"])
    clf = LetterNet()
    clf.load_state_dict(torch.load(MODELS / "runs" / args.classifier / "model.pt", weights_only=False)["state_dict"])
    NET.update(det=det.eval(), clf=clf.eval(), min_hand=args.min_hand, names=(args.detector, args.classifier))
    info = {}
    for key, run in (("detector", args.detector), ("classifier", args.classifier)):
        m = MODELS / "runs" / run / "metrics.json"
        info[key] = {k: v for k, v in json.loads(m.read_text()).items() if k != "history"} if m.exists() else {}
    NET["info"] = info


@torch.no_grad()
def predict(img):
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].contiguous()
    inp, _ = view(x, torch.zeros(1, 4), train=False)
    boxes, prob = decode(NET["det"](augment.normalize(inp)))
    b = (boxes[0] * torch.tensor([SRC_W / inp.shape[3], SRC_H / inp.shape[2]] * 2)).tolist()
    b = [max(0.0, b[0]), max(0.0, b[1]), min(float(SRC_W), b[2]), min(float(SRC_H), b[3])]
    out = {"hand": float(prob[0]), "box": [round(v, 1) for v in b], "probs": None}
    if out["hand"] > NET["min_hand"] and b[2] - b[0] > 4 and b[3] - b[1] > 4:
        crop = torch.from_numpy(np.asarray(context_crop(img, b)))[None].permute(0, 3, 1, 2).contiguous()
        out["probs"] = [round(float(p), 4) for p in NET["clf"](augment.eval_batch(crop)).softmax(1)[0]]
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def reply(self, obj, code=200):
        data = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            pass                        # the page was reloaded or closed mid-request: nobody is waiting for the answer

    def do_GET(self):
        if urlparse(self.path).path == "/api/info":
            return self.reply({"labels": LABELS, "detector": NET["names"][0], "classifier": NET["names"][1],
                               "min_hand": NET["min_hand"], "metrics": NET["info"]})
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self.reply({"error": "unknown route"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > 2_000_000:
                raise ValueError("bad request size")
            m = re.match(r"^data:image/jpeg;base64,(.+)$", json.loads(self.rfile.read(n)).get("image", ""), re.S)
            if not m:
                raise ValueError("expected a JPEG data URL")
            img = Image.open(io.BytesIO(base64.b64decode(m.group(1)))).convert("RGB")
            if img.size != (SRC_W, SRC_H):
                img = img.resize((SRC_W, SRC_H), Image.BILINEAR)
            t0 = time.perf_counter()
            with LOCK:
                out = predict(img)
            out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self.reply(out)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            self.reply({"error": str(e)}, 400)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--detector", default="detector")
    p.add_argument("--classifier", default="randomized_200")
    p.add_argument("--min-hand", type=float, default=0.5, help="detector probability needed to count as a hand")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    load(a)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}/"
    print(f"Sign lock running at {url}   (detector: {a.detector}, classifier: {a.classifier})   Ctrl+C to stop")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
