"""Local web app for building the sign-language dataset (letters C, O, D, R, I, N and "none").

Run:   python collect_server.py          then open http://localhost:8000 in Chrome or Edge
       python collect_server.py --port 8010 --no-browser

Two pages, both served from web/:
  /             the collector: streams the webcam and saves labeled photos
  /label.html   the box labeler: draw, check and fix the hand box on every photo

This server only saves files; it uses nothing outside the Python standard library and listens on
127.0.0.1 only.

Saved in dataset/:
  images/<id>.jpg     320x240 color photo, exactly as the camera saw it (not mirrored)
  samples.jsonl       one line per photo: id, label, box [x0, y0, x1, y1] in pixels (null = no hand),
                      checked (true once the box has been confirmed in the labeler), session, burst, time
"""
import argparse
import base64
import json
import re
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"
DATA = HERE / "dataset"
INDEX = DATA / "samples.jsonl"
LABELS = ["C", "O", "D", "R", "I", "N", "none"]
IMG_W, IMG_H = 320, 240
NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LOCK = threading.Lock()
MAX_BODY = 8_000_000


def read_index():
    if not INDEX.exists():
        return []
    return [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_index(rows):
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for attempt in range(20):                     # Windows: another program (indexer, antivirus) can hold the file for a moment
        try:
            tmp.replace(INDEX)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    tmp.replace(INDEX)


def data_url_bytes(s, kinds):
    m = re.match(r"^data:image/(png|jpeg);base64,(.+)$", s or "", re.S)
    if not m or m.group(1) not in kinds:
        raise ValueError("expected a base64 image data URL")
    return base64.b64decode(m.group(2))


def check_name(s, what):
    if not isinstance(s, str) or not NAME.match(s):
        raise ValueError(f"bad {what}")
    return s


def check_box(b):
    """A box is [x0, y0, x1, y1] in pixels of the 320x240 image, or None when there is no hand."""
    if b is None:
        return None
    if not isinstance(b, list) or len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
        raise ValueError("bad box")
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    x0, x1 = max(0, min(x0, x1)), min(IMG_W, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(IMG_H, max(y0, y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("box too small")
    return [x0, y0, x1, y1]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, fmt, *args):            # keep the terminal quiet except for saves
        pass

    # GET: the web pages, the saved photos, and the sample list
    def translate_path(self, path):
        p = unquote(urlparse(path).path)
        if p.startswith("/dataset/"):
            target = (DATA / p[len("/dataset/"):]).resolve()
            if DATA.resolve() not in target.parents:
                return str(WEB / "__nope__")
            return str(target)
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if urlparse(self.path).path == "/api/samples":
            with LOCK:
                return self.reply({"labels": LABELS, "samples": read_index()})
        return super().do_GET()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_BODY:
                raise ValueError("bad request size")
            body = json.loads(self.rfile.read(n))
            route = urlparse(self.path).path
            with LOCK:
                if route == "/api/save":
                    return self.reply(self.save_sample(body))
                if route == "/api/delete":
                    return self.reply(self.delete_sample(body))
                if route == "/api/box":
                    return self.reply(self.update_box(body))
            self.reply({"error": "unknown route"}, 404)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            self.reply({"error": str(e)}, 400)

    def reply(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── actions ──────────────────────────────────────────────────────────────
    def save_sample(self, b):
        label = b["label"]
        if label not in LABELS:
            raise ValueError("bad label")
        session = check_name(b["session"], "session")
        burst = check_name(b.get("burst") or "single", "burst")
        jpg = data_url_bytes(b["image"], {"jpeg"})
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rows = read_index()
        sid = f"{label}_{stamp}_{len(rows):05d}"
        (DATA / "images").mkdir(parents=True, exist_ok=True)
        (DATA / "images" / f"{sid}.jpg").write_bytes(jpg)
        row = {"id": sid, "label": label, "box": None, "checked": False, "session": session, "burst": burst,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with INDEX.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        counts = {l: 0 for l in LABELS}
        for r in rows + [row]:
            counts[r["label"]] += 1
        print(f"saved {sid}   ({label}: {counts[label]})")
        return {"sample": row, "counts": counts}

    def delete_sample(self, b):
        sid = check_name(b["id"], "id")
        rows = read_index()
        keep = [r for r in rows if r["id"] != sid]
        if len(keep) == len(rows):
            raise ValueError("no such sample")
        (DATA / "images" / f"{sid}.jpg").unlink(missing_ok=True)
        write_index(keep)
        print(f"deleted {sid}")
        return {"deleted": sid}

    def update_box(self, b):
        sid = check_name(b["id"], "id")
        box = check_box(b.get("box"))
        rows = read_index()
        for r in rows:
            if r["id"] == sid:
                r["box"] = box
                r["checked"] = bool(b.get("checked", True))
                write_index(rows)
                print(f"box {sid}: {box}")
                return {"sample": r}
        raise ValueError("no such sample")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--data", default=None, help="dataset folder (default: dataset/ next to this script)")
    a = p.parse_args()
    global DATA, INDEX
    if a.data:
        DATA = Path(a.data).resolve()
        INDEX = DATA / "samples.jsonl"
    DATA.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}/"
    print(f"Dataset collector running at {url}   box labeler at {url}label.html   (saving to {DATA})   Ctrl+C to stop")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
