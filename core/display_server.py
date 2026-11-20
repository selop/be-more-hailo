"""Serves the live face (``face/``) to a local browser and feeds it events.

The face is a web page drawn with three.js. This module is the bridge
between the engine and that page:

- static files: the page, its vendored libraries, and the picture currently
  on show;
- ``GET /events``: a Server-Sent Events stream of what the engine is doing
  (state, bubble text, mic level, speech loudness, timers, pictures);
- ``POST /action``: touches from the page (tap, long press).

It listens on 127.0.0.1 only. Standard library only, no Tkinter, so it can
be tested anywhere.
"""

import io
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .character import PROJECT_ROOT
from .config import CHARACTER
from .log import bmo_print

FACE_APP_DIR = os.path.join(PROJECT_ROOT, "face")
DEFAULT_PORT = int(os.environ.get("FACE_PORT", "8765"))

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".json": "application/json", ".png": "image/png", ".css": "text/css; charset=utf-8",
}

# Events whose latest value a newly connected page needs to catch up
_REPLAYED = ("state", "bubble", "timers", "image")


def speech_envelope(pcm_bytes: bytes, sample_rate: int = 22050, step_ms: int = 20) -> list[float]:
    """Loudness of int16 mono PCM in steps of *step_ms*, scaled 0…1 against
    the loudest step. The page plays this back to move the mouth."""
    import numpy as np
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    step = max(int(sample_rate * step_ms / 1000), 1)
    count = len(samples) // step
    if count == 0:
        return []
    rms = np.sqrt(np.mean(samples[:count * step].reshape(count, step) ** 2, axis=1))
    peak = float(rms.max())
    if peak < 1.0:
        return [0.0] * count
    return [round(float(v), 3) for v in np.clip(rms / peak, 0.0, 1.0)]


class DisplayServer:
    """HTTP server plus event fan-out for the face page.

    Parameters
    ----------
    port : int
    on_action : callable(dict) or None
        Called with the JSON body of every ``POST /action``.
    """

    def __init__(self, port: int = DEFAULT_PORT, on_action=None):
        self.port = port
        self.on_action = on_action
        self._clients = []                # one queue per connected page
        self._lock = threading.Lock()
        self._latest = {}                 # event name → last payload (for replay and de-duplication)
        self._image_png = None
        self._image_version = 0
        self._httpd = None
        self._browser = None
        self._browser_started = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start serving in a daemon thread. Returns False if the port is taken."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):       # keep the service log quiet
                pass

            def do_GET(self):
                server._handle_get(self)

            def do_POST(self):
                server._handle_post(self)

        try:
            self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        except OSError as e:
            bmo_print("FACE", f"Could not start the face server on port {self.port}: {e}")
            return False
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        bmo_print("FACE", f"Serving the face at http://127.0.0.1:{self.port}/")
        return True

    def stop(self):
        self.close_browser()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def open_browser(self) -> bool:
        """Show the page fullscreen in Chromium, on top of everything else."""
        binary = shutil.which("chromium") or shutil.which("chromium-browser")
        if binary is None:
            bmo_print("FACE", "Chromium is not installed; the live face cannot be shown")
            return False
        env = dict(os.environ)
        runtime = env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        cmd = [
            binary, "--kiosk", "--password-store=basic", "--no-first-run", "--noerrdialogs",
            "--disable-infobars", "--disable-session-crashed-bubble", "--overscroll-history-navigation=0",
            f"--user-data-dir={os.path.expanduser('~/.cache/live-face')}",
            f"http://127.0.0.1:{self.port}/?kiosk=1",
        ]
        if os.path.exists(os.path.join(runtime, "wayland-0")):
            env.setdefault("WAYLAND_DISPLAY", "wayland-0")
            cmd.insert(1, "--ozone-platform=wayland")
        try:
            self._browser = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            bmo_print("FACE", f"Could not start Chromium: {e}")
            return False
        self._browser_started = time.time()
        bmo_print("FACE", "Chromium started")
        return True

    def ensure_browser(self):
        """Start Chromium again if it has exited, or if its page never
        connected (seen at boot: a blank window). It is the only display."""
        # shortcut: retries forever at these intervals; add a limit if a broken install floods the log
        if self._browser is None:
            return
        age = time.time() - self._browser_started
        if self._browser.poll() is not None and age > 30:
            bmo_print("FACE", "Chromium exited; starting it again")
            self.open_browser()
        elif self.connected == 0 and age > 120:
            bmo_print("FACE", "The face page never connected; restarting Chromium")
            self.close_browser()
            self.open_browser()

    def close_browser(self):
        if self._browser is not None and self._browser.poll() is None:
            try:
                os.killpg(self._browser.pid, 15)
            except OSError:
                pass
        self._browser = None

    @property
    def connected(self) -> int:
        """Number of pages currently listening."""
        with self._lock:
            return len(self._clients)

    # ── Publishing ───────────────────────────────────────────────────────

    def publish(self, event: str, **data):
        """Send *event* to every connected page. For the replayed events
        (state, bubble, timers, image) an unchanged payload is not re-sent."""
        if event in _REPLAYED:
            if self._latest.get(event) == data:
                return
            self._latest[event] = data
        message = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.put_nowait(message)
            except queue.Full:
                pass        # a stalled page misses events; state is replayed when it reconnects

    def show_image(self, pil_image):
        """Put a picture (a PIL image) on the face's screen."""
        buffer = io.BytesIO()
        pil_image.convert("RGB").save(buffer, format="PNG")
        self._image_png = buffer.getvalue()
        self._image_version += 1
        self.publish("image", url=f"/media/image.png?v={self._image_version}")

    def clear_image(self):
        self.publish("image", url=None)

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _config(self) -> dict:
        return {
            "name": CHARACTER.name,
            "theme": {k: v for k, v in CHARACTER.theme.items() if isinstance(v, str)},
            "screensaver": CHARACTER.screensaver_faces,
        }

    def _send(self, handler, status, body: bytes, content_type: str):
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(body)

    def _send_file(self, handler, root: str, relative: str):
        path = os.path.realpath(os.path.join(root, relative))
        if not path.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(path):
            self._send(handler, 404, b"not found", "text/plain")
            return
        with open(path, "rb") as f:
            body = f.read()
        self._send(handler, 200, body, _CONTENT_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream"))

    def _handle_get(self, handler):
        path = handler.path.split("?", 1)[0]
        if path == "/events":
            self._serve_events(handler)
        elif path == "/config.json":
            self._send(handler, 200, json.dumps(self._config()).encode("utf-8"), "application/json")
        elif path == "/media/image.png":
            if self._image_png is None:
                self._send(handler, 404, b"no image", "text/plain")
            else:
                self._send(handler, 200, self._image_png, "image/png")
        else:
            self._send_file(handler, FACE_APP_DIR, "index.html" if path == "/" else path.lstrip("/"))

    def _handle_post(self, handler):
        if handler.path.split("?", 1)[0] != "/action":
            self._send(handler, 404, b"not found", "text/plain")
            return
        try:
            length = int(handler.headers.get("Content-Length", "0"))
            action = json.loads(handler.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(handler, 400, b"bad request", "text/plain")
            return
        if self.on_action is not None:
            try:
                self.on_action(action)
            except Exception as e:  # noqa: BLE001 - a touch must never take the server down
                bmo_print("FACE", f"Action {action} failed: {e!r}")
        self._send(handler, 200, b"{}", "application/json")

    def _serve_events(self, handler):
        client = queue.Queue(maxsize=200)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()
        try:
            # Catch the page up, then stream
            for event in _REPLAYED:
                if event in self._latest:
                    handler.wfile.write(f"event: {event}\ndata: {json.dumps(self._latest[event], ensure_ascii=False)}\n\n".encode("utf-8"))
            handler.wfile.flush()
            with self._lock:
                self._clients.append(client)
            while True:
                try:
                    message = client.get(timeout=15)
                except queue.Empty:
                    message = b": keep-alive\n\n"
                handler.wfile.write(message)
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self._lock:
                if client in self._clients:
                    self._clients.remove(client)


class StateMirror:
    """Watches the engine and reports changes to the face page: phase,
    expression, mute, status text, running timers, and when a picture
    stops being shown."""

    def __init__(self, display, store, get_status, get_timers, interval: float = 0.07):
        self.display = display
        self.store = store
        self.get_status = get_status
        self.get_timers = get_timers
        self.interval = interval
        self._stop = threading.Event()

    def snapshot(self):
        store = self.store
        self.display.publish(
            "state", phase=store.phase.value, expression=store.expression,
            muted=store.muted, status=self.get_status(),
        )
        self.display.publish("timers", timers=[
            {"ends_in_s": round(timer["remaining_s"]), "total_s": round(timer["total_s"], 1)}
            for timer in self.get_timers()
        ])
        if store.display_image is None:
            self.display.clear_image()

    def start(self):
        def loop():
            while not self._stop.wait(self.interval):
                try:
                    self.snapshot()
                except Exception as e:  # noqa: BLE001
                    bmo_print("FACE", f"Mirror error: {e!r}")
        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._stop.set()
