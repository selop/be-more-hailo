"""Tests for the live-face bridge (core/display_server.py). No browser needed.

Run with ``pytest tests/test_display.py`` or ``python tests/test_display.py``.
"""

import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request

os.environ["TRACE_DIR"] = tempfile.mkdtemp(prefix="bmo-test-trace-")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.display_server import DisplayServer, StateMirror, list_face_frames, speech_envelope  # noqa: E402
from core.state import Phase, StateStore  # noqa: E402


def serve(on_action=None):
    display = DisplayServer(port=0, on_action=on_action)      # port 0: any free port
    assert display.start()
    return display


def get(display, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{display.port}{path}", timeout=5) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


class EventStream:
    """A minimal Server-Sent Events reader."""

    def __init__(self, display):
        self.sock = socket.create_connection(("127.0.0.1", display.port), timeout=5)
        self.sock.sendall(b"GET /events HTTP/1.1\r\nHost: x\r\n\r\n")
        self.buffer = b""

    def next_event(self, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            while b"\n\n" in self.buffer:
                block, self.buffer = self.buffer.split(b"\n\n", 1)
                lines = block.decode().split("\n")
                name = next((l[7:] for l in lines if l.startswith("event: ")), None)
                data = next((l[6:] for l in lines if l.startswith("data: ")), None)
                if name:
                    return name, json.loads(data)
            self.buffer += self.sock.recv(4096)
            self.buffer = self.buffer.replace(b"\r\n", b"\n")
        raise AssertionError("no event arrived")

    def close(self):
        self.sock.close()


def test_page_library_and_configuration_are_served():
    display = serve()
    try:
        status, content_type, body = get(display, "/")
        assert status == 200 and "text/html" in content_type and b"three" in body
        status, content_type, body = get(display, "/vendor/three.module.min.js")
        assert status == 200 and "javascript" in content_type and len(body) > 100_000
        config = json.loads(get(display, "/config.json")[2])
        assert config["name"] and config["theme"]["face_bg"].startswith("#")
        assert "idle" in config["faces"] and config["screensaver"]
        assert config["frame_ms"]["speaking"] == 90
    finally:
        display.stop()


def test_face_frames_are_served_and_nothing_outside_the_folders():
    display = serve()
    try:
        frames = list_face_frames()
        status, content_type, body = get(display, f"/faces/idle/{frames['idle'][0]}")
        assert status == 200 and content_type == "image/png" and body[:4] == b"\x89PNG"
        for path in ("/faces/../core/config.py", "/../CLAUDE.md", "/vendor/../../CLAUDE.md", "/nope.html"):
            try:
                get(display, path)
                assert False, f"{path} should not be served"
            except urllib.error.HTTPError as e:
                assert e.code == 404
    finally:
        display.stop()


def test_events_reach_the_page_and_state_is_replayed_to_late_joiners():
    display = serve()
    try:
        display.publish("state", phase="idle", expression="idle", muted=False, status="Ready")
        stream = EventStream(display)
        assert stream.next_event() == ("state", {"phase": "idle", "expression": "idle", "muted": False, "status": "Ready"})
        for _ in range(100):                    # wait until the page counts as connected
            if display.connected:
                break
            time.sleep(0.01)
        display.publish("meter", level=0.5)
        assert stream.next_event() == ("meter", {"level": 0.5})
        # An unchanged state is not sent again; a changed one is
        display.publish("state", phase="idle", expression="idle", muted=False, status="Ready")
        display.publish("state", phase="speaking", expression="speaking", muted=False, status="Speaking...")
        assert stream.next_event()[1]["phase"] == "speaking"
        stream.close()
    finally:
        display.stop()


def test_touch_actions_reach_the_engine():
    received = []
    display = serve(on_action=received.append)
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{display.port}/action", data=json.dumps({"type": "tap"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
        assert received == [{"type": "tap"}]
    finally:
        display.stop()


def test_pictures_are_published_and_cleared():
    from PIL import Image
    display = serve()
    try:
        stream = EventStream(display)
        for _ in range(100):
            if display.connected:
                break
            time.sleep(0.01)
        display.show_image(Image.new("RGB", (80, 48), "#336699"))
        name, data = stream.next_event()
        assert name == "image" and data["url"].startswith("/media/image.png")
        status, content_type, body = get(display, data["url"])
        assert status == 200 and body[:4] == b"\x89PNG"
        display.clear_image()
        assert stream.next_event() == ("image", {"url": None})
        stream.close()
    finally:
        display.stop()


def test_speech_envelope_follows_loudness():
    import numpy as np
    rate = 22050
    quiet = np.zeros(rate // 2, dtype=np.int16)
    loud = (np.sin(np.arange(rate // 2) * 0.2) * 20000).astype(np.int16)
    envelope = speech_envelope(np.concatenate([quiet, loud, quiet]).tobytes(), rate, step_ms=20)
    assert len(envelope) == 75                               # 1.5 s in 20 ms steps
    assert max(envelope[:20]) == 0 and min(envelope[30:45]) > 0.9 and max(envelope[55:]) == 0
    assert speech_envelope(b"", rate) == []
    assert set(speech_envelope(quiet.tobytes(), rate)) == {0.0}


def test_mirror_reports_state_timers_and_cleared_pictures():
    sent = []

    class Recorder:
        def publish(self, event, **data):
            sent.append((event, data))

        def clear_image(self):
            sent.append(("image", {"url": None}))

    store = StateStore()
    store.set_phase(Phase.LISTENING, msg="Listening...")
    timers = [{"remaining_s": 41.6, "total_s": 60.0, "message": "x"}]
    StateMirror(Recorder(), store, lambda: "Listening...", lambda: timers).snapshot()
    assert ("state", {"phase": "listening", "expression": "listening", "muted": False, "status": "Listening..."}) in sent
    assert ("timers", {"timers": [{"ends_in_s": 42, "total_s": 60.0}]}) in sent
    assert ("image", {"url": None}) in sent


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
