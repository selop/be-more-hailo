"""Local control channel for testing the running agent.

A Unix socket (``logs/control.sock``, owner-only) accepts one JSON request
per connection and answers with one JSON line. It lets a tester on the
device — usually over SSH with ``scripts/bmoctl.py`` — run a turn without
speaking and read back what happened:

    {"cmd": "say", "text": "what time is it"}     run a turn from text
    {"cmd": "hear", "wav": "logs/audio/x.wav"}    run a turn from a recording
    {"cmd": "status"}                             phase, health, last turn

``say`` and ``hear`` answer with the finished turn's trace. Add
``"quiet": true`` to run the turn without playing sound.

Nothing listens on the network; the socket is a file only the service user
can open.
"""

import json
import os
import socket
import threading
import time

from . import trace
from .config import CHARACTER
from .log import bmo_print

SOCKET_PATH = os.environ.get("CONTROL_SOCKET", os.path.join(trace.LOG_DIR, "control.sock"))
TURN_TIMEOUT_SEC = 180

_started_at = time.time()


def _read_first_line(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.readline().strip()
    except OSError:
        return None


def system_health() -> dict:
    """Cheap host readings: CPU temperature, memory, load."""
    health = {}
    temp = _read_first_line("/sys/class/thermal/thermal_zone0/temp")
    if temp and temp.isdigit():
        health["cpu_temp_c"] = round(int(temp) / 1000, 1)
    try:
        with open("/proc/meminfo") as f:
            info = dict(line.split(":", 1) for line in f)
        health["mem_available_mb"] = int(info["MemAvailable"].split()[0]) // 1024
    except (OSError, KeyError, ValueError):
        pass
    try:
        health["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    return health


def status(conversation) -> dict:
    """Snapshot of the running agent."""
    from .npu import is_llm_ready
    from .stt import is_stt_npu
    store = conversation.store
    turns = trace.read("turns.jsonl", last=1)
    return {
        "character": CHARACTER.name,
        "pid": os.getpid(),
        "uptime_s": int(time.time() - _started_at),
        "phase": store.phase.value,
        "expression": store.expression,
        "muted": store.muted,
        "showing_image": store.display_image is not None,
        "timers": [
            {"remaining_s": round(timer["remaining_s"], 1), "total_s": timer["total_s"], "message": timer["message"]}
            for timer in getattr(conversation, "active_timers", lambda: [])()
        ],
        "llm_ready": is_llm_ready(),
        "stt_on_npu": is_stt_npu(),
        "history_messages": max(len(getattr(conversation.brain, "history", [])) - 1, 0),
        "last_turn": turns[0] if turns else None,
        **system_health(),
    }


class ControlServer:
    """Serves control requests for one :class:`core.turn.Conversation`."""

    def __init__(self, conversation, path: str = SOCKET_PATH):
        self.conversation = conversation
        self.path = path
        self._sock = None

    def start(self):
        """Open the socket and serve in a daemon thread. Failure to open it
        is logged and otherwise ignored: the agent works without it."""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            if os.path.exists(self.path):
                os.unlink(self.path)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(self.path)
            os.chmod(self.path, 0o600)
            sock.listen(4)
            self._sock = sock
        except OSError as e:
            bmo_print("CONTROL", f"Could not open control socket: {e}")
            return
        threading.Thread(target=self._serve, daemon=True).start()
        bmo_print("CONTROL", f"Listening on {self.path}")

    def _serve(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            with conn:
                raw = conn.makefile("r", encoding="utf-8").readline()
                try:
                    request = json.loads(raw)
                    reply = self.dispatch(request)
                except Exception as e:  # noqa: BLE001 - report every failure to the caller
                    reply = {"ok": False, "error": repr(e)}
                conn.sendall((json.dumps(reply, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
        except OSError:
            pass

    def dispatch(self, request: dict) -> dict:
        """Run one request and return the reply."""
        cmd = request.get("cmd")
        if cmd == "status":
            return {"ok": True, "status": status(self.conversation)}

        if cmd in ("say", "hear"):
            if cmd == "say":
                text = (request.get("text") or "").strip()
                if not text:
                    return {"ok": False, "error": "say needs a non-empty 'text'"}
                injection = self.conversation.inject(text=text, quiet=bool(request.get("quiet")))
            else:
                wav = request.get("wav") or ""
                if not os.path.isfile(wav):
                    return {"ok": False, "error": f"no such file: {wav}"}
                injection = self.conversation.inject(wav=os.path.abspath(wav), quiet=bool(request.get("quiet")))
            bmo_print("CONTROL", f"Injected turn ({cmd})")
            if not injection.done.wait(timeout=float(request.get("timeout", TURN_TIMEOUT_SEC))):
                return {"ok": False, "error": "turn did not finish in time (the agent may be busy or still booting)"}
            return {"ok": True, "turn": injection.result}

        return {"ok": False, "error": f"unknown cmd: {cmd!r}"}
