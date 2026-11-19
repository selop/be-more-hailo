"""Turn traces and events — the record a remote tester reads.

Every turn is written as one JSON line to ``logs/turns.jsonl`` with what was
heard, how it was routed, what the model produced, what was spoken and how
long each step took. Things that happen outside a turn (boot, timers, idle
thoughts, errors) go to ``logs/events.jsonl``. The recorded utterance is
copied to ``logs/audio/<turn id>.wav`` so a turn can be replayed.

Everything stays on the device. Set ``TRACE_AUDIO_KEEP=0`` to keep no
recordings, or ``TRACE_ENABLED=0`` to write nothing at all.

Usage inside the engine::

    from . import trace
    trace.mark("stt_done")              # time since the turn started
    trace.set("transcript", text)       # one value
    trace.add("spoken", {"text": ...})  # append to a list

All three are no-ops when no turn is active, so instrumented code also runs
unchanged from timers, the screensaver or tests.
"""

import json
import os
import shutil
import threading
import time

from .character import PROJECT_ROOT

LOG_DIR = os.environ.get("TRACE_DIR", os.path.join(PROJECT_ROOT, "logs"))
ENABLED = os.environ.get("TRACE_ENABLED", "1") != "0"
AUDIO_KEEP = int(os.environ.get("TRACE_AUDIO_KEEP", "50"))
MAX_FILE_BYTES = 5 * 1024 * 1024

_lock = threading.Lock()
_current = None          # the active TurnTrace, if any
_turn_counter = 0


class TurnTrace:
    """Timings and facts for one turn. Times are milliseconds since start."""

    def __init__(self, source: str):
        global _turn_counter
        _turn_counter += 1
        self._t0 = time.time()
        self.data = {
            "id": time.strftime("%Y%m%d-%H%M%S", time.localtime(self._t0)) + f"-{_turn_counter:03d}",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._t0)),
            "source": source,           # "voice", "text" or "wav"
            "marks": {},
        }

    def ms(self) -> int:
        return int((time.time() - self._t0) * 1000)

    def mark(self, name: str):
        self.data["marks"].setdefault(name, self.ms())

    def set(self, key: str, value):
        self.data[key] = value

    def add(self, key: str, value):
        self.data.setdefault(key, []).append(value)


# ── Module-level API (safe when no turn is active) ───────────────────────

def start_turn(source: str = "voice") -> TurnTrace:
    """Begin tracing a turn and make it the active one."""
    global _current
    _current = TurnTrace(source)
    return _current


def current() -> TurnTrace | None:
    return _current


def mark(name: str):
    t = _current
    if t is not None:
        t.mark(name)


def set(key: str, value):  # noqa: A001 - mirrors TurnTrace.set
    t = _current
    if t is not None:
        t.set(key, value)


def add(key: str, value):
    t = _current
    if t is not None:
        t.add(key, value)


def ms() -> int | None:
    t = _current
    return t.ms() if t is not None else None


def finish_turn(status: str = "ok", error: str | None = None) -> dict | None:
    """Close the active turn, write it to ``turns.jsonl`` and return it."""
    global _current
    t = _current
    _current = None
    if t is None:
        return None
    t.data["status"] = status
    if error:
        t.data["error"] = error
    t.data["total_ms"] = t.ms()
    marks = t.data["marks"]
    # The wait the user feels: from the end of their utterance to first audio
    if "first_audio" in marks:
        heard = marks.get("record_done", 0)
        t.data["response_ms"] = marks["first_audio"] - heard
    _write("turns.jsonl", t.data)
    return t.data


def keep_audio(wav_path: str):
    """Copy the turn's recording into ``logs/audio`` and prune old ones."""
    t = _current
    if t is None or not ENABLED or AUDIO_KEEP <= 0 or not wav_path:
        return
    try:
        audio_dir = os.path.join(LOG_DIR, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        dest = os.path.join(audio_dir, t.data["id"] + ".wav")
        shutil.copyfile(wav_path, dest)
        t.set("audio_file", os.path.relpath(dest, PROJECT_ROOT))
        kept = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))
        for old in kept[:-AUDIO_KEEP]:
            os.unlink(os.path.join(audio_dir, old))
    except OSError:
        pass


def event(name: str, **fields):
    """Record something that happened outside a turn (boot step, timer,
    idle thought, error)."""
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": name}
    record.update(fields)
    _write("events.jsonl", record)


# ── Storage ──────────────────────────────────────────────────────────────

def _write(filename: str, record: dict):
    if not ENABLED:
        return
    try:
        with _lock:
            os.makedirs(LOG_DIR, exist_ok=True)
            path = os.path.join(LOG_DIR, filename)
            if os.path.exists(path) and os.path.getsize(path) > MAX_FILE_BYTES:
                os.replace(path, path + ".1")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def read(filename: str = "turns.jsonl", last: int | None = None) -> list[dict]:
    """Return records from a trace file, oldest first."""
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[-last:] if last else records
