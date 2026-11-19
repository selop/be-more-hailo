"""Host and microphone readings for monitoring.

``system_health()`` reads the board (temperature, fan, throttling, memory,
load, Wi-Fi, process sizes). ``MicMonitor`` keeps a rolling picture of what
the wake-word loop hears: level, noise floor and wake scores. Standard
library plus numpy; nothing here talks to the NPU.
"""

import os
import subprocess
import threading
import time
from collections import deque


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _first_existing(pattern_dir: str, filename: str) -> str | None:
    """Value of ``<pattern_dir>/*/<filename>`` for the first sub-folder that has it."""
    try:
        for entry in sorted(os.listdir(pattern_dir)):
            value = _read(os.path.join(pattern_dir, entry, filename))
            if value is not None:
                return value
    except OSError:
        pass
    return None


def _vcgencmd(*args) -> str | None:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def _process_rss_mb() -> dict:
    """Resident memory of the agent and of the browser showing the face."""
    sizes = {"agent_mb": 0, "browser_mb": 0}
    try:
        out = subprocess.run(["ps", "-eo", "rss=,args="], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.TimeoutExpired):
        return sizes
    for line in out.splitlines():
        rss, _, args = line.strip().partition(" ")
        if not rss.isdigit():
            continue
        if "agent_hailo.py" in args and "python" in args:
            sizes["agent_mb"] += int(rss) // 1024
        elif "chromium" in args:
            sizes["browser_mb"] += int(rss) // 1024
    return sizes


def system_health() -> dict:
    """One snapshot of the host. Missing sensors are simply left out."""
    health = {}
    temp = _read("/sys/class/thermal/thermal_zone0/temp")
    if temp and temp.isdigit():
        health["cpu_temp_c"] = round(int(temp) / 1000, 1)

    fan_step = _read("/sys/class/thermal/cooling_device0/cur_state")
    if fan_step and fan_step.isdigit():
        health["fan_step"] = int(fan_step)
    fan_rpm = _first_existing("/sys/devices/platform/cooling_fan/hwmon", "fan1_input")
    if fan_rpm and fan_rpm.isdigit():
        health["fan_rpm"] = int(fan_rpm)

    throttled = _vcgencmd("get_throttled")          # "throttled=0x0"
    if throttled and "=" in throttled:
        flags = int(throttled.split("=", 1)[1], 16)
        health["throttled"] = hex(flags)
        health["throttled_now"] = bool(flags & 0x4)
        health["under_voltage_now"] = bool(flags & 0x1)
        health["throttled_since_boot"] = bool(flags & 0x40000)
        health["under_voltage_since_boot"] = bool(flags & 0x10000)
    clock = _vcgencmd("measure_clock", "arm")       # "frequency(0)=1700000000"
    if clock and "=" in clock:
        health["cpu_mhz"] = int(clock.split("=", 1)[1]) // 1_000_000

    try:
        with open("/proc/meminfo") as f:
            info = dict(line.split(":", 1) for line in f)
        health["mem_available_mb"] = int(info["MemAvailable"].split()[0]) // 1024
        health["mem_total_mb"] = int(info["MemTotal"].split()[0]) // 1024
    except (OSError, KeyError, ValueError):
        pass
    try:
        health["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass

    wireless = _read("/proc/net/wireless")          # third line: iface, status, link, level, noise
    if wireless:
        for line in wireless.splitlines()[2:]:
            parts = line.split()
            if len(parts) >= 4:
                health["wifi_link"] = int(float(parts[2]))
                health["wifi_dbm"] = int(float(parts[3]))
                break

    uptime = _read("/proc/uptime")
    if uptime:
        health["uptime_s"] = int(float(uptime.split()[0]))
    health.update(_process_rss_mb())
    return health


class MicMonitor:
    """Rolling statistics of the wake-word loop's audio, about 30 seconds deep.

    The loop calls :meth:`heard` for every chunk. ``snapshot()`` answers the
    questions that matter when the wake word is not getting through: is
    audio arriving, how loud is the room, and how close did the wake word
    come to the threshold.
    """

    def __init__(self, chunk_seconds: float = 0.08, window_seconds: float = 30.0):
        self._window = deque(maxlen=int(window_seconds / chunk_seconds))   # (rms, score, suppressed)
        self._lock = threading.Lock()
        self._last_chunk_at = 0.0
        self._chunks = 0
        self._detections = 0
        self._last_detection_at = 0.0
        self._last_detection_score = None
        self.threshold = None

    def heard(self, rms: float, score: float | None, suppressed: bool):
        with self._lock:
            self._window.append((float(rms), score, suppressed))
            self._last_chunk_at = time.time()
            self._chunks += 1

    def detected(self, score: float):
        with self._lock:
            self._detections += 1
            self._last_detection_at = time.time()
            self._last_detection_score = round(float(score), 3)

    def snapshot(self) -> dict:
        with self._lock:
            window = list(self._window)
            last_chunk_at, chunks = self._last_chunk_at, self._chunks
            detections, last_at, last_score = self._detections, self._last_detection_at, self._last_detection_score
        if not window:
            return {"listening": False, "chunks": 0}
        levels = sorted(rms for rms, _, _ in window)
        scores = [score for _, score, _ in window if score is not None]
        now = time.time()
        snapshot = {
            # False while a turn is running: the mic then belongs to the recorder
            "listening": (now - last_chunk_at) < 1.0,
            "last_audio_age_s": round(now - last_chunk_at, 1),
            "chunks": chunks,
            # Levels as RMS of 16-bit samples (full scale 32768)
            "level_now": round(window[-1][0]),
            "noise_floor": round(levels[len(levels) // 5]),        # quietest fifth: the room without speech
            "level_median": round(levels[len(levels) // 2]),
            "level_peak": round(levels[-1]),
            "suppressed_share": round(sum(1 for _, _, s in window if s) / len(window), 2),
            "wake_threshold": self.threshold,
            "wake_score_now": round(scores[-1], 3) if scores else None,
            "wake_score_max": round(max(scores), 3) if scores else None,
            "detections": detections,
        }
        if last_at:
            snapshot["last_detection_age_s"] = round(now - last_at, 1)
            snapshot["last_detection_score"] = last_score
        return snapshot


mic_monitor = MicMonitor()
