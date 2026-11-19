"""Tests for the monitoring readings (core/health.py) and the laptop monitor's logic.

Run with ``pytest tests/test_monitor.py`` or ``python tests/test_monitor.py``.
"""

import os
import sys
import tempfile

os.environ["TRACE_DIR"] = tempfile.mkdtemp(prefix="bmo-test-trace-")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import bmo_monitor  # noqa: E402
from core.health import MicMonitor, system_health  # noqa: E402


def healthy_sample(**mic):
    microphone = {"listening": True, "chunks": 300, "level_now": 90, "noise_floor": 70, "level_median": 85,
                  "level_peak": 2400, "wake_threshold": 0.35, "wake_score_now": 0.01, "wake_score_max": 0.03,
                  "detections": 2, "last_audio_age_s": 0.1}
    microphone.update(mic)
    return {
        "t": "2026-11-19T18:00:00",
        "system": {"cpu_temp_c": 55.0, "fan_step": 1, "fan_rpm": 3900, "throttled": "0x0", "throttled_now": False,
                   "under_voltage_now": False, "throttled_since_boot": False, "under_voltage_since_boot": False,
                   "mem_available_mb": 6000, "mem_total_mb": 8062, "load_1m": 1.2, "wifi_dbm": -52, "wifi_link": 58},
        "agent": {"phase": "idle", "expression": "idle", "muted": False, "llm_ready": True, "stt_on_npu": True,
                  "agent_uptime_s": 100, "timers": 0},
        "mic": microphone,
        "last_turn": None,
    }


def test_system_health_never_fails_and_reports_memory_or_nothing():
    health = system_health()
    assert isinstance(health, dict)
    for key, value in health.items():
        assert value is not None, key


def test_mic_monitor_tracks_noise_floor_peak_and_wake_scores():
    mic = MicMonitor(chunk_seconds=0.08, window_seconds=8)
    mic.threshold = 0.35
    assert mic.snapshot() == {"listening": False, "chunks": 0}
    for i in range(100):
        loud = i in (40, 41, 42)
        mic.heard(3000 if loud else 60 + (i % 5), 0.3 if loud else 0.01, suppressed=False)
    snapshot = mic.snapshot()
    assert snapshot["listening"] and snapshot["chunks"] == 100
    assert 60 <= snapshot["noise_floor"] <= 64 and snapshot["level_peak"] == 3000
    assert snapshot["wake_score_max"] == 0.3 and snapshot["wake_threshold"] == 0.35
    assert snapshot["detections"] == 0
    mic.detected(0.71)
    assert mic.snapshot()["detections"] == 1 and mic.snapshot()["last_detection_score"] == 0.71


def test_suppressed_audio_is_counted_but_not_scored():
    mic = MicMonitor()
    for _ in range(10):
        mic.heard(500, None, suppressed=True)
    snapshot = mic.snapshot()
    assert snapshot["suppressed_share"] == 1.0 and snapshot["wake_score_max"] is None


def test_a_healthy_sample_raises_no_alerts():
    assert bmo_monitor.alerts(healthy_sample()) == []


def test_alerts_name_the_problem():
    def has(sample, text):
        found = bmo_monitor.alerts(sample)
        assert any(text in a for a in found), (text, found)

    has({"unreachable": True}, "unreachable")
    hot = healthy_sample(); hot["system"]["cpu_temp_c"] = 81.0
    has(hot, "very hot")
    throttled = healthy_sample(); throttled["system"]["throttled_now"] = True
    has(throttled, "throttled right now")
    down = healthy_sample(); down["agent"] = None; down["agent_error"] = "no socket"
    has(down, "agent not running")
    slow = healthy_sample(); slow["agent"]["stt_on_npu"] = False
    has(slow, "slow CPU path")
    has(healthy_sample(noise_floor=650), "noise floor 650")
    has(healthy_sample(wake_score_max=0.29), "nearly heard")
    has(healthy_sample(listening=False, last_audio_age_s=12.0), "no microphone audio")
    has(healthy_sample(level_peak=95), "nothing above the noise floor")


def test_a_running_turn_is_not_reported_as_a_dead_microphone():
    busy = healthy_sample(listening=False, last_audio_age_s=6.0)
    busy["agent"]["phase"] = "speaking"
    assert not any("microphone" in a for a in bmo_monitor.alerts(busy))


def test_summary_and_sparkline():
    samples = []
    for i, temp in enumerate([55.0, 57.0, 61.0]):
        s = healthy_sample(noise_floor=70 + i * 10, detections=2 + i)
        s["system"]["cpu_temp_c"] = temp
        samples.append(s)
    samples.append({"t": "x", "unreachable": True})
    summary = bmo_monitor.summarise(samples)
    assert summary["samples"] == 4 and summary["unreachable_samples"] == 1
    assert summary["cpu_temp_c"] == {"min": 55.0, "median": 57.0, "max": 61.0}
    assert summary["mic_noise_floor"]["max"] == 90 and summary["wake_detections"] == 2
    assert summary["alerts"].get("Pi unreachable") == 1
    assert bmo_monitor.sparkline([1, 2, 3]) == "▁▄█" and bmo_monitor.sparkline([]) == ""
    assert "No problems detected" in bmo_monitor.render(healthy_sample(), samples[:1], plain=True)


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
