#!/usr/bin/env python3
"""Live monitor for the assistant, run on a development machine.

    python scripts/bmo_monitor.py                 # dashboard until Ctrl+C
    python scripts/bmo_monitor.py --duration 60   # watch for a minute, then print a summary
    python scripts/bmo_monitor.py --once          # one reading

It keeps one SSH connection to the Pi running ``bmoctl.py watch`` and shows
the host (temperature, fan, throttling, memory, Wi-Fi), the agent (phase,
models), the microphone (level, noise floor, wake-word score against its
threshold) and the last turn. Every sample is also appended to
``logs/monitor-<date>.jsonl`` here, and a dropped connection is recorded
and retried.

BMO_HOST selects the device (default selop@raspi), BMO_DIR its checkout.
Standard library only.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS = "▁▂▃▄▅▆▇█"

# Reading → (warn at, alarm at); higher is worse
LIMITS = {"cpu_temp_c": (65, 78), "load_1m": (3.5, 6.0)}


# ── Pure helpers (tested in tests/test_monitor.py) ────────────────────────

def alerts(sample: dict) -> list[str]:
    """Plain-language problems visible in one sample."""
    found = []
    if sample.get("unreachable"):
        return ["Pi unreachable"]
    system, mic, agent = sample.get("system") or {}, sample.get("mic") or {}, sample.get("agent")
    if system.get("throttled_now"):
        found.append("CPU is being throttled right now")
    if system.get("under_voltage_now"):
        found.append("under-voltage right now (power supply)")
    if system.get("throttled_since_boot"):
        found.append("CPU was throttled at some point since boot")
    if system.get("under_voltage_since_boot"):
        found.append("under-voltage occurred since boot")
    temp = system.get("cpu_temp_c")
    if temp is not None and temp >= LIMITS["cpu_temp_c"][1]:
        found.append(f"CPU very hot: {temp} °C")
    elif temp is not None and temp >= LIMITS["cpu_temp_c"][0]:
        found.append(f"CPU warm: {temp} °C")
    if system.get("mem_available_mb") is not None and system["mem_available_mb"] < 500:
        found.append(f"low memory: {system['mem_available_mb']} MB available")
    if system.get("wifi_dbm") is not None and system["wifi_dbm"] < -75:
        found.append(f"weak Wi-Fi: {system['wifi_dbm']} dBm")
    if agent is None:
        found.append("agent not running: " + str(sample.get("agent_error", "")))
        return found
    if not agent.get("llm_ready"):
        found.append("LLM not loaded")
    if agent.get("stt_on_npu") is False:
        found.append("speech recognition is on the slow CPU path")
    if agent.get("muted"):
        found.append("muted")
    idle = agent.get("phase") in ("idle", "screensaver")
    if mic and idle and not mic.get("listening"):
        found.append(f"no microphone audio for {mic.get('last_audio_age_s')} s while idle")
    threshold = mic.get("wake_threshold")
    floor, peak = mic.get("noise_floor"), mic.get("level_peak")
    if floor is not None and floor >= 400:
        found.append(f"noisy room or fan: noise floor {floor}")
    best = mic.get("wake_score_max")
    if threshold and best is not None and 0.5 * threshold <= best < threshold:
        found.append(f"wake word nearly heard: best score {best} against threshold {threshold}")
    if floor is not None and peak is not None and idle and mic.get("listening") and peak < max(2 * floor, 150):
        found.append("microphone hears nothing above the noise floor")
    return found


def sparkline(values: list, width: int = 30) -> str:
    values = [v for v in values[-width:] if v is not None]
    if not values:
        return ""
    low, high = min(values), max(values)
    if high == low:
        return BARS[0] * len(values)
    return "".join(BARS[int((v - low) / (high - low) * (len(BARS) - 1))] for v in values)


def series(samples: list, section: str, key: str) -> list:
    return [(s.get(section) or {}).get(key) for s in samples if not s.get("unreachable")]


def summarise(samples: list) -> dict:
    """Minimum, median and maximum of the main readings over a run."""
    def stats(section, key):
        values = [v for v in series(samples, section, key) if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not values:
            return None
        return {"min": min(values), "median": round(statistics.median(values), 2), "max": max(values)}

    reachable = [s for s in samples if not s.get("unreachable")]
    detections = [d for d in series(samples, "mic", "detections") if d is not None]
    seen = {}
    for s in samples:
        for a in alerts(s):
            key = a.split(":")[0]
            seen[key] = seen.get(key, 0) + 1
    return {
        "samples": len(samples),
        "unreachable_samples": len(samples) - len(reachable),
        "cpu_temp_c": stats("system", "cpu_temp_c"),
        "fan_rpm": stats("system", "fan_rpm"),
        "fan_step": stats("system", "fan_step"),
        "load_1m": stats("system", "load_1m"),
        "mem_available_mb": stats("system", "mem_available_mb"),
        "wifi_dbm": stats("system", "wifi_dbm"),
        "mic_noise_floor": stats("mic", "noise_floor"),
        "mic_level_peak": stats("mic", "level_peak"),
        "wake_score_max": stats("mic", "wake_score_max"),
        "wake_detections": (detections[-1] - detections[0]) if len(detections) > 1 else 0,
        "alerts": seen,
    }


# ── Display ───────────────────────────────────────────────────────────────

def colour(text, code, plain):
    return text if plain else f"\033[{code}m{text}\033[0m"


def level_colour(key, value):
    warn, alarm = LIMITS.get(key, (None, None))
    if value is None or warn is None:
        return "0"
    return "31" if value >= alarm else "33" if value >= warn else "32"


def render(sample: dict, history: list, plain: bool) -> str:
    lines = []
    title = f"BMO monitor   {sample.get('t', '')}   ({len(history)} samples)"
    lines.append(colour(title, "1", plain))
    if sample.get("unreachable"):
        lines.append(colour("  Pi unreachable — retrying", "31;1", plain))
        return "\n".join(lines)

    system, mic, agent, last = sample.get("system") or {}, sample.get("mic") or {}, sample.get("agent"), sample.get("last_turn")
    temp = system.get("cpu_temp_c")
    lines.append("")
    lines.append(colour("Board", "1", plain))
    lines.append(f"  temperature  {colour(str(temp) + ' °C', level_colour('cpu_temp_c', temp), plain):<10}  {sparkline(series(history, 'system', 'cpu_temp_c'))}")
    lines.append(f"  fan          step {system.get('fan_step')}  {system.get('fan_rpm')} rpm   {sparkline(series(history, 'system', 'fan_rpm'))}")
    throttle = "no" if system.get("throttled") == "0x0" else colour(str(system.get("throttled")), "31", plain)
    lines.append(f"  throttling   {throttle}     clock {system.get('cpu_mhz')} MHz     load {system.get('load_1m')}")
    lines.append(f"  memory       {system.get('mem_available_mb')} MB free of {system.get('mem_total_mb')}   agent {system.get('agent_mb')} MB   browser {system.get('browser_mb')} MB")
    lines.append(f"  wifi         {system.get('wifi_dbm')} dBm (link {system.get('wifi_link')})")

    lines.append("")
    lines.append(colour("Agent", "1", plain))
    if agent is None:
        lines.append(colour(f"  not running: {sample.get('agent_error')}", "31", plain))
    else:
        lines.append(f"  phase        {agent.get('phase')}  (face: {agent.get('expression')})" + ("   MUTED" if agent.get("muted") else ""))
        lines.append(f"  models       LLM {'ready' if agent.get('llm_ready') else 'NOT LOADED'}   speech recognition on {'NPU' if agent.get('stt_on_npu') else 'CPU'}"
                     f"   up {agent.get('agent_uptime_s')} s   timers {agent.get('timers')}")

        lines.append("")
        lines.append(colour("Microphone (what the wake-word loop hears, last 30 s)", "1", plain))
        if not mic or not mic.get("chunks"):
            lines.append("  no audio seen yet")
        else:
            state = "listening" if mic.get("listening") else f"paused ({mic.get('last_audio_age_s')} s since last audio)"
            lines.append(f"  state        {state}")
            lines.append(f"  level        now {mic.get('level_now')}   noise floor {mic.get('noise_floor')}   peak {mic.get('level_peak')}   {sparkline(series(history, 'mic', 'level_now'))}")
            lines.append(f"  wake word    now {mic.get('wake_score_now')}   best {mic.get('wake_score_max')}   threshold {mic.get('wake_threshold')}   {sparkline(series(history, 'mic', 'wake_score_max'))}")
            heard = f"{mic.get('detections')} detections"
            if mic.get("last_detection_age_s") is not None:
                heard += f", last {mic['last_detection_age_s']} s ago (score {mic.get('last_detection_score')})"
            lines.append(f"               {heard}")

    if last:
        lines.append("")
        lines.append(colour("Last turn", "1", plain))
        response = last.get("response_ms")
        lines.append(f"  {last.get('id')}  [{last.get('source')}]  {last.get('status')}"
                     + (f"   response {response / 1000:.1f} s" if response is not None else ""))
        lines.append(f"  heard: {last.get('transcript')!r}")
        if last.get("said"):
            lines.append(f"  said:  {last['said']!r}")

    problems = alerts(sample)
    lines.append("")
    if problems:
        lines.append(colour("Attention", "33;1", plain))
        lines.extend(colour(f"  • {p}", "33", plain) for p in problems)
    else:
        lines.append(colour("No problems detected", "32", plain))
    return "\n".join(lines)


def print_summary(samples: list):
    print("\nSummary")
    summary = summarise(samples)
    for key, value in summary.items():
        if isinstance(value, dict) and set(value) == {"min", "median", "max"}:
            print(f"  {key:<18} min {value['min']}   median {value['median']}   max {value['max']}")
        else:
            print(f"  {key:<18} {value}")


# ── Collection ────────────────────────────────────────────────────────────

def stream(host: str, directory: str, interval: float):
    """Yield samples from the Pi. Yields ``{"unreachable": True}`` and
    reconnects whenever the SSH connection drops."""
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=4",
               "-o", "ServerAliveCountMax=2", host,
               f"cd {directory} && venv/bin/python -u scripts/bmoctl.py watch -i {interval}"]
    while True:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            for line in process.stdout:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        finally:
            process.kill()
        yield {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "unreachable": True}
        time.sleep(3)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Live monitor for the assistant on the Pi.")
    parser.add_argument("--host", default=os.environ.get("BMO_HOST", "selop@raspi"))
    parser.add_argument("--dir", default=os.environ.get("BMO_DIR", "Git/be-more-hailo"))
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between samples")
    parser.add_argument("--duration", type=float, default=0, help="stop after this many seconds and print a summary")
    parser.add_argument("--once", action="store_true", help="print one reading and exit")
    parser.add_argument("--plain", action="store_true", help="no colours, no screen clearing")
    parser.add_argument("--log", default=None, help="where to append samples (default logs/monitor-<date>.jsonl)")
    args = parser.parse_args(argv)

    plain = args.plain or args.once or not sys.stdout.isatty()
    log_path = args.log or os.path.join(ROOT, "logs", f"monitor-{time.strftime('%Y%m%d')}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    samples, started = [], time.time()
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            for sample in stream(args.host, args.dir, args.interval):
                samples.append(sample)
                log.write(json.dumps(sample, ensure_ascii=False) + "\n")
                log.flush()
                if args.once:
                    print(render(sample, samples, plain=True))
                    return 1 if sample.get("unreachable") else 0
                if args.duration:
                    if plain and not sample.get("unreachable"):
                        mic, system = sample.get("mic") or {}, sample.get("system") or {}
                        print(f"{sample.get('t')}  {system.get('cpu_temp_c')}°C fan {system.get('fan_step')}/{system.get('fan_rpm')}  "
                              f"mic now {mic.get('level_now')} floor {mic.get('noise_floor')} peak {mic.get('level_peak')}  "
                              f"wake now {mic.get('wake_score_now')} best {mic.get('wake_score_max')}  "
                              f"phase {(sample.get('agent') or {}).get('phase')}", flush=True)
                    elif plain:
                        print(f"{sample.get('t')}  unreachable", flush=True)
                else:
                    sys.stdout.write(("" if plain else "\033[2J\033[H") + render(sample, samples, plain) + "\n")
                    sys.stdout.flush()
                if args.duration and time.time() - started >= args.duration:
                    break
    except KeyboardInterrupt:
        pass
    if samples and not args.once:
        print_summary(samples)
        print(f"\nSamples saved to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
