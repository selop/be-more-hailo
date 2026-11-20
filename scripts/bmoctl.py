#!/usr/bin/env python3
"""Observe and drive the running agent from a shell on the device.

    bmoctl.py status                    phase, health, last turn
    bmoctl.py say "what time is it"     run a turn from text, print its trace
    bmoctl.py hear logs/audio/X.wav     run a turn from a recording
    bmoctl.py turns [-n 10]             table of recent turns
    bmoctl.py show [last|<turn id>]     one turn in full
    bmoctl.py stats [-n 50]             timing summary over recent turns
    bmoctl.py events [-n 20]            boot, timer, music and error events
    bmoctl.py screenshot [file.png]     capture what is on the display
    bmoctl.py watch [-i 2]              one JSON line of readings every 2 s (for the monitor)

Add --quiet to say/hear to run the turn without sound. Add --json to any
command for machine-readable output. From another machine use
scripts/bmo-remote.sh, which runs this over SSH.

Standard library only.
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import trace  # noqa: E402  (stdlib-only module)

SOCKET_PATH = os.environ.get("CONTROL_SOCKET", os.path.join(trace.LOG_DIR, "control.sock"))


def request(payload: dict, timeout: float = 200) -> dict:
    """Send one request to the running agent and return its reply."""
    if not os.path.exists(SOCKET_PATH):
        return {"ok": False, "error": f"agent is not running or still starting (no {SOCKET_PATH})"}
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(SOCKET_PATH)
            s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            return json.loads(s.makefile("r", encoding="utf-8").readline())
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"control channel failed: {e!r}"}


# ── Formatting ───────────────────────────────────────────────────────────

def _sec(ms):
    return "-" if ms is None else f"{ms / 1000:.1f}s"


def _reply_text(turn: dict) -> str:
    return " ".join(s.get("text", "") for s in turn.get("spoken", []))


def print_turn(turn: dict):
    """Human-readable view of one turn."""
    marks = turn.get("marks", {})
    print(f"turn {turn.get('id')}  [{turn.get('source')}]  status={turn.get('status')}")
    if turn.get("error"):
        print(f"  error:      {turn['error']}")
    print(f"  heard:      {turn.get('transcript')!r}")
    print(f"  route:      {turn.get('route', '-')}" + (f"  actions={turn['actions']}" if turn.get("actions") else ""))
    if turn.get("search"):
        s = turn["search"]
        print(f"  search:     {_sec(s.get('ms'))}  used={s.get('used')}  {s.get('result', '')[:110]!r}")
    if "llm_raw" in turn:
        print(f"  llm said:   {turn['llm_raw']!r}")
    for s in turn.get("spoken", []):
        extra = f"  ERROR {s['error']}" if s.get("error") else f"  (synth {s.get('synth_ms')}ms, {s.get('audio_s')}s audio)"
        print(f"  spoke:      @{_sec(s.get('at_ms'))} {s.get('text')!r}{extra}")
    print(
        "  timing:     "
        f"response={_sec(turn.get('response_ms'))} (end of speech to first audio)  "
        f"stt={_sec(_delta(marks, 'record_done', 'stt_done'))}  "
        f"first_token={_sec(turn.get('ttft_ms'))}  llm={_sec(turn.get('llm_ms'))}  "
        f"total={_sec(turn.get('total_ms'))}"
    )
    if turn.get("mic"):
        m = turn["mic"]
        print(f"  mic:        {m.get('seconds')}s recorded, ended by {m.get('ended')}, "
              f"max level {m.get('max_level')} (threshold {m.get('threshold')}, noise floor {m.get('noise_floor')})")
    if turn.get("wake_score") is not None:
        print(f"  wake score: {turn['wake_score']}")
    if turn.get("phases"):
        print("  phases:     " + " → ".join(f"{p[1]}@{_sec(p[0])}" for p in turn["phases"]))
    for key in ("stt_backend", "stt_npu_error", "llm_error", "action_parse_errors", "history_reset", "audio_file"):
        if turn.get(key):
            print(f"  {key + ':':<11} {turn[key]}")


def _delta(marks, a, b):
    return marks[b] - marks[a] if a in marks and b in marks else None


def print_turns_table(turns: list[dict]):
    print(f"{'started':<19} {'src':<5} {'status':<10} {'resp':>6} {'llm':>6} {'total':>6}  heard → said")
    for t in turns:
        heard = (t.get("transcript") or "")[:38]
        said = _reply_text(t)[:50] or (str(t.get("actions", ""))[:50] if t.get("actions") else "")
        print(f"{t.get('started', ''):<19} {t.get('source', ''):<5} {t.get('status', ''):<10} "
              f"{_sec(t.get('response_ms')):>6} {_sec(t.get('llm_ms')):>6} {_sec(t.get('total_ms')):>6}  {heard!r} → {said!r}")


def stats(turns: list[dict]) -> dict:
    """Median and worst-case timings, plus counts of what went wrong."""
    def summary(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return {"n": len(values), "median_ms": int(statistics.median(values)), "max_ms": max(values)}

    by_status = {}
    for t in turns:
        by_status[t.get("status", "?")] = by_status.get(t.get("status", "?"), 0) + 1
    return {
        "turns": len(turns),
        "by_status": by_status,
        "by_source": {s: sum(1 for t in turns if t.get("source") == s) for s in ("voice", "text", "wav")},
        "response": summary([t.get("response_ms") for t in turns]),
        "first_token": summary([t.get("ttft_ms") for t in turns]),
        "llm": summary([t.get("llm_ms") for t in turns]),
        "stt": summary([_delta(t.get("marks", {}), "record_done", "stt_done") for t in turns]),
        "search": summary([t["search"].get("ms") for t in turns if t.get("search")]),
        "stt_on_cpu": sum(1 for t in turns if t.get("stt_backend") == "cpu"),
        "llm_errors": sum(1 for t in turns if t.get("llm_error")),
        "tts_errors": sum(1 for t in turns for s in t.get("spoken", []) if s.get("error")),
        "history_resets": sum(1 for t in turns if t.get("history_reset")),
        "action_parse_errors": sum(len(t.get("action_parse_errors", [])) for t in turns),
    }


def sample() -> dict:
    """One monitoring sample: host readings always, agent and microphone
    readings when the agent answers."""
    reply = request({"cmd": "status"}, timeout=5)
    record = {"t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if reply.get("ok"):
        status = reply["status"]
        last = status.get("last_turn") or {}
        record["system"] = status.get("system", {})
        record["mic"] = status.get("mic", {})
        record["agent"] = {key: status.get(key) for key in (
            "phase", "expression", "muted", "showing_image", "llm_ready", "stt_on_npu",
            "agent_uptime_s", "history_messages")}
        record["agent"]["timers"] = len(status.get("timers") or [])
        record["last_turn"] = {key: last.get(key) for key in (
            "id", "source", "status", "transcript", "response_ms", "wake_score")} if last else None
        if last.get("spoken"):
            record["last_turn"]["said"] = _reply_text(last)[:120]
    else:
        from core.health import system_health        # the agent is down: still report the host
        record["system"] = system_health()
        record["agent"] = None
        record["agent_error"] = reply.get("error")
    return record


def screenshot(path: str) -> dict:
    """Capture the Wayland display with grim."""
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    env.setdefault("WAYLAND_DISPLAY", "wayland-0")
    try:
        subprocess.run(["grim", path], check=True, env=env, capture_output=True, timeout=15)
    except FileNotFoundError:
        return {"ok": False, "error": "grim is not installed (sudo apt install grim)"}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": e.stderr.decode(errors="replace").strip() or repr(e)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "grim timed out"}
    return {"ok": True, "file": os.path.abspath(path)}


# ── Commands ─────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Observe and drive the running agent.")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for name, arg in (("say", "text"), ("hear", "wav")):
        sp = sub.add_parser(name)
        sp.add_argument(arg)
        sp.add_argument("--quiet", action="store_true", help="run the turn without sound")
    for name, default in (("turns", 10), ("stats", 50), ("events", 20)):
        sub.add_parser(name).add_argument("-n", type=int, default=default)
    sub.add_parser("show").add_argument("turn_id", nargs="?", default="last")
    sub.add_parser("screenshot").add_argument("file", nargs="?", default=os.path.join(trace.LOG_DIR, "screenshot.png"))
    sub.add_parser("watch").add_argument("-i", type=float, default=2.0, help="seconds between samples")
    args = p.parse_args(argv)

    def out(obj, pretty=None):
        if args.json or pretty is None:
            print(json.dumps(obj, indent=2, ensure_ascii=False))
        else:
            pretty()

    if args.cmd == "watch":
        try:
            while True:
                print(json.dumps(sample(), ensure_ascii=False), flush=True)
                time.sleep(max(args.i, 0.5))
        except (KeyboardInterrupt, BrokenPipeError):
            return 0

    if args.cmd == "status":
        reply = request({"cmd": "status"}, timeout=10)
        if not reply.get("ok"):
            print(reply.get("error"), file=sys.stderr)
            return 1
        out(reply["status"])
        return 0

    if args.cmd in ("say", "hear"):
        payload = {"cmd": args.cmd, "quiet": args.quiet}
        payload["text" if args.cmd == "say" else "wav"] = args.text if args.cmd == "say" else os.path.abspath(args.wav)
        reply = request(payload)
        if not reply.get("ok"):
            print(reply.get("error"), file=sys.stderr)
            return 1
        out(reply["turn"], lambda: print_turn(reply["turn"]))
        return 0 if reply["turn"].get("status") == "ok" else 2

    if args.cmd == "turns":
        turns = trace.read("turns.jsonl", last=args.n)
        out(turns, lambda: print_turns_table(turns))
        return 0

    if args.cmd == "show":
        turns = trace.read("turns.jsonl")
        match = turns[-1:] if args.turn_id == "last" else [t for t in turns if t.get("id") == args.turn_id]
        if not match:
            print("no such turn", file=sys.stderr)
            return 1
        out(match[0], lambda: print_turn(match[0]))
        return 0

    if args.cmd == "stats":
        out(stats(trace.read("turns.jsonl", last=args.n)))
        return 0

    if args.cmd == "events":
        events = trace.read("events.jsonl", last=args.n)
        out(events, lambda: [print(json.dumps(e, ensure_ascii=False)) for e in events])
        return 0

    if args.cmd == "screenshot":
        os.makedirs(os.path.dirname(os.path.abspath(args.file)), exist_ok=True)
        reply = screenshot(args.file)
        if not reply.get("ok"):
            print(reply.get("error"), file=sys.stderr)
            return 1
        out(reply, lambda: print(reply["file"]))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
