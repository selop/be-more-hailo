#!/usr/bin/env python3
"""Run the live face on a development machine, fed with made-up events.

    python scripts/face_demo.py            # serves the face and opens your browser

It walks through the states a real turn goes through (idle, listening with a
moving mic meter, thinking, speaking with a mouth that follows an invented
voice, a timer, each emotion) so the page can be judged and tuned without
the Pi. Edit face/index.html and reload the browser to see changes.
"""

import math
import os
import random
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.display_server import DisplayServer  # noqa: E402


def fake_envelope(seconds: float, step_ms: int = 20) -> list[float]:
    """A speech-like loudness curve: syllables, word gaps, a pause or two."""
    values, level, target, until = [], 0.0, 0.0, 0
    for i in range(int(seconds * 1000 / step_ms)):
        if i >= until:
            target = 0.0 if random.random() < 0.25 else random.uniform(0.35, 1.0)
            until = i + random.randint(3, 10)
        level += (target - level) * (0.6 if target > level else 0.3)
        values.append(round(level, 3))
    return values


def main():
    display = DisplayServer(port=int(os.environ.get("FACE_PORT", "8765")))
    if not display.start():
        sys.exit(1)
    url = f"http://127.0.0.1:{display.port}/"
    print(f"Live face at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)

    def state(phase, expression=None, status="", muted=False):
        display.publish("state", phase=phase, expression=expression or phase, muted=muted, status=status)

    def wait(seconds, tick=None):
        end = time.time() + seconds
        while time.time() < end:
            if tick:
                tick()
            time.sleep(0.05)

    try:
        while True:
            state("idle", status="Ready")
            display.publish("timers", timers=[])
            wait(5)

            state("listening", status="Listening...")
            wait(3, lambda: display.publish("meter", level=round(abs(math.sin(time.time() * 5)) * random.uniform(0.3, 1.0), 3)))

            state("thinking", status="Thinking...")
            display.publish("bubble", text="set a timer for twenty seconds and tell me a joke", hold_ms=8000)
            wait(2.5)

            for sentence_seconds in (2.4, 3.2):
                state("speaking", status="Speaking...")
                display.publish("speak", envelope=fake_envelope(sentence_seconds), step_ms=20)
                wait(sentence_seconds)
                state("idle", status="Ready...")
                wait(0.3)

            started = time.time()
            for expression in ("happy", "sad", "angry", "surprised", "heart", "starry_eyed", "sleepy",
                               "dizzy", "cheeky", "confused"):
                state("idle", expression=expression, status=f"Feeling {expression}...")
                wait(3.5, lambda: display.publish("timers", timers=[
                    {"ends_in_s": max(round(20 - (time.time() - started)), 0), "total_s": 20.0}
                ] if time.time() - started < 20 else []))

            state("screensaver", status="Screensaver...")
            wait(12)
    except KeyboardInterrupt:
        display.stop()


if __name__ == "__main__":
    main()
