"""Tests for timer and reminder recognition (core/timers.py, core/actions.py).

Run with ``pytest tests/test_timers.py`` or ``python tests/test_timers.py``.
"""

import os
import sys
import tempfile

os.environ["TRACE_DIR"] = tempfile.mkdtemp(prefix="bmo-test-trace-")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.actions import detect_pre_llm_action  # noqa: E402
from core.config import CHARACTER  # noqa: E402
from core.timers import format_clock, format_duration, parse_timer_request, second_person  # noqa: E402

# utterance → seconds
TIMERS = {
    "Set a timer for 90 seconds.": 90,
    "set a timer for five minutes": 300,
    "Set a 5-minute timer": 300,
    "start a ten minute timer please": 600,
    "Can you set a timer for 1 minute and 30 seconds?": 90,
    "timer for 2.5 minutes": 150,
    "Set a timer for an hour": 3600,
    "set a timer for an hour and a half": 5400,
    "set a timer for one and a half minutes": 90,
    "set a timer for half an hour": 1800,
    "Set a timer for twenty five minutes.": 1500,
    "set a timer for forty-five seconds": 45,
    "put a timer on for 3 mins": 180,
    "Set a timer for a couple of minutes": 120,
    "wake me up in 20 minutes": 1200,
    "countdown for 10 seconds": 10,
}

# utterance → (seconds, what to be reminded of)
REMINDERS = {
    "Remind me in ten minutes to call mom.": (600, "call mom"),
    "remind me to take out the trash in 5 minutes": (300, "take out the trash"),
    "Remind me in 1 hour and 15 minutes that the laundry is done": (4500, "the laundry is done"),
    "remind me about the pizza in 12 minutes": (720, "the pizza"),
    "Remind me in 30 seconds": (30, None),
}

# Must not start a timer
NOT_TIMERS = [
    "what time is it",
    "how does a timer work",
    "cancel the timer",
    "how long is left on the timer",
    "tell me a story about five minutes of fame",
    "remind me to call mom tomorrow",          # no duration: left to the LLM
    "I waited ten minutes for the bus",
    "set a timer",                              # no duration
    "play some music",
    "set a timer for zero seconds",
]


def test_timer_requests_are_recognised():
    for text, seconds in TIMERS.items():
        parsed = parse_timer_request(text)
        assert parsed is not None, text
        assert abs(parsed[0] - seconds) < 0.01, (text, parsed)
        assert parsed[1] is None, (text, parsed)


def test_reminders_carry_what_to_remind():
    for text, expected in REMINDERS.items():
        assert parse_timer_request(text) == expected, (text, parse_timer_request(text))


def test_other_sentences_do_not_start_timers():
    for text in NOT_TIMERS:
        assert parse_timer_request(text) is None, (text, parse_timer_request(text))


def test_routing_builds_the_timer_action():
    action = detect_pre_llm_action("Set a timer for 90 seconds.")
    assert action == {"action": "set_timer", "minutes": 1.5}
    action = detect_pre_llm_action("remind me in ten minutes to call my mom")
    assert action["action"] == "set_timer" and action["minutes"] == 10
    assert "call your mom" in action["message"]
    assert "${" not in action["message"]


def test_a_reminder_wins_over_other_keywords():
    # "take a photo" alone triggers the camera; inside a reminder it must not
    action = detect_pre_llm_action("remind me in 5 minutes to take a photo")
    assert action["action"] == "set_timer"


def test_durations_are_spoken_naturally():
    assert format_duration(90) == "1 minute and 30 seconds"
    assert format_duration(300) == "5 minutes"
    assert format_duration(3600) == "1 hour"
    assert format_duration(3725) == "1 hour, 2 minutes and 5 seconds"
    assert format_duration(1) == "1 second"
    assert format_duration(90, "de") == "1 Minute und 30 Sekunden"


def test_clock_face_text():
    assert format_clock(90) == "1:30"
    assert format_clock(59.2) == "1:00"        # rounds up
    assert format_clock(9) == "0:09"
    assert format_clock(0.3) == "0:01"
    assert format_clock(0) == "0:00"
    assert format_clock(-2) == "0:00"
    assert format_clock(3725) == "1:02:05"


def test_second_person():
    assert second_person("call my mom") == "call your mom"
    assert second_person("I need to leave") == "you need to leave"
    assert second_person("the mystery box") == "the mystery box"


def test_confirmation_lines_exist_in_every_language():
    for language in ("en", "de"):
        line = CHARACTER.say("timer_set", language, duration=format_duration(90, language))
        assert "${" not in line and any(ch.isdigit() for ch in line), line
        assert "${" not in CHARACTER.say("reminder", language, message="x")


def test_german_timers_and_reminders():
    from core.timers import parse_timer_request as parse
    assert parse("Stell einen Timer auf 5 Minuten.") == (300, None)
    assert parse("Stell einen Timer auf fünf Minuten") == (300, None)
    assert parse("Timer für fünfundzwanzig Sekunden") == (25, None)
    assert parse("Stell den Timer auf eine halbe Stunde") == (1800, None)
    assert parse("Timer auf anderthalb Stunden") == (5400, None)
    assert parse("Timer auf eine Viertelstunde") == (900, None)
    assert parse("Stell einen Timer auf 2,5 Minuten") == (150, None)
    assert parse("Weck mich in zwei Stunden") == (7200, None)
    assert parse("Erinnere mich in zehn Minuten an den Tee.") == (600, "den tee")
    assert parse("Erinnere mich in 10 Minuten daran, den Müll rauszubringen") == (600, "den müll rauszubringen")
    # What speech-to-text made of "Erinnere mich" in the measurement
    assert parse("der inneren mich in 10 Minuten an den Tee.") == (600, "den tee")
    assert parse("Dainnere mich in zehn Minuten an den Tee.") == (600, "den tee")
    # Not a request to start one
    assert parse("Stopp den Timer") is None
    assert parse("Wie lange läuft der Timer noch?") is None
    assert parse("Ich erinnere mich an den Urlaub") is None          # no duration
    assert parse("Was sind fünf Minuten in Sekunden?") is None       # no timer word


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
