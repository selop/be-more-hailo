"""Recognise timer and reminder requests without the LLM.

Pure functions, no hardware and no character text. The small on-device
model does not reliably emit the timer action, so requests such as
"set a timer for 90 seconds" or "remind me in ten minutes to call mom"
are parsed here before the LLM is asked.
"""

import re

_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}

_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9,
}
_NUMBER_WORDS = {
    "a": 1, "an": 1, **_ONES,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "a couple of": 2, "couple of": 2, "a few": 3,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_UNIT = r"(?P<unit>" + "|".join(sorted(_UNIT_SECONDS, key=len, reverse=True)) + r")\b"
_TENS_RE = "|".join(_TENS)
_ONES_RE = "|".join(_ONES)
_WORDS_RE = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_NUMBER = (
    r"(?P<num>\d+(?:\.\d+)?"
    rf"|(?:{_TENS_RE})(?:[ -](?:{_ONES_RE}))?"
    rf"|{_WORDS_RE})"
)

# "five minutes", "1.5 hours", "one and a half minutes", "an hour and a half"
_AMOUNT_RE = re.compile(
    rf"\b{_NUMBER}(?P<half_before> and a half)?[ -]{_UNIT}(?P<half_after> and a half)?"
)
# "half an hour", "half a minute"
_HALF_RE = re.compile(rf"\bhalf (?:an?|of an?) {_UNIT}")

_TIMER_RE = re.compile(r"\b(?:timer|countdown)\b")
_REMIND_RE = re.compile(r"\bremind me\b")
_WAKE_RE = re.compile(r"\bwake me\b")
# Requests about a timer that are not "start one"
_NOT_A_START_RE = re.compile(r"\b(?:cancel|stop|delete|remove|clear|how long|how much|what is|what's)\b")

MAX_SECONDS = 24 * 3600


def _number_value(text: str) -> float:
    text = text.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    parts = re.split(r"[ -]", text)
    return float(_TENS[parts[0]] + (_ONES[parts[1]] if len(parts) > 1 else 0))


def find_durations(text: str) -> tuple[float, list[tuple[int, int]]]:
    """Return the total seconds of every duration mentioned in *text*
    (lower-case), and the character spans where they were found."""
    total = 0.0
    spans = []
    for m in _HALF_RE.finditer(text):
        total += 0.5 * _UNIT_SECONDS[m.group("unit")]
        spans.append(m.span())
    for m in _AMOUNT_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue    # "an hour" inside "half an hour"
        amount = _number_value(m.group("num"))
        if m.group("half_before") or m.group("half_after"):
            amount += 0.5
        total += amount * _UNIT_SECONDS[m.group("unit")]
        spans.append(m.span())
    return total, sorted(spans)


def parse_timer_request(user_text: str) -> tuple[float, str | None] | None:
    """Recognise a request to start a timer or a timed reminder.

    Returns ``(seconds, reminder)`` where *reminder* is what to be reminded
    of ("call mom") or ``None`` for a plain timer. Returns ``None`` when the
    text is not such a request, including when it names no duration.
    """
    text = user_text.lower().replace("’", "'")
    text = re.sub(r"[.!?,]+(?=\s|$)", " ", text)       # sentence punctuation, not decimals
    text = re.sub(r"\s+", " ", text).strip()

    is_timer = bool(_TIMER_RE.search(text))
    is_reminder = bool(_REMIND_RE.search(text))
    is_wake = bool(_WAKE_RE.search(text))
    if not (is_timer or is_reminder or is_wake):
        return None
    if _NOT_A_START_RE.search(text):
        return None

    seconds, spans = find_durations(text)
    if seconds <= 0 or seconds > MAX_SECONDS:
        return None

    reminder = None
    if is_reminder:
        # Remove the "in ten minutes" parts, then take what follows "remind me to"
        rest = text
        for start, end in reversed(spans):
            head = re.sub(r"\b(?:in|after|and)\s*$", "", rest[:start])
            rest = head + " " + rest[end:]
        rest = re.sub(r"\s+", " ", rest).strip()
        m = re.search(r"\bremind me (?:to|that|about|of)\s+(.+)$", rest)
        if m:
            reminder = m.group(1).strip(" .") or None
    return seconds, reminder


# ── Speaking a duration ──────────────────────────────────────────────────

_DURATION_WORDS = {
    "en": {"hour": ("hour", "hours"), "minute": ("minute", "minutes"),
           "second": ("second", "seconds"), "and": "and"},
    "de": {"hour": ("Stunde", "Stunden"), "minute": ("Minute", "Minuten"),
           "second": ("Sekunde", "Sekunden"), "and": "und"},
}


def format_duration(seconds: float, language: str = "en") -> str:
    """Spell a duration for speech: ``90`` → "1 minute and 30 seconds"."""
    words = _DURATION_WORDS.get(language, _DURATION_WORDS["en"])
    total = int(round(seconds))
    if total <= 0:
        total = 1
    parts = []
    for key, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        count, total = divmod(total, size)
        if count:
            parts.append(f"{count} {words[key][0 if count == 1 else 1]}")
    if len(parts) > 1:
        return ", ".join(parts[:-1]) + f" {words['and']} " + parts[-1]
    return parts[0]


def format_clock(seconds: float) -> str:
    """Remaining time for a clock face: ``83`` → "1:23", ``3725`` → "1:02:05".
    Rounds up, so the display reaches 0:00 only when the timer is over."""
    total = max(int(-(-seconds // 1)), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def second_person(text: str) -> str:
    """Turn the user's wording into what is said back to them:
    "call my mom" → "call your mom"."""
    swaps = {"my": "your", "i": "you", "i'm": "you're", "me": "you", "myself": "yourself", "mine": "yours"}
    return re.sub(
        r"\b(?:i'm|myself|mine|my|me|i)\b",
        lambda m: swaps[m.group(0).lower()],
        text,
        flags=re.IGNORECASE,
    )
