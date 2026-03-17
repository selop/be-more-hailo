"""Pure-function action detection and response cleaning.

No hardware dependencies — all functions operate on strings and return
strings, dicts, or None.  Trivially testable without an NPU.
"""

import json
import re
import urllib.parse


# ── Keyword lists (single source of truth) ────────────────────────────────

CAMERA_KEYWORDS = [
    "take a photo", "take a picture", "take photo", "take picture",
    "look at", "what do you see", "what can you see", "use your camera",
    "photograph", "snap a photo",
]

DISPLAY_IMAGE_KEYWORDS = [
    "show me a picture", "show me an image", "show me a photo",
    "show a picture", "show an image", "show a photo",
    "display a picture", "display an image", "display a photo",
    "picture of", "image of", "photo of",
    "generate an image", "generate a picture",
    "draw me", "draw a",
]

MUSIC_KEYWORDS = [
    "play music", "play a song", "play me a song", "play some music",
    "sing a song", "sing me a song", "sing for me", "sing something",
    "play a tune", "play me a tune", "play some tunes",
    "can you sing", "will you sing", "do you sing",
    "play your music", "jam out", "dance for me",
    "sing for bmo", "bmo sing", "play me some music",
]

VOICE_EQ_KEYWORDS = [
    "tune your voice", "voice settings", "voice tuner",
    "adjust your voice", "change your voice", "voice equalizer",
    "eq settings", "tune voice", "voice eq",
]

REALTIME_KEYWORDS = [
    "weather", "forecast", "temperature", "tonight", "tomorrow",
    "news", "latest", "right now", "score", "stocks", "bitcoin",
    "crypto", "price of", "happening", "recently", "live",
]

QUESTION_MARKERS = [
    "what", "who", "when", "where", "find", "search", "tell me",
    "look up", "check", "is there", "did", "?",
]

PROMPT_LEAK_PATTERNS = [
    re.compile(r'CRIT(?:IC)?AL:.*?JSON', re.IGNORECASE | re.DOTALL),
    re.compile(r'you MUST output.*?JSON', re.IGNORECASE | re.DOTALL),
    re.compile(r'output (?:exactly )?this JSON', re.IGNORECASE),
    re.compile(r'If the user asks.*?(?:output|JSON)', re.IGNORECASE | re.DOTALL),
    re.compile(r'Replace YOUR_PROMPT_HERE', re.IGNORECASE),
    re.compile(r'Valid emotions are:', re.IGNORECASE),
]

# Prefixes stripped when extracting the subject from display-image requests
_DISPLAY_IMAGE_PREFIXES = [
    "show me a picture of", "show me an image of", "show me a photo of",
    "show a picture of", "show an image of", "show a photo of",
    "display a picture of", "display an image of", "display a photo of",
    "generate an image of", "generate a picture of",
    "draw me a", "draw me an", "draw me", "draw a", "draw an",
    "picture of", "image of", "photo of",
]


# ── Pre-LLM action detection ─────────────────────────────────────────────

def detect_pre_llm_action(user_text: str) -> dict | None:
    """Check *user_text* against keyword lists and return a pre-built action
    dict, or ``None`` if no shortcut matches.

    Returned dict always has an ``"action"`` key (``"take_photo"``,
    ``"display_image"``, or ``"play_music"``).
    """
    lower = user_text.lower()

    if any(kw in lower for kw in CAMERA_KEYWORDS):
        return {"action": "take_photo"}

    if any(kw in lower for kw in DISPLAY_IMAGE_KEYWORDS):
        return json.loads(build_display_image_action(user_text))

    if any(kw in lower for kw in MUSIC_KEYWORDS):
        return {"action": "play_music"}

    if any(kw in lower for kw in VOICE_EQ_KEYWORDS):
        return {"action": "voice_eq"}

    return None


def detect_pre_llm_action_json(user_text: str) -> str | None:
    """Like :func:`detect_pre_llm_action` but returns the JSON string
    (or ``None``), matching the old API surface used by ``Brain``."""
    action = detect_pre_llm_action(user_text)
    if action is None:
        return None
    return json.dumps(action)


# ── Web-search heuristic ─────────────────────────────────────────────────

def needs_web_search(user_text: str) -> bool:
    """Return True when *user_text* looks like a real-time / factual question."""
    lower = user_text.lower()
    has_realtime = any(kw in lower for kw in REALTIME_KEYWORDS)
    has_question = any(q in lower for q in QUESTION_MARKERS)
    return has_realtime and has_question


def inject_search_context(user_text: str, search_result: str) -> str:
    """Build the modified user message with ``[LIVE DATA: ...]`` context."""
    clean_result = re.sub(r"^SEARCH RESULTS for '.*?':\n?", "", search_result).strip()
    return (
        f"[LIVE DATA: {clean_result}] "
        f"Using only the above live data, answer in one or two sentences as BMO: {user_text}"
    )


# ── Display-image / VLM helpers ──────────────────────────────────────────

def build_display_image_action(user_text: str) -> str:
    """Return a ``display_image`` JSON string with a Pollinations URL."""
    subject = user_text
    for prefix in _DISPLAY_IMAGE_PREFIXES:
        if subject.lower().startswith(prefix):
            subject = subject[len(prefix):].strip()
            break
    subject = subject.rstrip("?.!")
    prompt = urllib.parse.quote(subject.strip() or "something fun")
    return json.dumps({
        "action": "display_image",
        "image_url": f"https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true",
    })


def vlm_question(user_text: str) -> str:
    """Convert camera-trigger phrases into an actual VLM question."""
    if not user_text:
        return "What do you see in this image? Describe it briefly."
    lower = user_text.lower()
    if any(lower.strip().rstrip(".!?") == p for p in CAMERA_KEYWORDS) or \
       any(lower.startswith(p) for p in ["take a photo", "take a picture", "take photo",
                                          "take picture", "snap a photo", "photograph"]):
        return "What do you see in this image? Describe it briefly."
    return user_text


# ── Response cleaning ────────────────────────────────────────────────────

def clean_llm_response(content: str) -> str:
    """Consolidate all post-LLM cleaning: stop tokens, smart quotes,
    prompt leakage, BMO spelling, and empty-response fallback."""
    # Strip stop tokens
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        content = content.replace(tok, "")
    # Normalise smart quotes
    content = content.replace('\u201c', '"').replace('\u201d', '"')
    content = content.replace('\u2018', "'").replace('\u2019', "'")
    content = content.strip()
    # Strip prompt leakage
    content = strip_prompt_leakage(content)
    # Fix BMO spelling
    content = re.sub(r'\bBeemo\b', 'BMO', content, flags=re.IGNORECASE)
    # Fallback
    if not content.strip():
        content = "BMO is here! How can I help?"
    return content


def strip_prompt_leakage(content: str) -> str:
    """Remove text that looks like the model echoing system prompt instructions."""
    for pat in PROMPT_LEAK_PATTERNS:
        match = pat.search(content)
        if match:
            content = content[:match.start()].strip()
    return content


# ── JSON action extraction ───────────────────────────────────────────────

def extract_json_action(content: str) -> dict | None:
    """Parse the first ``{...}`` block with an ``"action"`` key, or ``None``."""
    clean = content.replace('\u201c', '"').replace('\u201d', '"')
    clean = clean.replace('\u2018', "'").replace('\u2019', "'")
    match = re.search(r'\{.*?\}', clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass
    return None


# ── Pronunciation tag helpers ────────────────────────────────────────────

def extract_pronunciation(content: str) -> tuple[str, str] | None:
    """Extract ``!PRONOUNCE: word=phonetic`` from *content*, or ``None``."""
    m = re.search(r'!PRONOUNCE:\s*([a-zA-Z0-9_-]+)\s*=\s*([a-zA-Z0-9_-]+)', content, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def strip_pronunciation_tag(content: str) -> str:
    """Remove the ``!PRONOUNCE:`` tag from *content*."""
    return re.sub(r'!PRONOUNCE:.*', '', content, flags=re.IGNORECASE).strip()
