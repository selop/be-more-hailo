"""Character pack loader.

A character pack is a folder ``characters/<id>/`` holding ``character.toml``
(name, prompts, spoken lines, expressions, theme, voice settings) and the
assets it points to. The engine reads everything character-specific through
this module, so swapping the character means selecting another pack.

Only the standard library is used here so the module stays importable in
tests without any hardware or audio dependencies.
"""

import os
import random
import re
from string import Template

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARACTERS_DIR = os.path.join(PROJECT_ROOT, "characters")
DEFAULT_CHARACTER = "bmo"

REQUIRED_PROMPTS = ("system", "vision", "search_summary", "live_data", "idle_system", "idle_thought")
REQUIRED_LINES = (
    "empty_reply", "confused_reset", "think_error", "stream_error",
    "vision_missing", "vision_timeout", "vision_error",
    "timer_set", "timer_done", "reminder",
    "music_intros", "no_music", "camera_intros", "no_camera", "camera_error",
    "draw_image", 
)
REQUIRED_THEME = (
    "face_bg", "status_fg", "alert_fg", "bubble_fill", "bubble_outline",
    "bubble_text", "frame_inner", "frame_outer",
)


class CharacterError(Exception):
    """Raised when a character pack is missing or incomplete."""


class Character:
    """Read-only view of one character pack."""

    def __init__(self, pack_id: str, data: dict):
        self.id = pack_id
        self._data = data
        self._validate()

        identity = data["identity"]
        self.name: str = identity["name"]
        self.heard_as: list[str] = list(identity.get("heard_as", []))

        face = data.get("face", {})
        self.expressions: set[str] = set(face.get("expressions", []))
        self.personas: set[str] = set(face.get("personas", []))
        self.idle_expressions: list[str] = list(face.get("idle_expressions", []))
        self.screensaver_faces: list[str] = list(face.get("screensaver", []))

        idle = data.get("idle", {})
        self.search_topics: list[str] = list(idle.get("search_topics", []))
        self.fallback_phrases: list[str] = list(idle.get("fallback_phrases", []))

        self.theme: dict = dict(data["theme"])
        self.wake_threshold: float | None = data.get("hearing", {}).get("wake_threshold")

        if self.heard_as:
            alternatives = "|".join(re.escape(s) for s in self.heard_as)
            self._heard_as_re = re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)
        else:
            self._heard_as_re = None

    # ── Validation ───────────────────────────────────────────────────────

    def _validate(self):
        d = self._data
        missing = []
        if not d.get("identity", {}).get("name"):
            missing.append("identity.name")
        for key in REQUIRED_PROMPTS:
            if not d.get("prompts", {}).get(key):
                missing.append(f"prompts.{key}")
        for key in REQUIRED_LINES:
            if not d.get("lines", {}).get("en", {}).get(key):
                missing.append(f"lines.en.{key}")
        for key in REQUIRED_THEME:
            if not d.get("theme", {}).get(key):
                missing.append(f"theme.{key}")
        if not d.get("voice", {}).get("en", {}).get("model"):
            missing.append("voice.en.model")
        if missing:
            raise CharacterError(
                f"Character pack '{self.id}' is missing: {', '.join(missing)}"
            )

    # ── Text ─────────────────────────────────────────────────────────────

    def prompt(self, key: str, **values) -> str:
        """Return prompt *key* with its ``$placeholders`` filled in."""
        return Template(self._data["prompts"][key]).substitute(**values)

    def line(self, key: str, language: str = "en"):
        """Return the spoken line (a string, or a list of alternatives) for
        *key* in *language*, falling back to English, then to the key."""
        lines = self._data.get("lines", {})
        value = lines.get(language, {}).get(key)
        if value is None:
            value = lines.get("en", {}).get(key, key)
        return value

    def say(self, key: str, language: str = "en", **values) -> str:
        """Like :meth:`line`, but always one string: a random pick if the
        line has alternatives. Keyword arguments fill ``${placeholders}``
        in the line."""
        value = self.line(key, language)
        text = random.choice(value) if isinstance(value, list) else value
        return Template(text).safe_substitute(**values) if values else text

    def normalize_name(self, text: str) -> str:
        """Replace misheard or misspelled forms of the name with the name."""
        if self._heard_as_re is None:
            return text
        return self._heard_as_re.sub(self.name, text)

    # ── Voice ────────────────────────────────────────────────────────────

    def voice(self, language: str = "en") -> dict:
        """Piper settings for *language* (``model``, ``length_scale``),
        falling back to English."""
        voice = self._data["voice"]
        settings = voice.get(language) or voice["en"]
        return {"model": settings["model"], "length_scale": settings.get("length_scale", 1.0)}

    @property
    def voice_filter(self) -> dict:
        voice = self._data["voice"]
        return {
            "enabled": voice.get("lpf_enabled", False),
            "cutoff": int(voice.get("lpf_cutoff", 4000)),
            "order": int(voice.get("lpf_order", 4)),
        }

    @property
    def default_pronunciations(self) -> dict:
        return dict(self._data["voice"].get("pronunciations", {}))

    # ── Assets ───────────────────────────────────────────────────────────

    def asset(self, key: str) -> str:
        """Absolute path of asset *key* (``sounds``, ``wakeword``,
        ``pronunciations``, ``voice_eq``).

        Paths in the pack are relative to the project root. When a key is
        not listed, the asset is expected inside the pack folder.
        """
        defaults = {
            "sounds": "sounds",
            "wakeword": "wakeword.onnx",
            "pronunciations": "pronunciations.json",
            "voice_eq": "voice_eq.json",
        }
        configured = self._data.get("assets", {}).get(key)
        if configured is not None:
            return os.path.join(PROJECT_ROOT, configured)
        return os.path.join(CHARACTERS_DIR, self.id, defaults[key])


def load_character(pack_id: str) -> Character:
    """Load ``characters/<pack_id>/character.toml``."""
    path = os.path.join(CHARACTERS_DIR, pack_id, "character.toml")
    if not os.path.exists(path):
        available = sorted(
            d for d in os.listdir(CHARACTERS_DIR)
            if os.path.exists(os.path.join(CHARACTERS_DIR, d, "character.toml"))
        ) if os.path.isdir(CHARACTERS_DIR) else []
        raise CharacterError(
            f"No character pack '{pack_id}' at {path}. Available: {', '.join(available) or 'none'}"
        )
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Character(pack_id, data)


_active: Character | None = None


def get_character() -> Character:
    """Return the active character, loading it on first use from the
    ``CHARACTER`` environment variable (default ``bmo``)."""
    global _active
    if _active is None:
        _active = load_character(os.environ.get("CHARACTER", DEFAULT_CHARACTER))
    return _active
