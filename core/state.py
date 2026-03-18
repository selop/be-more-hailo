"""Centralized state store for BMO.

Separates *phase* (what BMO is doing) from *expression* (what face to
display).  Provides thread-safe phase transitions with hook callbacks.
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field


class Phase(Enum):
    WARMUP = "warmup"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CAPTURING = "capturing"
    SCREENSAVER = "screensaver"
    JAMMING = "jamming"
    SHHH = "shhh"
    DISPLAY_IMAGE = "display_image"
    ERROR = "error"


# Pure emotion expressions — phase stays unchanged when these are set
VALID_EXPRESSIONS = {
    "happy", "sad", "angry", "surprised", "sleepy",
    "dizzy", "cheeky", "heart", "starry_eyed", "confused",
}

# Persona states are expression-only (phase stays SCREENSAVER)
PERSONA_EXPRESSIONS = {"football", "detective", "sir_mano", "low_battery", "bee"}

# Every string value that has a faces/ directory — used for animation loading
ALL_EXPRESSION_NAMES = (
    {p.value for p in Phase}
    | VALID_EXPRESSIONS
    | PERSONA_EXPRESSIONS
)


@dataclass
class BotState:
    phase: Phase = Phase.WARMUP
    expression: str = "warmup"          # face to display (orthogonal to phase)
    muted: bool = False
    display_image: object = None        # PhotoImage ref (prevents GC)
    last_spoke_at: float = 0.0
    last_phase_change: float = field(default_factory=time.time)


class StateStore:
    """Single source of truth for BMO's runtime state."""

    def __init__(self):
        self._state = BotState()
        self._lock = threading.Lock()
        self._on_phase_change: list = []

    # --- Read access (atomic field reads — no lock needed) ---

    @property
    def phase(self) -> Phase:
        return self._state.phase

    @property
    def expression(self) -> str:
        return self._state.expression

    @property
    def muted(self) -> bool:
        return self._state.muted

    @property
    def display_image(self):
        return self._state.display_image

    @property
    def last_spoke_at(self) -> float:
        return self._state.last_spoke_at

    @property
    def last_phase_change(self) -> float:
        return self._state.last_phase_change

    # --- Write access ---

    def set_phase(self, phase: Phase, expression: str | None = None, msg: str = ""):
        """Change phase and optionally expression.  Fires on_phase_change hooks."""
        with self._lock:
            old_phase = self._state.phase
            changed = old_phase != phase
            self._state.phase = phase
            if expression is not None:
                self._state.expression = expression
            elif changed:
                # Default: expression follows phase name
                self._state.expression = phase.value
            if changed:
                self._state.last_phase_change = time.time()
        if changed:
            for hook in self._on_phase_change:
                hook(old_phase, phase, msg)

    def set_expression(self, expr: str):
        """Change face expression WITHOUT changing phase."""
        self._state.expression = expr

    def set_muted(self, muted: bool):
        self._state.muted = muted

    def set_display_image(self, img):
        self._state.display_image = img

    def clear_display_image(self):
        self._state.display_image = None

    def mark_spoke(self):
        self._state.last_spoke_at = time.time()

    def on_phase_change(self, callback):
        """Register hook: ``callback(old_phase, new_phase, msg)``."""
        self._on_phase_change.append(callback)


class BotStates:
    """Backward-compat aliases — prefer ``Phase`` enum + expression strings.

    Kept so that existing code (animation loading, screensaver state lists)
    can continue referencing ``BotStates.IDLE`` etc. as plain strings.
    """
    IDLE = Phase.IDLE.value
    LISTENING = Phase.LISTENING.value
    THINKING = Phase.THINKING.value
    SPEAKING = Phase.SPEAKING.value
    ERROR = Phase.ERROR.value
    CAPTURING = Phase.CAPTURING.value
    WARMUP = Phase.WARMUP.value
    DISPLAY_IMAGE = Phase.DISPLAY_IMAGE.value
    SCREENSAVER = Phase.SCREENSAVER.value
    JAMMING = Phase.JAMMING.value
    SHHH = Phase.SHHH.value

    # Expressions
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    SLEEPY = "sleepy"
    DIZZY = "dizzy"
    CHEEKY = "cheeky"
    HEART = "heart"
    STARRY_EYED = "starry_eyed"
    CONFUSED = "confused"

    # Personas
    FOOTBALL = "football"
    DETECTIVE = "detective"
    SIR_MANO = "sir_mano"
    LOW_BATTERY = "low_battery"
    BEE = "bee"

    ALL = sorted(ALL_EXPRESSION_NAMES)

    VALID_EXPRESSIONS = VALID_EXPRESSIONS
