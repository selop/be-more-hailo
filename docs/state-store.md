# State Store: Phase vs Expression

## The Problem

BMO's state was a single flat string (`current_state`) that conflated two orthogonal concerns:

1. **What BMO is doing** — idle, listening, thinking, speaking, capturing, etc.
2. **What face BMO is showing** — happy, sad, heart, detective, etc.

Setting an expression like `set_state("happy")` would overwrite the phase, so code checking "is BMO still speaking?" would break. Five shadow fields (`is_muted`, `last_spoke_at`, `last_state_change`, `current_display_image`, `thinking_audio_process`) lived directly on `BotGUI` with no thread safety and no central place to observe transitions.

```
# Before: expression overwrites phase — lost track of SPEAKING
self.set_state(BotStates.HAPPY, "Feeling happy...")
# self.current_state is now "happy", not "speaking"
```

## The Design

`core/state.py` introduces a clean separation:

```
StateStore
├── phase: Phase          # what BMO is doing (enum: IDLE, SPEAKING, THINKING, ...)
├── expression: str       # what face to show (orthogonal: "happy", "detective", ...)
├── muted: bool
├── display_image: object # PhotoImage ref (prevents GC)
├── last_spoke_at: float
└── last_phase_change: float
```

### Phase (enum)

Represents BMO's operational mode. Exactly one phase is active at a time. Phases drive control flow decisions: wake word suppression, thinking audio loops, screensaver entry.

```python
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
```

### Expression (string)

The face animation to display. Can change independently of phase. During `SPEAKING`, the LLM can set expressions like `"happy"` or `"surprised"` without interrupting the speaking phase.

```python
# After: expression changes, phase stays SPEAKING
store.set_expression("happy")
# store.phase == Phase.SPEAKING (unchanged)
# store.expression == "happy" (face changes)
```

Expressions include emotion faces (`happy`, `sad`, `heart`, ...) and persona faces (`football`, `detective`, `bee`, ...). By default, when phase changes, expression auto-follows (`Phase.IDLE` → expression `"idle"`).

## Thread Safety

Phase transitions are protected by a threading lock. The lock covers the read-modify-write of `phase`, `expression`, and `last_phase_change` as an atomic unit. Hooks fire outside the lock to avoid deadlocks.

Read access is lock-free — Python's GIL makes single-field reads atomic.

```python
def set_phase(self, phase, expression=None, msg=""):
    with self._lock:
        old_phase = self._state.phase
        changed = old_phase != phase
        self._state.phase = phase
        if expression is not None:
            self._state.expression = expression
        elif changed:
            self._state.expression = phase.value
        if changed:
            self._state.last_phase_change = time.time()
    if changed:
        for hook in self._on_phase_change:
            hook(old_phase, phase, msg)
```

## Phase-Change Hooks

Instead of inline side effects in `set_state()`, observers register hooks:

```python
store.on_phase_change(self._on_phase_change)

def _on_phase_change(self, old, new, msg):
    self.anim.current_frame = 0
    if new == Phase.LISTENING:
        self.meter.show()
    else:
        self.meter.hide()
```

This keeps `StateStore` decoupled from GUI widgets. Hooks receive `(old_phase, new_phase, msg)` and fire only when the phase actually changes (not on repeated sets of the same phase).

## How Modules Read State

Modules stay decoupled — they receive string callbacks, not `StateStore` directly:

| Module | Reads | Via |
|--------|-------|-----|
| `AnimationEngine` | expression (face frames), phase (speed, screensaver trigger) | `get_state() -> str`, `get_phase() -> str` |
| `screensaver.py` | phase (is screensaver active?), expression (revert guards) | `get_state() -> str`, `get_expression() -> str` |
| `audio_input.py` | phase (wake word suppression) | `get_state() -> str` (returns `phase.value`) |
| `camera.py` | phase (analyzing sound loop guard) | `get_state() -> str` (unchanged) |
| `dispatch.py` | nothing — uses `on_expression` callback | No state read |
| `sounds.py` | muted flag | `is_muted_fn() -> bool` |

## The `set_state()` Bridge

`BotGUI.set_state(state_str, msg)` is the backward-compatible entry point. It routes automatically:

```python
def set_state(self, state, msg=""):
    try:
        phase = Phase(state)           # "idle", "speaking", etc.
        self.store.set_phase(phase, msg=msg)
    except ValueError:
        self.store.set_expression(state)  # "happy", "football", etc.
    if msg:
        self.status_label.config(text=msg)
```

Phase names (`idle`, `speaking`, `thinking`) trigger a full phase transition with hooks. Expression-only names (`happy`, `football`, `detective`) change the face without touching the phase.

## BotStates Compat Layer

`BotStates` remains as a thin alias class in `core/state.py` so existing string comparisons and animation loading continue to work:

```python
class BotStates:
    IDLE = Phase.IDLE.value        # "idle"
    HAPPY = "happy"
    ALL = sorted(ALL_EXPRESSION_NAMES)
    VALID_EXPRESSIONS = VALID_EXPRESSIONS
```

New code should prefer `Phase` enum for control flow and plain strings for expressions.

## What's NOT in the Store

- `thinking_audio_process` — subprocess handle, not state. Lives on `BotGUI`.
- `last_screensaver_audio_time` — screensaver-specific bookkeeping, stays on `BotGUI`.
- Audio/TTS/LLM singletons — managed by their own modules (`core/tts.py`, `core/llm.py`, `core/npu.py`).
