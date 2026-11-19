---
name: use-state-store
description: Guide for using BMO's centralized StateStore when implementing features that interact with BMO's phase or expression state. Use this skill when adding new phases, expressions, state-dependent behavior, or wiring callbacks that read/write BMO's state.
user-invocable: true
argument-hint: "[feature description]"
---

# Using the BMO State Store

Read `docs/state-store.md` for the full design rationale. Read `core/state.py` for the implementation.

## Core Concepts

BMO's state has two orthogonal axes:

- **Phase** (`core/state.py:Phase` enum) — what BMO is doing: `IDLE`, `LISTENING`, `THINKING`, `SPEAKING`, `CAPTURING`, `SCREENSAVER`, `JAMMING`, `SHHH`, `DISPLAY_IMAGE`, `ERROR`, `WARMUP`
- **Expression** (string) — what face to display: `"happy"`, `"sad"`, `"detective"`, `"idle"`, etc.

The `StateStore` instance lives on `BotGUI` as `self.store`. All state reads and writes go through it.

## When to change Phase vs Expression

**Change phase** when BMO enters a new operational mode:
```python
self.store.set_phase(Phase.LISTENING, msg="Listening...")
```

**Change expression** when only the face should change (e.g. during speech):
```python
self.store.set_expression("happy")
# Phase stays whatever it was (SPEAKING, SCREENSAVER, etc.)
```

**The bridge** — `BotGUI.set_state(state_str, msg)` auto-routes:
- Phase names (`"idle"`, `"speaking"`) → `store.set_phase()`
- Expression names (`"happy"`, `"football"`) → `store.set_expression()`

Use this when you have a string and don't know if it's a phase or expression.

## Reading state

In `agent_hailo.py` (has direct store access):
```python
if self.store.phase == Phase.SPEAKING:
    ...
if self.store.muted:
    ...
expr = self.store.expression
```

In core modules (use callbacks — no direct store import):
```python
# Module receives callbacks at construction:
def __init__(self, get_phase, get_expression, ...):
    self.get_phase = get_phase

# Later:
if self.get_phase() == "speaking":
    ...
```

## Adding a new Phase

1. Add the value to `Phase` enum in `core/state.py`
2. If it needs a distinct animation speed, add to `AnimationEngine._speed_for_state()` in `core/animation.py`
3. If wake word should be suppressed during this phase, add to the `suppressed` set in `BotGUI.wait_for_wakeword()`
4. Add a pose with the phase's name to `POSES` in `face/index.html` (see "Adding New Face States" in `CLAUDE.md`)

## Adding a new Expression

1. Add to `VALID_EXPRESSIONS` (emotions) or `PERSONA_EXPRESSIONS` (persona gags) in `core/state.py` — this auto-includes it in `BotStates.ALL` for animation loading
2. Add a pose with the same name to `POSES` in `face/index.html`
3. If the LLM should be able to trigger it, the expression name must match what the LLM emits in `{"action": "set_expression", "value": "<name>"}`
4. For screensaver use: add to `expression_states` or `persona_states` lists in `BotGUI.screensaver_audio_loop()`

## Registering phase-change hooks

Hooks fire when phase changes (not on repeated sets of the same phase):
```python
def my_hook(old_phase: Phase, new_phase: Phase, msg: str):
    if new_phase == Phase.LISTENING:
        do_something()

self.store.on_phase_change(my_hook)
```

Hooks fire outside the lock, so they can safely call other store methods.

## Common patterns

**Check phase in a loop** (e.g. waiting for a phase to end):
```python
while self.store.phase == Phase.THINKING:
    time.sleep(0.1)
```

**Temporary expression with revert** (e.g. mute animation):
```python
self.store.set_expression("shhh")
time.sleep(3)
if self.store.expression == "shhh":  # guard: only revert if nothing else changed it
    self.store.set_expression(self.store.phase.value)
```

**Pass state to a core module via callbacks** (keeps modules decoupled):
```python
my_module = MyModule(
    get_phase=lambda: self.store.phase.value,
    get_expression=lambda: self.store.expression,
    is_muted=lambda: self.store.muted,
)
```

## What NOT to put in the store

- Subprocess handles (`thinking_audio_process`) — not state, stays on `BotGUI`
- Module-specific bookkeeping (`last_screensaver_audio_time`) — stays local
- Hardware singletons (LLM, TTS, STT) — managed by their own modules

## Now implement the feature

Apply the patterns above to: $ARGUMENTS
