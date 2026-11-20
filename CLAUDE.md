# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Be More Agent — Hailo-10H Edition** is a fully on-device conversational AI agent (BMO) for Raspberry Pi 5 with Hailo-10H NPU. BMO is a genderless robot character (Adventure Time inspired) with animated expressions, voice interaction, vision, games, timers, and music. Fork of `@brenpoly/be-more-agent`.

## Developer Tasks

`make help` lists everything. The common ones:

```bash
make check        # compile and run the hardware-free tests
make ship         # check, deploy to the Pi, run the tests there, restart the agent
make deploy       # copy the code only (make restart loads it)
make monitor      # live dashboard: temperature, fan, memory, microphone, last turn
make say TEXT="what time is it"   # silent test turn on the Pi, prints its trace
make screenshot   # capture the Pi's display and open it
make face         # open the face page locally; make face-demo feeds it fake events
```

`python .claude/skills/solid-python/solid_check.py` lists classes and functions worth a look under the SOLID rules (see the `solid-python` skill).

The device defaults to `selop@raspi`; use `BMO_HOST=selop@bmo` for the SSD install. `make setup` creates `.venv-dev` from `requirements-dev.txt` for a development machine.

## Running the Project

```bash
# Activate venv first
source venv/bin/activate

# On-device agent (opens the live face in Chromium)
./start_agent.sh
# or directly: python agent_hailo.py

# CLI chat (minimal, for quick testing)
python cli_chat.py

# Systemd services (production)
sudo systemctl start bmo-gui
sudo journalctl -u bmo-gui -f  # view logs
```

Tests that need no hardware live in `tests/` and run with `pytest tests/` or directly (`python tests/test_character.py`, `python tests/test_turn.py`, `python tests/test_brain.py`, `python tests/test_timers.py`, `python tests/test_display.py`).

To test on the device without speaking, use `scripts/bmo-remote.sh` (see `docs/observability.md`): `say "..." --quiet` runs a real turn from text and prints its trace, `turns`/`stats` read past turns, `screenshot` captures the display. `python scripts/bmo_monitor.py --duration 60` reads temperature, fan, throttling, memory and the microphone (level, noise floor, wake-word score) for a minute. Verify a change there before calling it done. Benchmark files (`scripts/benchmark_llm.py`, `scripts/benchmark_stt.py`) are standalone verification scripts run manually on the Pi.

## Architecture

### Engine and Character Pack

```
characters/<id>/character.toml  ──► core/character.py  (who the assistant is)

agent_hailo.py  ───► core/  ──► hailo_platform.genai (NPU) ──► Hailo-10H
                         │
                         ├── character.py    # Loads the character pack (name, prompts, lines, theme, voice)
                         ├── turn.py         # Conversation: turn loop, speaking, timers, music (no Tkinter)
                         ├── trace.py        # Per-turn traces and events written to logs/ (see docs/observability.md)
                         ├── control.py      # Local control socket: inject test turns, query status
                         ├── health.py       # Host readings (temperature, fan, throttling, memory) and the mic monitor
                         ├── display_server.py # Serves the live face page and streams engine events to it
                         ├── state.py        # Centralized StateStore: Phase enum + expression (see docs/state-store.md)
                         ├── config.py       # Paths, models, ALSA devices; exposes the active CHARACTER
                         ├── npu.py          # VDevice + LLM + VLM lifecycle (hardware layer)
                         ├── actions.py      # Keyword matching, response cleaning (pure functions, no HW)
                         ├── llm.py          # Brain class: conversation history + inference orchestration
                         ├── dispatch.py     # Stream chunk → action dispatch (camera, music, timer)
                         ├── timers.py       # Recognises timer and reminder requests before the LLM (pure functions)
                         ├── audio_input.py  # Wake word detection + mic recording
                         ├── screensaver.py  # Idle thought generation + animation loop
                         ├── tts.py          # Piper TTS with persistent AudioPlayer
                         ├── stt.py          # Speech2Text on NPU (CPU whisper.cpp fallback)
                         └── search.py       # DuckDuckGo web search integration
```

- **`agent_hailo.py`** — `Agent`: builds the engine, connects it to the live face (the only display), handles touch and the idle-to-screensaver timer. Implements `core.turn.TurnUI`; the conversation loop itself is `core.turn.Conversation`
- **`characters/bmo/`** — BMO's character pack. The engine (`core/`, `agent_hailo.py`, `cli_chat.py`) must stay character-neutral: no names, lines or prompts as string literals. `tests/test_character.py` enforces this
- **`core/`** — Shared modules used by the GUI and `cli_chat.py`. `npu.py` owns all Hailo hardware interaction, `actions.py` and `dispatch.py` are pure-function modules testable without NPU hardware

### AI/ML Stack

| Component | Runtime | Model |
|-----------|---------|-------|
| LLM | Hailo-10H NPU | Qwen2.5-1.5B-Instruct via `hailo_platform.genai.LLM` (direct Python API) |
| Vision (VLM) | Hailo-10H NPU | Qwen2-VL-2B-Instruct via `hailo_platform.genai.VLM` (runs in subprocess — NPU only holds one generative model at a time) |
| STT | Hailo-10H NPU | Whisper-Base via `hailo_platform.genai.Speech2Text` (coexists with LLM on shared VDevice; CPU whisper.cpp fallback) |
| TTS | CPU | Piper en_GB-semaine-medium (persistent AudioPlayer stream) |
| Wake Word | CPU | OpenWakeWord (custom wakeword.onnx) |

LLM + Speech2Text coexist on a shared VDevice (`group_id="SHARED"`) because Whisper (125MB) is small enough. VLM (2.3GB) cannot coexist with LLM (2.3GB) — not a memory issue (8GB available) but a HailoRT 5.1.1 genai runtime limitation that only allows one generative model at a time. VLM runs in a child process with its own VDevice.

### Key Design Patterns

- **State store (`core/state.py`)**: Centralized `StateStore` separates *phase* (what BMO is doing: `Phase.IDLE`, `Phase.SPEAKING`, etc.) from *expression* (what face to display: `"happy"`, `"detective"`, etc.). Phase changes are thread-safe with hooks; expressions change independently. Use `store.set_phase()` for phase transitions, `store.set_expression()` for face-only changes. See `docs/state-store.md` for full design and `/use-state-store` skill for implementation guidance.
- **LLM action dispatch**: LLM outputs JSON on its own line (`{"action": "set_expression", "value": "happy"}`, `{"action": "set_timer", ...}`, `{"action": "play_music"}`, `{"action": "take_photo"}`) which the agent parses and executes. Requests the small model does not handle reliably are recognised before the LLM in `core/actions.py`: camera, image, music (keywords) and timers/reminders (`core/timers.py`)
- **Sentence-streaming TTS**: Response is split sentence-by-sentence; TTS plays each sentence as LLM continues generating
- **System prompt caching**: `llm.save_context()` / `llm.load_context()` caches system prompt KV state at startup — not re-processed every turn
- **Turn loop outside the GUI**: `core.turn.Conversation` runs boot and the turn loop on a worker thread and reports to the display through the `TurnUI` callbacks; it owns `speak()`, timers and music
- **Timer clock**: `Conversation.active_timers()` lists running timers; `StateMirror` publishes them and the face page draws the countdown
- **Wake word suppression**: Wake word detection is suppressed during `Phase.SPEAKING` and `Phase.JAMMING` to prevent BMO's own audio from triggering false detections
- **Pronunciation system**: User corrects pronunciation → LLM appends `!PRONOUNCE: word=phonetic` → saved to `pronunciations.json` → Piper substitutions
- **VLM subprocess swap**: Main process releases STT+LLM+VDevice → forks child → child loads VLM → inference → child exits → parent reloads LLM+STT+system prompt cache
- **Screensaver**: Triggers after 60s idle, cycles through expression personas, speaks random LLM-generated thoughts every ~30 min (respects night hours 10PM-8AM)

### Adding New Face States

Faces are drawn live by `face/index.html`; there are no image frames.
1. Add the name to `characters/bmo/character.toml` under `[face]`: `expressions` (emotion faces the LLM may request), `personas` (screensaver gags) or `screensaver`. `core/state.py` builds `VALID_EXPRESSIONS` and `PERSONA_EXPRESSIONS` from the pack.
2. Add a pose with the same name to `POSES` in `face/index.html`: eye and mouth shape weights, a few numbers (`dotR`, `slant`, `eyeY`, `open`, …) and motion flags (`bounce`, `droop`, `tremble`, `pulse`, `gaze`).
3. If the face needs something that is not an eye or a mouth (a hat, a bee, microphones), add a function to `PROPS` and name it in the pose's `prop`. Props are painted on the canvas layer in face pixels (800×480) and get the time since the pose began.
4. Only add a new shape to the shader when the existing eye and mouth shapes cannot express it.
5. Check it: open `face/index.html?freeze=1&fx=0&pose=<name>` for a still, or step through all faces with the arrow keys. `tests/test_character.py` fails if the pack names a face the page has no pose for.
6. New sound categories also need to be registered in `self.sounds` dict in `SoundManager.__init__()` (`core/sounds.py`)

### Changing or Adding a Character

- All character text lives in `characters/<id>/character.toml`: name, misheard spellings, prompts (with `${placeholder}` variables), spoken lines per language, expressions, theme colours, voice model and filter, idle topics, and `[assets]` paths.
- Select a pack with the `CHARACTER` environment variable (default `bmo`).
- In engine code, read character data through `core.config.CHARACTER` (`.name`, `.prompt(key, ...)`, `.theme`, `.normalize_name()`), and spoken lines through `core.config.t(key)` (string or list) or `core.config.say(key)` (always one string). Never write a character's name or lines into engine code.
- A new spoken line needs a key in `[lines.en]` and in `REQUIRED_LINES` in `core/character.py`.
- A new character also needs its own faces, a wake-word model and a Piper voice.

### Live Face

The face is a web page (`face/index.html`, three.js, CRT tube shader) shown by Chromium in kiosk mode. It is the only display; there is no Tkinter window. `core/display_server.py` serves it on 127.0.0.1:8765 and streams engine events to it (`/events`); touches come back through `POST /action`.

- All 27 faces are poses in `POSES` in `face/index.html`. Eyes and mouth are drawn by the shader, so any pose blends into any other; props (microphones, camera, hat, bee, …) are painted live on a 2D canvas (`PROPS`).
- The speech bubble, status line, mic meter, timer clock, mute label and pictures are painted on a 2D canvas that the shader lays over the face, so they sit inside the tube too.
- The mouth follows the loudness of the audio being played (`speak` event, built by `speech_envelope()`).
- `Agent.run()` restarts Chromium if it exits (at most once per 30 s).
- Tinker on a laptop with `python scripts/face_demo.py` (fake events) or by opening `face/index.html` as a file (demo buttons). Still frames: `?freeze=1&pose=happy&fx=0`.
- Touch: a tap mutes, a 5 s press shuts down.

### Observability

Every turn is traced to `logs/turns.jsonl` on the device (transcript, route, model output, actions, spoken sentences, timings, phases, mic statistics). When adding behaviour to a turn, record what happened with `trace.set()` / `trace.add()` / `trace.mark()` from `core/trace.py` so it can be checked remotely. See `docs/observability.md`.

### Asset Directories

- `face/` — the live face page (`index.html`) and its vendored three.js
- `sounds/` — Categorized audio: `boot_sounds/`, `greeting_sounds/`, `thinking_sounds/`, `analyzing_sounds/`, `ack_sounds/`, `camera_sounds/`, `music/`
- `piper/` — TTS engine binary and voice model (gitignored)
- `whisper.cpp/` — Compiled whisper binary for CPU fallback (gitignored)
- `models/` — HEF files: `Qwen2.5-1.5B-Instruct.hef` (LLM), `Qwen2-VL-2B-Instruct.hef` (VLM), `Whisper-Base.hef` (STT), `ggml-base.en.bin` (CPU fallback) — all gitignored

## Configuration

All configuration lives in `core/config.py`. Key settings:
- `LLM_HEF_PATH` — LLM model HEF (default `./models/Qwen2.5-1.5B-Instruct.hef`)
- `VLM_HEF_PATH` — VLM model HEF (default `./models/Qwen2-VL-2B-Instruct.hef`)
- `WHISPER_HEF_PATH` — STT model HEF (default `./models/Whisper-Base.hef`)
- `ALSA_DEVICE` — Speaker output device (override with `ALSA_DEVICE` env var)
- `MIC_DEVICE_INDEX` — sounddevice input index
- `WAKE_WORD_THRESHOLD` — Wake word sensitivity (default 0.35)
- End of speech is decided by `Endpointer` in `core/audio_input.py`: speech is anything above twice the room's measured noise floor; a recording ends 0.9 s after speech stops, after 4 s without speech, or at 15 s
- `get_system_prompt()` — fills the character pack's `prompts.system` with time, date and language rule
- `CHARACTER` — the active character pack; set the `CHARACTER` env var to choose one (default `bmo`)

Optional env vars: `GEMINI_API_KEY` (for Gemini integration), `BMO_LANGUAGE` (en/de).

## Deployment

Deploy to Raspberry Pi via rsync. **Always use `--relative`** to preserve subdirectory structure:
```bash
rsync -avz --relative core/config.py core/tts.py agent_hailo.py selop@raspi:~/Git/be-more-hailo/
```

## Important Conventions

- BMO is genderless — use they/them, never he/she
- BMO's personality is defined in `prompts.system` in `characters/bmo/character.toml` — keep responses warm, short (2-4 sentences), with occasional third-person self-reference
- LLM must respond only in English regardless of input language
- External APIs used: DuckDuckGo (no key), Pollinations.ai (image gen, no key), optional Gemini (key required)
- Hardware target is Raspberry Pi 5 + Hailo-10H — keep code compatible with ARM64/Bookworm
