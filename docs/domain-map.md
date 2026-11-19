# Domain Map

The project cut into problem domains that a dedicated agent can own. Maintained by the `/question-architecture` skill; update it when files move or a seam changes.

Two rules shape the structure:

- **Engine and character are separate.** The engine (everything under `core/`, plus `agent_hailo.py` and `cli_chat.py`) is character-neutral. Everything that makes the assistant BMO lives in a character pack under `characters/`. `tests/test_character.py` fails if an engine string literal contains a character's name.
- **A seam change needs both owners.** A seam is a call or shared resource that crosses a domain boundary. An agent may change anything inside its domain, but not a seam on its own.

## Domains

| Domain | Problem it solves | Files |
|---|---|---|
| **Hearing** | Getting the user's words in: mic, wake word, endpointing, transcription | `core/audio_input.py`, `core/stt.py`, `core/meter.py`, `tests/test_wakeword.py`, `scripts/find_mic.py`, `scripts/list_devices.py`, `scripts/benchmark_stt.py` |
| **Voice** | Getting speech and sounds out: synthesis, playback, voice filter | `core/tts.py`, `core/sounds.py`, `core/voice_eq.py`, `scripts/benchmark_tts.py` |
| **Brain** | What is said: conversation history, memory, web search | `core/llm.py`, `core/search.py`, `get_system_prompt()` in `core/config.py`, `scripts/benchmark_llm.py` |
| **Intent** | Deciding what the user wants done and turning model output into actions | `core/actions.py`, `core/dispatch.py` |
| **NPU runtime** | Model lifecycle on the Hailo-10H: VDevice, load, release, VLM swap | `core/npu.py`, `core/camera.py`, `models/`, `scripts/hailo_detect.py` |
| **Face** | What is shown: phase and expression state, animation, overlays, screensaver, the Tkinter shell | `core/state.py`, `core/animation.py`, `core/bubble.py`, `core/image_display.py`, `core/screensaver.py`, `agent_hailo.py` |
| **Conversation** | One turn end to end: listen, think, speak, act; plus speaking, timers, music | `core/turn.py`, `tests/test_turn.py` |
| **Platform** | Install, services, deploy, settings, logging, the character loader | `setup.sh`, `setup_services.sh`, `start_agent.sh`, `requirements.txt`, `core/log.py`, `core/config.py`, `core/character.py`, `tests/test_character.py` |
| **Character** (data, not code) | Who the assistant is | `characters/bmo/character.toml`, `faces/`, `sounds/`, `wakeword.onnx`, `generate_faces.py`, `scripts/generate_sounds.py`, `scripts/generate_chiptunes.py`, `scripts/convert_music.py` |

`cli_chat.py` is a minimal Brain harness. The web interface (`web_app.py`, `templates/`) was removed on 2026-11-19; the assistant runs on the Pi only.

## The character pack

```
characters/bmo/
  character.toml   name, heard-as spellings, prompts, spoken lines, expressions,
                   personas, theme colours, voice settings, idle topics, asset paths
```

The engine loads the pack named by the `CHARACTER` setting (default `bmo`) through `core/character.py`, and every module reaches it as `core.config.CHARACTER`.

| Domain | What it reads from the pack |
|---|---|
| Brain | `prompts.system`, `prompts.search_summary`, `prompts.live_data`, `prompts.idle_*`, fallback lines, the name (for break-character detection) |
| NPU runtime | `prompts.vision` |
| Hearing | `assets.wakeword`, `hearing.wake_threshold`, `identity.heard_as` |
| Voice | `voice.<language>`, the filter defaults, `voice.pronunciations`, `assets.sounds` |
| Face | `face.*`, `theme.*`, `assets.faces` |
| Intent | the name (for "sing for <name>"), `lines.timer_done`, `lines.empty_reply` |
| Conversation | spoken lines for music, images and the voice tuner |

BMO's faces, sounds and wake-word model still sit in their original folders; `[assets]` in the pack points at them. A new pack would keep its assets inside its own folder.

## Seams

| Seam | Between | Contract today |
|---|---|---|
| Display | Conversation → Face | `core.turn.TurnUI`: `set_status`, `show_user_prompt`, `meter_feed`, `set_boot_frame`, `take_photo`, `voice_tuner_visible`, `open_voice_tuner`, `close_voice_tuner`, `show_image`. `BotGUI` implements it |
| Wake word and recording | Conversation → Hearing | `wait_for_wakeword(oww, stop_event, get_state, suppressed_states, ...)` and `record_until_silence(stop_event, meter_cb, ...)`; each opens its own mic stream. Conversation loads the wake-word model and decides when detection is suppressed |
| Transcript | Hearing → Conversation | `transcribe_audio(path) -> str`; recording handed over as a WAV file on disk |
| Routing | Intent ↔ Brain | `Brain` calls `detect_pre_llm_action_json()` and `needs_web_search()` before generating |
| Sentence stream | Brain → Intent | `Brain.stream_think()` yields sentence strings or one action JSON string |
| Action result | Intent → Conversation | `dispatch_stream()` returns `ActionResult`; callbacks for expression, timer, music |
| Speech | Conversation → Voice | `Conversation.speak()` calls `get_synthesizer().synthesize(text)` then `get_player().play(pcm)`. Camera and screensaver speak through `Conversation.speak` |
| Model handle | Brain → NPU runtime | `core/llm.py` imports private names from `core/npu.py` (`_get_llm`, `_prepare_prompt`, `_release_llm`, `_vlm_worker`) |
| Shared device | Hearing ↔ NPU runtime | STT and LLM share one VDevice; `core/stt.py` imports the private `_get_vdevice` and `_resolve_hef`; `_release_llm()` calls `release_stt()` and `reload_after_vlm()` calls `init_stt()` |
| Level meter | Hearing → Face | `meter_cb(vol)` in the recorder's loudness units; `MicMeter.VOL_MAX` in `core/meter.py` assumes that scale |
| Phase | everyone → Face | `store.set_phase()` / `store.set_expression()`; see `docs/state-store.md`. `speak()` sets `SPEAKING` per sentence, not per reply |
| Mic and speaker | Hearing ↔ Voice | No code contract. Wake word is suppressed while `Phase.SPEAKING` / `Phase.JAMMING` plus a 2 s cooldown. Sound effects and music play through `aplay`, a second output path outside `AudioPlayer` |

## Known boundary problems

- Brain and Hearing both reach into NPU runtime through underscore-prefixed names, so they cannot be changed independently of it.
- Nothing serialises use of the LLM: the screensaver thread and the conversation thread can both generate.
- `Conversation.speak()` is called from four threads (turn loop, timers, music, screensaver) with no queue between them.
- Conversation still owns pieces of Hearing (wake-word model loading, suppression policy) and of Voice (the speak sequence); they moved out of the GUI but not yet into their domains.
- Hearing and Voice share physical devices with only phase-based suppression between them, and the `aplay` path is not covered by it.
- `TurnUI` methods are called from worker threads and touch Tkinter widgets directly, as the old loop did.

## Still to do in the character split

- Move BMO's `faces/`, `sounds/` and `wakeword.onnx` into `characters/bmo/` (needs the same move on the Pi).
- Move `generate_faces.py` and the sound generators into the pack.
- Split the system prompt into personality (pack) and action protocol (engine).
- Rename engine identifiers that carry the name (`bmo_print`, `apply_bmo_border`, `BMO_LANGUAGE`).
