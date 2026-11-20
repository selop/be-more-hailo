# Weak Links

Ledger of architectural weak links, in rank order (user impact against effort). Maintained by the `/question-architecture` skill. Domains refer to `docs/domain-map.md`. IDs are stable; rank is the table order.

Last sweep: 2026-11-19. The web interface was removed the same day; WEB-1 was dropped with it. The character pack and the Conversation module were added the same day, and line references were updated to the new locations. Entries marked measured were checked on the Pi with `scripts/bmo-remote.sh` (see `docs/observability.md`). All evidence is read from code unless marked otherwise. The Pi was not reachable, so latency figures are estimates.

| Rank | ID | Weak link | Domain | Status |
|---|---|---|---|---|
| 1 | INTENT-1 | Speech waits for the whole LLM reply | Intent, Conversation | open |
| 2 | BRAIN-1 | Clock frozen at boot; prompt and context budget issues | Brain, NPU runtime | open |
| 3 | NPU-1 | A failed photo leaves models unloaded; nothing locks the NPU | NPU runtime | open |
| 4 | TEST-1 | No utterance eval set; tests cover only part of the engine | Intent, Brain | partly fixed |
| 5 | INTENT-2 | Intent routing misfires on substrings | Intent | open |
| 6 | HEAR-1 | End-of-speech detection runs in undefined units | Hearing | fixed (uncommitted), awaiting a spoken check |
| 7 | BRAIN-2 | Pronunciation learning does not work on the device | Brain | open |
| 8 | FACE-1 | A generated image is never cleared | Face | fixed (uncommitted) |
| 9 | VOICE-1 | A reply cannot be cancelled; speakers cut each other off | Voice, Conversation | open |
| 10 | HEAR-2 | Aliased wake-word audio; speech lost between two mic streams | Hearing | partly fixed (uncommitted) |
| 11 | HEAR-3 | Hearing failures are silent | Hearing, Conversation | open |
| 12 | BRAIN-3 | No persistent memory; timers lost on restart | Brain, Conversation | open |
| 13 | PLAT-1 | Boot runs pip; cold boot takes about five minutes | Platform | partly fixed |
| 14 | FACE-2 | Emotion faces requested by the LLM are overwritten before they show | Face, Conversation | open |
| 15 | INTENT-3 | Timers did not work: the LLM does not emit the timer action | Intent, Brain | fixed (uncommitted) |
| 16 | CHAR-1 | The character was hardcoded across the engine | all | fixed (uncommitted) |
| 17 | FACE-3 | The Tkinter shell class does five jobs, so display changes touch conversation behaviour | Face, Conversation, Platform | partly fixed (uncommitted) |

## INTENT-3. Timers did not work (fixed 2026-11-19, uncommitted)
Was: "set a timer for 90 seconds" got "I'm sorry, I can't do that" and no timer, twice, because the feature depended on the 1.5B model emitting the timer action.
Fix: `core/timers.py` recognises timer and timed-reminder requests before the LLM (`detect_pre_llm_action()` in `core/actions.py`), and `Conversation.respond()` speaks a confirmation from the character pack (`lines.timer_set`, `lines.reminder`). Covered by `tests/test_timers.py` and `tests/test_turn.py`.
Measured on the Pi: a 10 s timer and a 15 s reminder were confirmed after 0.3 s and 0.5 s and both fired on time (`timer_set` / `timer_done` in `logs/events.jsonl`).
A countdown clock (`core/timer_clock.py`) shows the remaining time in the top-left corner while a timer runs, confirmed by screenshots; spoken requests were recognised too ("Start timer for 44 seconds.", response 1.0 s).
Still open: English phrasing only; no cancel or list; timers are lost on restart (BRAIN-3); reminders at a clock time ("at 5 pm") still go to the LLM.

## INTENT-1. Speech waits for the whole LLM reply
Evidence: `core/dispatch.py:60` collects every chunk into `speak_chunks` before returning; `core/turn.py:258` speaks only afterwards.
Cost: first word comes after up to 4 sentences / 100 tokens. Measured on the Pi on 2026-11-19: 9 s from transcript to first audio for a two-sentence reply (2 s web search, 7.3 s generation, 1 s synthesis); the first sentence was ready after about 6 s. Injected turns the same day waited 4.2 s, 7.2 s, 10.5 s and 14.1 s (`response_ms`), with the first token after 0.4–0.7 s each time. A fixed 0.3 s sleep follows every sentence (`core/turn.py:123`).
Smallest fix: make `dispatch_stream()` a generator so each sentence is spoken as it arrives, and synthesise sentence N+1 while N plays.

## BRAIN-1. Clock frozen at boot; prompt and context budget issues
Evidence: `core/npu.py:93` caches the system prompt, including the current time (`characters/bmo/character.toml:123`), once; `_prepare_prompt()` at `core/npu.py:109` then drops the live system message.
Cost: asking the time or date returns the boot time until a successful photo forces a reload.
Related:
- The prompt expects `[Web search results for context: ...]` (`characters/bmo/character.toml:137`) but `inject_search_context()` writes `[LIVE DATA: ...]` (`core/actions.py:129`).
- History is trimmed by message count (`core/llm.py:43`), not tokens, against a 2048-token context with a prompt of roughly 650–700 tokens (estimated).
Smallest fix: put time and date in the user turn, align the search tag, trim history by token budget.

## NPU-1. A failed photo leaves models unloaded; nothing locks the NPU
Evidence: `core/camera.py:73` starts `_release_llm` before the camera is checked; the reload at `:141` is skipped by both `except` branches (`:146`–`152`). The screensaver thread generates on the shared LLM (`core/screensaver.py:38`) with no lock in `core/npu.py`.
Cost: after one failed photo, transcription silently uses CPU whisper (about 1.9 s against 0.26 s, documented figures) until restart, and the next turn pays a model reload. What HailoRT does with overlapping generate calls is unknown.
Smallest fix: a context manager in `core/npu.py` that owns release and reload in a `try/finally`, plus a lock around generation.

## TEST-1. No utterance eval set; tests cover only part of the engine
Evidence: `tests/test_character.py` (character pack, engine guard) and `tests/test_turn.py` (turn loop with fake hardware) exist since 2026-11-19. `core/actions.py` routing and `core/dispatch.py` parsing still have no dedicated tests, and there is no set of real utterances with expected routing.
Cost: routing regressions go unnoticed; no objective way to compare prompts or LLMs.
Smallest fix: about 100 real utterances with expected routing under pytest, plus `dispatch_stream()` edge cases (split JSON, several actions in one reply).

## INTENT-2. Intent routing misfires on substrings
Evidence: `detect_pre_llm_action()` at `core/actions.py:83` and `needs_web_search()` at `:118` match bare substrings. Reproduced locally: "Look at the bright side" returns `take_photo`; "Can you draw a conclusion from that?" returns `display_image`; "What should I cook tonight?" triggers a web search.
Cost: a false camera trigger costs the 20–30 s VLM swap, and with no camera attached it triggers NPU-1.
Smallest fix: anchored patterns with word boundaries, checked against the TEST-1 set.

## HEAR-1. End-of-speech detection runs in undefined units
Evidence: `core/audio_input.py:124` opens the recording stream with no `blocksize`; loudness at `:111` is `np.linalg.norm(indata) * 10`, which scales with block length; recording ends after 40 "chunks" (`:138`).
Cost: the wait after the last word is somewhere between about 0.4 s and 1.7 s depending on the driver's block size (the docs and a reverted commit disagree), and the effective threshold moves with it. Paid on every turn.
Smallest fix: log the block size once on the Pi and pin it, then express timeouts in seconds and the threshold as RMS. Extract the decision into a pure function so it can be tested on recordings.

## BRAIN-2. Pronunciation learning does not work on the device
Evidence: `think()` handles `!PRONOUNCE` (`core/llm.py:159`); `stream_think()` (`core/llm.py:190`), the only path the device uses, does not. With the web interface gone, `think()` is used only by `cli_chat.py`.
Cost: a pronunciation correction is never saved and the tag reaches TTS.
Smallest fix: one shared post-processing step used by both paths, or drop `think()` and have the CLI join the stream.

## FACE-1. A generated image was never cleared (fixed 2026-11-19, uncommitted)
Was: after "draw me a small robot" the picture stayed on screen with the speech bubble overlapping it, the face never returned and the screensaver never started. Confirmed with a screenshot from the Pi.
Fix: `BotGUI.show_image()` in `agent_hailo.py` hides the bubble and clears the picture after 30 s (`IMAGE_SECONDS`), or sooner when a new voice turn starts. Verified with screenshots: picture without bubble, then the face back after 30 s.
Still open: the screensaver's own image path is unchanged.

## VOICE-1. A reply cannot be cancelled; speakers cut each other off
Evidence: the speak loop has no stop check (`core/turn.py:258`); `AudioPlayer.play()` clears the stop flag and interrupts whatever is playing (`core/tts.py:136`–`138`); four threads call `speak()`; `speak()` drops to `IDLE` after each sentence (`core/turn.py:122`). Sounds and music go out through `aplay`, a second output path.
Cost: a timer or music intro can truncate an answer; muting mid-reply does not end the reply. Blocks barge-in.
Smallest fix: one serialised speak call in Voice that takes a per-turn cancel token and holds `SPEAKING` for the whole reply. Do it with INTENT-1.

## HEAR-2. Aliased wake-word audio; speech lost between two mic streams
Evidence: `core/audio_input.py:46` downsamples 48 kHz to 16 kHz with `data[::3]` and no low-pass filter; the wake-word stream (`:43`) closes before the recording stream (`:124`) opens. Accuracy impact not measured.
Cost: probably lower wake-word accuracy; first words dropped when the command follows the wake word in one breath. Follow-up turns, barge-in and adaptive silence detection were all reverted on this layer.
Smallest fix: `scipy.signal.resample_poly(data, 1, 3)` now; then one always-on mic stream with a ring buffer feeding wake word and recorder.

## HEAR-3. Hearing failures are silent
Evidence: no speech and empty transcripts return to idle with no feedback (`core/turn.py:370`, `:382`); an STT init failure falls back to CPU with one log line (`core/stt.py:36`); the mic is chosen by index (`core/config.py:105`).
Cost: the user cannot tell "did not hear me" from "broken".
Smallest fix: return a reason with the result so the turn loop can say "didn't catch that"; select the mic by name.

## BRAIN-3. No persistent memory; timers lost on restart
Evidence: `Brain.history` lives in RAM (`core/llm.py:67`); timers are `time.sleep(minutes * 60)` in a daemon thread (`core/turn.py:161`) and depend on the LLM emitting correct JSON.
Cost: BMO forgets the user on every restart; a restart silently drops timers; no cancel or list.
Smallest fix: a small persisted facts file injected into the prompt; timers stored on disk and parsed deterministically before the LLM.

## PLAT-1. Boot runs pip; cold boot takes about five minutes
Evidence: `start_agent.sh:11` runs `pip install -r requirements.txt` on every start with unpinned requirements. Measured on the Pi on 2026-11-19 (root on SD card): reading the 2.36 GB LLM model file took 232 s on a cold boot, about 10 MB/s; total startup 293 s against 18 s warm.
Cost: a restart can silently upgrade a dependency; after power-on the assistant is unusable for about five minutes.
Fixed 2026-11-19 (uncommitted): a boot failure used to kill the loop thread while the GUI kept animating. `Conversation.run()` now reports it and `BotGUI.boot_failed()` exits so systemd restarts the service. `core/npu.py` reads the model file into the page cache first, because the cold load inside HailoRT failed with status 72 after about 3.5 minutes. The service unit now starts at boot (`WantedBy=graphical.target`).
Smallest fix for the rest: remove pip from the start script and pin requirements; faster storage for the cold boot.

## FACE-2. Emotion faces requested by the LLM are overwritten before they show
Evidence: `dispatch_stream()` applies `set_expression` while collecting the reply (`core/dispatch.py:90`), before anything is spoken. `speak()` then sets `Phase.SPEAKING` (`core/turn.py:112`), and a phase change resets the expression to the phase name (`core/state.py:101`). Reproduced in `tests/test_turn.py`: after a reply containing `set_expression: happy`, the expression is not `happy`.
Cost: the "happy", "sad" and other emotion faces the prompt asks the LLM to use are never visible during a spoken reply.
Smallest fix: keep a requested expression across the `SPEAKING` transition, or apply it per sentence once INTENT-1 streams sentences.

## CHAR-1. The character was hardcoded across the engine (fixed 2026-11-19, uncommitted)
Fix: `characters/bmo/character.toml` holds the name, prompts, spoken lines, expressions, personas, theme, voice settings and idle topics; `core/character.py` loads the pack chosen by `CHARACTER`. `tests/test_character.py` fails if any engine string literal contains a character's name.
Still open:
- BMO's faces, sounds and wake word stay in their original folders; the pack points at them through `[assets]`.
- `scripts/generate_sounds.py` is a BMO-specific asset tool outside the pack, and the poses in `face/index.html` are BMO's face.
- The system prompt in the pack mixes personality with the engine's action protocol, so a new character has to copy the protocol text.
- Engine identifiers (`bmo_print`, `apply_bmo_border`, `BMO_LANGUAGE`, the `bmo-gui` service) and many docstrings still carry the name.

## Fixed without a ledger entry

- **Last sentence spoken twice (fixed 2026-11-19, uncommitted).** When a reply reached the four-sentence limit, `stream_think()` left the fourth sentence in its buffer and yielded it again. Found in a turn trace (the same text in `spoken` twice); covered by `tests/test_brain.py`.

## Observed, not yet analysed

- **The fan is audible to the microphone, and the live face keeps it at the higher step.** Measured on the Pi on 2026-11-19 with the live face running: the board sits at 58–62 °C, right at the 60 °C trip point, so the fan stays on step 2 (about 6,360 rpm, against about 3,960 rpm on step 1 earlier the same day). Forcing the fan off for 14 s lowered the mic's median level from 312 to 216 and its minimum from 204 to 125; it rose again when the fan came back. A wake word still scored 0.748 against the 0.35 threshold with the fan on step 2. The face is now capped at 30 frames per second, which lowered the median from 60.0 °C to 56.8 °C on fan step 2. That is not enough on its own: the fan only drops back below 55 °C (5 °C hysteresis), and when forced to step 1 the board crept to 59.5 °C in three minutes, just under the 60 °C trip. Remaining candidates: raise the step-2 trip in `/boot/firmware/config.txt` (`dtparam=fan_temp1=65000`), make the shader cheaper, or move the mic away from the fan.
- **The Pi drops off the network for short periods.** Several SSH sessions were cut on 2026-11-19 although Wi-Fi signal read −30 to −37 dBm. Cause unknown.

- **Web search feeds news to weather questions.** "What's the weather today in ... Germany?" returned a travel article and the reply was about "remote small towns in the UK" (`core/search.py` tries news first).
- **Reply quality of the 1.5B model.** "Tell me something nice" produced "Sure, let me know something nice! How about you tell them?". Use `scripts/bmo-remote.sh say` with a fixed set of prompts to compare models after the HailoRT upgrade.

## FACE-3. The Tkinter shell class does five jobs

Domain: Face (seams with Conversation and Platform). Found 2026-11-20, read from code.

`BotGUI` (`agent_hailo.py:90`, 403 lines, 26 methods) is at once:

1. the place everything is built and started (`agent_hailo.py:110-125`, `210-221`);
2. the Tkinter widgets (`130-171`);
3. the bridge to the live face, including replacing `bubble.show`/`hide` at run time (`238-249`) and a second publish in each display callback (`399-414`);
4. behaviour that is not display: mute stops audio and reverts phases (`314-345`), shutdown, exit on boot failure (`428-434`), how long a picture stays (`436-454`);
5. host of the screensaver, wired with 16 callbacks (`475-492`). The idle-to-screensaver timer sits in `AnimationEngine.update()` (`core/animation.py`), a class whose frames no longer exist.

Cost: no user-visible fault today. Mute, shutdown and the screensaver cannot be tested without Tkinter, and every display change is made twice (window and page).

Smallest fix: move jobs 4 and 5 into `core/` (mute and picture lifetime to `Conversation`, a `Screensaver` class that owns its timer), build the objects in a `main()`, and split the display into a Tk one and a web one behind `TurnUI`. Dropping the Tk display afterwards is then a deletion; it is blocked by the voice tuner and by pictures being stored as `ImageTk` objects.

Partly fixed 2026-11-20 (uncommitted; run on the Pi: turn, mute, screensaver, photo on screen, Chromium restart): the Tkinter display is gone. `core/animation.py`, `bubble.py`, `meter.py`, `timer_clock.py` and `voice_eq.py` (the voice tuner) are deleted; `BotGUI` became `Agent` (about 230 lines), which only publishes to the live face. The idle-to-screensaver timer is in `Agent.run()`, which also restarts Chromium if it exits. Still in `Agent`: mute, shutdown and the screensaver wiring.

Observed 2026-11-20:

- The 30 fps cap made the face strobe purple. The springs in `face/index.html` (`Eased.step`) took one integration step per frame and diverged once speed × frame time neared 1 (mouth while speaking at 33 ms). Measured: alternate frames at brightness 68 instead of 168. Fixed by sub-stepping; 264 captured frames across a turn and a mute toggle show none.
- The first photo after a reboot fails: the vision model's child process hit its 90 s limit (`core/llm.py:348`) about 12 minutes after boot. Likely the 2.3 GB model being read from the SD card for the first time (the language model took 329 s on the same boot); not confirmed.
- The picture service returned HTTP 503 and 402 for two requests, so generated pictures could not be checked.

HEAR-1 and HEAR-2, 2026-11-20:

- Measured on the 13 kept recordings: with the fan on its loud step the room's noise floor is 220 to 245 (RMS), up from 105 to 140, and every recording since then ran to the 30 s limit (7 of 7; one of them had people talking throughout). The driver delivered blocks of 12 to 46 frames, so the old threshold and the "40 chunks" wait meant nothing fixed.
- Fix: `Endpointer` in `core/audio_input.py`. Blocks are pinned to 30 ms, levels are RMS, speech is anything above twice the noise floor the mic monitor measured before the wake word (at least 300), and a recording ends 0.9 s after speech stops, after 4 s without speech, or at 15 s. Replayed on the kept recordings it ends the 30 s ones after 2.2 to 4.6 s and leaves the short ones as they were. `SILENCE_THRESHOLD` is gone.
- Wake word: the 48 to 16 kHz step now uses `resample_poly` instead of dropping samples. Effect on wake scores not measured; there are no wake-word recordings to replay.
- Not done: one always-on mic stream (the gap between wake word and recording remains). Not yet confirmed with a spoken turn on the device.
