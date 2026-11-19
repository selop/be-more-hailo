# Observing and Testing the Running Agent

How to see what the agent did and how to drive it without speaking to it. Everything here stays on the device; nothing is sent anywhere.

## From a development machine

`scripts/bmo-remote.sh` runs the tool on the Pi over SSH (`BMO_HOST`, default `selop@raspi`).

```bash
scripts/bmo-remote.sh status                          # phase, health, last turn
scripts/bmo-remote.sh say "what time is it" --quiet   # run a turn from text, print its trace
scripts/bmo-remote.sh hear logs/audio/<id>.wav        # replay a kept recording through speech recognition
scripts/bmo-remote.sh turns -n 10                     # table of recent turns
scripts/bmo-remote.sh show last                       # one turn in full
scripts/bmo-remote.sh stats -n 50                     # median and worst timings, error counts
scripts/bmo-remote.sh events -n 20                    # boot, timer, music and error events
scripts/bmo-remote.sh screenshot [file.png]           # capture the display, copy it back, print the local path
scripts/bmo-remote.sh logs -n 80                      # recent service log
```

On the Pi itself, call `venv/bin/python scripts/bmoctl.py <command>` from the project folder. Add `--json` for machine-readable output.

`--quiet` runs a turn without playing sound: the reply is still generated and synthesised, and the trace is the same.

## Live monitor

`python scripts/bmo_monitor.py` runs on the development machine and shows a dashboard that refreshes every two seconds: board temperature, fan step and speed, throttling and under-voltage flags, clock, load, memory (agent and browser), Wi-Fi signal, the agent's phase and models, the microphone, and the last turn. It lists problems in plain words, keeps every sample in `logs/monitor-<date>.jsonl` on the development machine, and reconnects after a dropped connection.

```bash
python scripts/bmo_monitor.py                 # dashboard until Ctrl+C, then a summary
python scripts/bmo_monitor.py --duration 60   # one line per sample for a minute, then min/median/max
python scripts/bmo_monitor.py --once          # one reading
```

The microphone readings come from the wake-word loop itself (`MicMonitor` in `core/health.py`), over the last 30 seconds:

| Reading | Meaning |
|---|---|
| `level_now`, `level_peak` | RMS of the 16-bit samples (full scale 32768) |
| `noise_floor` | The quietest fifth of the window: the room without speech |
| `wake_score_now`, `wake_score_max` | The wake-word model's score, to compare with `wake_threshold` |
| `detections`, `last_detection_score` | Wake words recognised since the agent started |
| `listening` | False while a turn runs (the recorder has the mic) or when no audio arrives |

When the wake word does not get through, say it with the monitor open: a `wake_score_max` just under the threshold means it was heard but scored too low; a flat level means the mic is not hearing you.

## What a turn trace contains

Each turn is one JSON line in `logs/turns.jsonl`.

| Field | Meaning |
|---|---|
| `source` | `voice` (wake word), `text` (injected text) or `wav` (injected recording) |
| `status` | `ok`, `no_speech`, `empty_transcript` or `error` (with `error`) |
| `transcript` | What was heard |
| `route` | `keyword` (matched before the LLM) or `llm` |
| `search` | Web search time, whether its result was used, and the first 600 characters of the result |
| `llm_raw`, `ttft_ms`, `llm_ms` | The model's full output, time to first token, total generation time |
| `actions`, `action_parse_errors` | Actions parsed from the reply, and JSON that failed to parse |
| `spoken` | Each sentence spoken: text, when, synthesis time, audio length, or an error |
| `response_ms` | The wait the user feels: end of their speech to first audio |
| `marks` | Milliseconds since turn start for `record_done`, `stt_done`, `search_done`, `first_token`, `reply_ready`, `first_audio` |
| `phases` | Every phase change with its time |
| `mic` | Recording length, driver block size, loudest block, silence threshold |
| `wake_score` | Wake-word score at detection (voice turns) |
| `stt_backend` | `npu` or `cpu` |
| `audio_file` | The kept recording of the utterance (voice turns) |

Things outside a turn go to `logs/events.jsonl`: `boot`, `boot_failed`, `timer_set`, `timer_done`, `music`, `tts_error`.

## Recordings

The last 50 utterances are kept in `logs/audio/` so a turn can be replayed with `hear`. Set `TRACE_AUDIO_KEEP=0` in the service environment to keep none, or `TRACE_ENABLED=0` to write no traces at all. `logs/` is ignored by git.

## How it works

- `core/trace.py` holds the active turn's trace. Engine code calls `trace.mark()`, `trace.set()` and `trace.add()`; they do nothing when no turn is active.
- `core/control.py` opens a Unix socket at `logs/control.sock`, readable only by the service user. Nothing listens on the network.
- `Conversation.inject()` in `core/turn.py` queues a test turn; the turn loop picks it up in place of the wake word.
- Screenshots use `grim` on the Pi's Wayland display.

## Adding a trace point

Call `trace.set("my_key", value)` where the fact is known, then show it in `print_turn()` in `scripts/bmoctl.py` if it should appear in the readable view.
