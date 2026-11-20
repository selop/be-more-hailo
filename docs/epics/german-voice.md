# BMO understands and speaks German

Branch: epic/german-voice (from feature/cloud-llm, which sits on refactor/solid)     Status: in progress

German mode half exists: `BMO_LANGUAGE=de` switches the Piper voice, the speech-to-text language and a "respond only in German" rule. The gaps are below.

## Decisions

| # | Question | Answer | Decided |
|---|---|---|---|
| D1 | Which German voice? | `de_DE-ramona-low` for now (already in `[voice.de]` and installed on the Pi). Piper has no medium-quality ramona; kerstin-medium is the closest step up if it is revisited. | 2026-11-20 |
| D2 | Who writes German replies: the local 1.5B model or the hosted one (Haiku, `LLM_BACKEND=cloud`)? | open | open |
| D3 | Wake word | Keep the current model for now | 2026-11-20 |

## Steps

| # | Step | Depends on | Done when | Status |
|---|---|---|---|---|
| 1 | Voice audition | – | – | dropped: the user chose ramona-low (D1) |
| 2 | Measure German speech-to-text on the NPU with ten German sentences | – | At least 8 of 10 transcripts are usable; numbers in the Log | done (measurement only, see Log) |
| 3 | Confirm ramona-low in German mode on the Pi (speed 1.4 and the speaker filter) | – | `BMO_LANGUAGE=de make say` speaks German; the user confirms speed and tone | todo |
| 4 | German timers and reminders in `core/timers.py` | 2 | New tests in `tests/test_timers.py` pass for "Stell einen Timer auf 5 Minuten" and "Erinnere mich in 10 Minuten an den Tee"; a `make say` turn starts a timer | todo |
| 5 | German keywords for photo, picture, music and web search, moved into the character pack per language | 2 | Tests: "Mach ein Foto", "Spiel ein Lied", "Mal mir ein Bild von einer Katze", "Wie wird das Wetter morgen in Berlin?" each take the right route | todo |
| 6 | Translate the seven lines missing from `[lines.de]` (`confused_reset`, `empty_reply`, `stream_error`, `think_error`, `vision_error`, `vision_missing`, `vision_timeout`) | – | A test fails when a key in `[lines.en]` has no German line | todo |
| 7 | Pronunciation fixes per language (the saved file maps "bmo" to "beemo", wrong for a German voice); German misheard spellings in `heard_as` | 3 | The name is spoken correctly in a German `make say` turn; the user confirms | todo |
| 8 | German system prompt with German date and time, instead of one "answer in German" sentence | D2 | Five German `make say` questions get German answers in character; traces in the Log | todo |
| 9 | German vision, search-summary and idle prompts, idle topics and stock phrases | 8 | A German photo turn and a forced screensaver thought are in German | todo |
| 10 | German refusal phrases in the stuck-loop detection (`core/llm.py`) | 8 | A test with a German refusal resets the history | todo |
| 11 | German sound clips (`greeting_sounds_de`, `ack_sounds_de`, `thinking_sounds_de`) in the chosen voice | 3 | The clips exist in the repository and play in German mode | todo |
| 12 | Reminder rewrite for German, or skipped in German | 4 | A German reminder is spoken back naturally; test | todo |
| 13 | Rollout: `BMO_LANGUAGE=de` in the Pi's service file, clear `sounds/.cache`, restart | 3–12 | The German test script (timer, reminder, photo, music, weather, small talk) passes through `make say`; the user has listened | todo |
| 14 | Strip leaked speech-to-text tokens (`<\|de\|>`) from transcripts; a transcript that is only such residue counts as nothing heard | 2 | Test in `tests/` for `_clean_transcription`; clip 05 no longer reaches the model as text | todo |

## Log

- 2026-11-20: Six German requests tested against the current code (`parse_timer_request`, `detect_pre_llm_action`, `needs_web_search`): timer, reminder, photo, music, picture and weather were all unrecognised.
- 2026-11-20: `[lines.de]` has 9 of the 16 lines in `[lines.en]`.
- 2026-11-20: The repository has no `*_de` sound folders; the Pi has 28 sound files of unknown origin that may be them.
- 2026-11-20: D1 decided: ramona-low for now. Step 1 dropped, step 3 reduced to a listening check.
- 2026-11-20, step 2: ten German sentences, speech-to-text in German mode on the Pi (service stopped for the test).
  - Clean speech (macOS voice Anna, 16 kHz): NPU 8 of 10 exact; "Erinnere mich" became "der inneren mich"; "Wie wird das Wetter morgen in Berlin?" came back as `Wir<|de|>ga<|de|>`. CPU model (`ggml-base.bin`): 9 of 10 usable, same "Erinnere" slip ("Dainnere mich").
  - The same sentences spoken by the ramona-low voice at BMO's speed (1.4): NPU 1 of 10 usable, CPU about 3 of 7. So Whisper barely understands that voice; this says something about how clear it is, not about the user's speech.
  - Consequences: the reminder parser must tolerate a mangled "Erinnere" (step 4) and both digits and number words ("5", "zehn"). Leaked `<|de|>` tokens need stripping (new step 14).
  - Not measured: the user's own voice through BMO's microphone with the fan running. Clips kept on the Pi in `logs/german_anna/`.
