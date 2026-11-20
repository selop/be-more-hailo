# BMO understands and speaks German

Branch: epic/german-voice (not created yet)     Status: planning

German mode half exists: `BMO_LANGUAGE=de` switches the Piper voice, the speech-to-text language and a "respond only in German" rule. The gaps are below.

## Decisions

| # | Question | Answer | Decided |
|---|---|---|---|
| D1 | Which German voice? | open. Asked for: ramona at medium quality. Piper publishes ramona only at low quality (checked against the official voice index, 2026-11-20). Medium or better German voices: kerstin-medium, thorsten-medium, thorsten-high, thorsten_emotional-medium, mls-medium (236 speakers). | open |
| D2 | Who writes German replies: the local 1.5B model or the hosted one (Haiku, `LLM_BACKEND=cloud`)? | open | open |
| D3 | Wake word | Keep the current model for now | 2026-11-20 |

## Steps

| # | Step | Depends on | Done when | Status |
|---|---|---|---|---|
| 1 | Voice audition: synthesise the same three German sentences with ramona-low and each medium/high candidate, play them on the Pi | – | The user has picked one by ear (D1) | todo |
| 2 | Measure German speech-to-text on the NPU: ten recorded German sentences through `bmoctl hear` | – | At least 8 of 10 transcripts are usable; numbers in the Log | todo |
| 3 | Install the chosen voice and set it in `[voice.de]` | 1 | `BMO_LANGUAGE=de make say` speaks German with the chosen voice; the user confirms speed and tone | todo |
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

## Log

- 2026-11-20: Six German requests tested against the current code (`parse_timer_request`, `detect_pre_llm_action`, `needs_web_search`): timer, reminder, photo, music, picture and weather were all unrecognised.
- 2026-11-20: `[lines.de]` has 9 of the 16 lines in `[lines.en]`.
- 2026-11-20: The repository has no `*_de` sound folders; the Pi has 28 sound files of unknown origin that may be them.
