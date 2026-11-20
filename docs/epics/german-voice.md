# BMO understands and speaks German

Branch: epic/german-voice (from feature/cloud-llm, which sits on refactor/solid)     Status: blocked (the user has to listen and speak: steps 7 and 13)

German mode half exists: `BMO_LANGUAGE=de` switches the Piper voice, the speech-to-text language and a "respond only in German" rule. The gaps are below.

## Decisions

| # | Question | Answer | Decided |
|---|---|---|---|
| D1 | Which German voice? | `de_DE-thorsten-high`. Chosen by the user knowing it is slow to synthesise (0.90 s of computing per second of speech). | 2026-11-20 |
| D2 | Who writes German replies: the local 1.5B model or the hosted one (Haiku, `LLM_BACKEND=cloud`)? | Hosted (Haiku). Inferred: the user set `LLM_BACKEND=cloud` and a key in the Pi's `.env` on 2026-11-20; not said in words. | 2026-11-20 (inferred) |
| D3 | Wake word | Keep the current model for now | 2026-11-20 |

## Steps

| # | Step | Depends on | Done when | Status |
|---|---|---|---|---|
| 1 | Voice audition | – | – | dropped: the user chose ramona-low (D1) |
| 2 | Measure German speech-to-text on the NPU with ten German sentences | – | At least 8 of 10 transcripts are usable; numbers in the Log | done (measurement only, see Log) |
| 3 | Install thorsten-high, set it in `[voice.de]`, confirm it in German mode on the Pi | – | `make say` speaks German with thorsten-high; the user confirms the voice | done 1ef6e33 (voice confirmed by the user; speed 1.4 not commented on) |
| 4 | German timers and reminders in `core/timers.py` | 2 | New tests in `tests/test_timers.py` pass for "Stell einen Timer auf 5 Minuten" and "Erinnere mich in 10 Minuten an den Tee"; a `make say` turn starts a timer | done 6f2c79a |
| 5 | German keywords for photo, picture, music and web search, next to the English ones in `core/actions.py` (moving them into the pack was dropped: nothing needs it yet) | 2 | Tests: "Mach ein Foto", "Spiel ein Lied", "Mal mir ein Bild von einer Katze", "Wie wird das Wetter morgen in Berlin?" each take the right route | done (this commit) |
| 6 | Translate the seven lines missing from `[lines.de]` (`confused_reset`, `empty_reply`, `stream_error`, `think_error`, `vision_error`, `vision_missing`, `vision_timeout`) | – | A test fails when a key in `[lines.en]` has no German line | done (this commit) |
| 7 | Pronunciation fixes per language (the saved file maps "bmo" to "beemo", wrong for a German voice); German misheard spellings in `heard_as` | 3 | The name is spoken correctly in a German `make say` turn; the user confirms | done (see git log); how "Bimo" sounds is for the user to confirm |
| 8 | German system prompt with German date and time, instead of one "answer in German" sentence | D2 | Five German `make say` questions get German answers in character; traces in the Log | done (see git log) |
| 9 | German vision, search-summary and idle prompts, idle topics and stock phrases | 8 | A German photo turn and a forced screensaver thought are in German | done (this commit) |
| 10 | German refusal phrases in the stuck-loop detection (`core/llm.py`) | 8 | A test with a German refusal resets the history | done (see git log) |
| 11 | All prerecorded spoken clips in German, in the chosen voice (the user, 2026-11-20: "translate all the prerecorded sounds as well") | 3 | Every sound folder with spoken clips has a `_de` folder with the same number of clips in the repository; they play in German mode | done (this commit) |
| 12 | Reminder rewrite for German, or skipped in German | 4 | A German reminder is spoken back naturally; test | done (this commit) |
| 13 | Rollout: `BMO_LANGUAGE=de` in the Pi's service file, clear `sounds/.cache`, restart | 3–12 | The German test script (timer, reminder, photo, music, weather, small talk) passes through `make say`; the user has listened | doing: switched and tested by text; the user has not listened yet |
| 14 | Strip leaked speech-to-text tokens (`<\|de\|>`) from transcripts; a transcript that is only such residue counts as nothing heard | 2 | Test in `tests/` for `_clean_transcription`; clip 05 no longer reaches the model as text | done (this commit) |
| 15 | Tests force the local model (a device `.env` with `LLM_BACKEND=cloud` made the brain tests call the real API) | – | `make pi-test` passes on a Pi configured for the hosted model | done 84e4eb1 |
| 16 | An action's JSON is never cut in two by the sentence splitter | – | Test `test_action_json_after_a_blank_line_is_not_cut_in_two`; a hosted-model turn with an action speaks no JSON | done 84e4eb1 |
| 17 | A blank line in the reply ends the chunk before it, instead of flushing whatever follows word by word ("Yay," was spoken alone) | 16 | Test: JSON, blank line, then a sentence gives the sentence as one chunk | done (this commit) |
| 18 | A dot after a digit does not end a spoken chunk ("der 20." / "November 2026." were spoken apart) | 8 | Test `test_german_date_is_not_cut_at_the_ordinal_dot`; a German date turn on the Pi is spoken in one piece | done (this commit) |
| 19 | Hide the slow synthesis: prepare the next sentence while the current one is playing | 3 | In a four-sentence German turn the gaps between sentences are under 1 s (trace: each `spoke` starts within 1 s of the previous audio ending) | done (this commit), except the first gap: see Log |
| 20 | The recorder gives up when the microphone stream delivers no audio (the agent was stuck on "Listening" after the user's first spoken German query) | – | Test `test_recorder_gives_up_when_the_microphone_delivers_nothing`; on the Pi a stalled recording ends within 18 s and logs `mic_stalled` | done (this commit) for the hang; why the stream stalled is open |

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
- 2026-11-20: The Pi's `.env` now selects the hosted model (set by the user). First turns through Haiku: first text after 2.0 s and 5.6 s.
- 2026-11-20, steps 15 and 16 (found while shipping step 4): `make pi-test` failed because the brain tests reached the real API; and Haiku's timer reply had its action JSON cut in two at a blank line, so no timer started and the JSON was spoken. Both fixed in 84e4eb1; the two steps share that commit. Checked on the Pi: a reply starting with a `set_expression` action ran the action and spoke no JSON, but "Yay," was spoken as its own chunk (new step 17).
- 2026-11-20, step 4: on the Pi "Stell einen Timer auf 5 Sekunden" took the keyword route and rang after 5 s; "Erinnere mich in 6 Sekunden an den Tee" set a reminder with the message "den tee" (events log: timer_set and timer_done for both). The confirmation is still spoken in English because the Pi runs with `BMO_LANGUAGE=en`.
- 2026-11-20, step 14: a transcript containing a leaked decoder token is dropped whole (both such outputs in the measurement were garbage). Checked on the Pi with clip 05 from `logs/german_anna/`, see below.
- 2026-11-20, step 6: `[lines.de]` now has all 16 lines; `test_every_spoken_line_exists_in_every_language_of_the_pack` guards it. The existing German lines were written without umlauts ("fuer", "laeuft"), which the voice would read letter by letter; they now use ü, ä, ö. Not yet heard on the device (step 3).
- 2026-11-20, step 5: on the Pi "Spiel ein Lied für mich" took the music route and "Wie wird das Wetter morgen in Berlin?" ran a web search and was answered in German by the hosted model; "Guten Morgen, wie geht es dir heute?" took no shortcut. Photo and picture were checked by test only (a photo turn unloads the models; the picture service was refusing requests earlier).
  - Seen on the way, not part of this epic: a `--quiet` music turn says "no songs loaded" because mute is treated as no music; lines spoken by the music thread land in the next turn's trace; the weather search returned a stale-looking result (22 degrees in November).
- 2026-11-20, step 12: possessives and "mich"/"mir" in a German reminder are turned round ("meine mutter anzurufen" becomes "deine ..."); "ich" is left alone. Test only; heard on the device with step 13.
- 2026-11-20, steps 8 and 10: the Pi's service now runs with `BMO_LANGUAGE=de` (changed in `/etc/systemd/system/bmo-gui.service`; `sounds/.cache` cleared; the log shows the ramona-low voice loading). Five German turns through the hosted model, all answered in German and in character:
  - "Wie spät ist es und welcher Tag ist heute?": "Es ist gerade 14:20 Uhr, und heute ist Freitag, der 20. November 2026." (correct; first audio after 2.6 s)
  - a robot joke with a `set_expression` action and no spoken JSON (6.6 s), "Die Hauptstadt von Frankreich ist Paris!" (1.3 s), "Wer bist du?" (1.5 s)
  - "Stell einen Timer auf 4 Sekunden": keyword route, confirmed with the German line "Timer läuft! ... sagt in 4 Sekunden Bescheid."
  - The system prompt's instructions stayed in English; only the date, time and language rule are German. A full translation was not needed for the hosted model.
  - Step 10 is covered by `test_german_refusal_resets_the_history` only; no refusal was provoked on the device.
- 2026-11-20, step 18: the date above was spoken as "… der 20." and "November 2026." in two chunks; fixed and tested.
- Open for the user's ear (steps 3 and 7): the ramona-low voice at speed 1.4 through the speaker filter, and the name: the saved pronunciation turns "BMO" into "beemo", which the German voice reads its own way.
- Not measured yet: the user's own German through BMO's microphone.
- 2026-11-20, step 3: voice changed to thorsten-high (114 MB, downloaded to the Pi; `setup.sh` now fetches it). The same ten sentences, spoken through BMO's speaker filter and transcribed by the CPU Whisper model: at speed 1.4, 8 of 10 exact and 2 with one word off ("ist das gerade", "Robote"); at speed 1.0, 7 exact and 3 slightly off. Ramona-low at 1.4 managed about 3 of 7. Speed stays at 1.4 until the user has listened.
- 2026-11-20, step 3, cost of thorsten-high: measured on the Pi with the agent running, seconds of computing per second of speech: thorsten-high 0.90, thorsten-medium 0.15, ramona-low 0.16. In a real turn a 7.6 s sentence took 5.6 s to synthesise and the fourth sentence of a reply started 23.5 s after the question. Clarity by the same Whisper test at speed 1.4: high 8 of 10 exact, medium 6 of 10 (two badly off: "Teig mal" for "Timer", "Erkarte" for "einer Katze"), ramona-low about 3 of 7. thorsten-medium is downloaded to the Pi as well. The pack is set to thorsten-high; the choice between high (clearer, slow) and medium (fast) is the user's.
- 2026-11-20: D1 final: thorsten-high. New step 19 for the pauses it causes. The German clips on the Pi (3 ack, 10 greeting, 5 thinking, dated March 2026) are in the old voice; step 11 regenerates them.
- 2026-11-20, step 19: following sentences are now synthesised while the current one plays. A four-sentence German reply aloud on the Pi: gaps between sentences 2.7 s, 0.3 s and 0.2 s (before, each gap was the full synthesis time, 4 to 6 s). The first gap stays because the opening sentence is shorter than the time the second one needs; closing it means starting synthesis while the model is still writing (INTENT-1 in `docs/weak-links.md`), which is outside this epic.
- 2026-11-20, step 7: pronunciation fixes are per language (`pronunciations_de.json` on the device, defaults from `[voice.de.pronunciations]`). The German voice gets "Bimo" for the name; Whisper heard "Bimo" back in the boot clip. Whether that is how the user wants the name to sound is theirs to say.
- 2026-11-20, step 11: `scripts/generate_sounds.py` now uses the character's own voice through `core.tts` and holds German texts for every spoken category: 20 greeting, 20 thinking, 3 ack, 10 boot, 10 analyzing (63 clips, thorsten-high, in the repository under `sounds/*_de`). The old German clips on the Pi literally said "Backslash" (a shell-quoting slip when they were made) and are replaced. After a restart the agent cached 63 clips and played a German boot and greeting clip. The English `analyzing_sounds` existed only on the Pi and are now in the repository too. The camera shutter and the music are not speech.
- 2026-11-20, step 9:
  - Idle thoughts: with the hosted model on, the screensaver's thought comes from it, in German ("BMO hat gerade etwas Erstaunliches gelernt: Der Mond entfernt sich jedes Jahr etwa 3,8 Zentimeter von der Erde! ..."). German idle topics and stock phrases are in `[idle.de]`.
  - Photos: the local vision model cannot write German. Told to answer in German it produced "Im Bild sehe ich einnection, dasstance,, / Das ist ein einfäiOPY,". It now describes in English and the hosted model retells that in German; a photo turn on the Pi said "Ich sehe einen schlichten, soliden Hintergrund in einer einheitlichen Farbe ..." 38 s after the request. Only the text of the description leaves the device, not the photo. Without the hosted model the description stays English.
- 2026-11-20, step 13 so far: service in German mode since step 8. Text turns passed for timer, reminder ("deine Mutter anzurufen"), photo, music route, weather and small talk. Still missing: the user listening, and the user speaking German to it through the microphone.
- 2026-11-20, step 20: the user's first spoken query after the German rollout left the agent on "Listening". Thread dump (py-spy): the recorder was waiting in its loop and no audio callback was running; the microphone device was held open but delivered nothing for over a minute. The endpointer counted audio only, so it never ended. It now has a wall-clock limit. Why the stream stalled is not known: no USB or audio errors in the kernel log. `vcgencmd get_throttled` reads 0xe0000, so the CPU was throttled for heat at some point since the 11:41 boot (not while the monitor ran until 11:48); the long thorsten-high synthesis runs are the likely time. The wake-word loop reads the microphone with a blocking call and could hang the same way; not changed here.
