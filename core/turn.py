"""Conversation: one voice turn end to end.

Owns the turn loop (wake word → record → transcribe → think → speak → act)
together with the things that had no home before: speaking, timers, music
and the thinking sounds. It has no Tkinter dependency. Whatever shows the
turn on screen implements :class:`TurnUI` and is told what happened.

Extracted from ``BotGUI.main_loop`` without behaviour changes.
"""

import atexit
import queue
import random
import threading
import time
import traceback

from . import audio_input, dispatch, trace
from .audio_input import record_until_silence, wait_for_wakeword
from .config import ALSA_DEVICE, LANGUAGE, WAKE_WORD_MODEL, say, t
from .llm import Brain, init_llm
from .log import bmo_print
from .state import Phase, VALID_EXPRESSIONS
from .stt import init_stt, transcribe_audio
from .timers import format_duration
from .tts import (
    clean_text_for_speech, get_player, get_synthesizer, init_audio,
    precache_voice_sounds, shutdown_audio,
)


class TurnUI:
    """What the conversation needs from a display. Every method is a no-op
    here so a headless caller (tests, a future CLI) can pass ``TurnUI()``.
    Methods are called from the conversation's worker threads."""

    def set_status(self, text: str):
        """Show a short status line."""

    def show_user_prompt(self, text: str):
        """Show what the user was heard to say."""

    def meter_feed(self, volume: float):
        """Receive the mic level for each recorded chunk."""

    def set_boot_frame(self, frame: int):
        """Advance the boot animation to *frame*."""

    def take_photo(self, user_text: str):
        """Run the camera flow for *user_text*."""

    def voice_tuner_visible(self) -> bool:
        return False

    def open_voice_tuner(self):
        """Show the voice tuner."""

    def close_voice_tuner(self):
        """Hide the voice tuner."""

    def show_image(self, image_url: str):
        """Download and display a generated image."""

    def speech_started(self, pcm: bytes, sample_rate: int):
        """Called just before a sentence is played, with its audio (int16
        mono PCM), so a display can move the mouth in time with it."""

    def boot_failed(self, error: Exception):
        """Called when audio or model loading raised during boot. The phase
        is already ``ERROR``. A long-running shell should exit with a
        failure status here so the service manager restarts it."""


class Injection:
    """A test turn requested through the control channel: either *text*
    (skips recording and transcription) or a *wav* file (skips recording).
    ``done`` is set when the turn has finished; ``result`` is its trace."""

    def __init__(self, text=None, wav=None, quiet=False):
        self.text = text
        self.wav = wav
        self.quiet = quiet
        self.done = threading.Event()
        self.result = None


class Conversation:
    """Runs the turn loop and everything a turn can trigger.

    Parameters
    ----------
    store : StateStore
        Shared phase/expression state.
    sounds : SoundManager
        Sound-effect and music playback.
    ui : TurnUI
        Display callbacks.
    stop_event : threading.Event
        Set to end the loop.
    brain : Brain or None
        Conversation history and inference. Created if not given.
    """

    def __init__(self, *, store, sounds, ui, stop_event, brain=None):
        self.store = store
        self.sounds = sounds
        self.ui = ui
        self.stop_event = stop_event
        self.brain = brain if brain is not None else Brain()
        # Handle of the currently playing thinking sound (a subprocess)
        self.thinking_audio_process = None
        # Running timers, for the on-screen clock (see active_timers())
        self._timers = []
        self._timers_lock = threading.Lock()
        # Test turns waiting to run (see inject())
        self._injections = queue.Queue()
        # Record every phase change of a turn in its trace
        self.store.on_phase_change(
            lambda old, new, msg: trace.add("phases", [trace.ms(), new.value])
        )

    # ------------------------------------------------------------------
    # Test turns
    # ------------------------------------------------------------------

    def inject(self, text=None, wav=None, quiet=False):
        """Queue a turn that starts without the wake word. Returns the
        :class:`Injection`; wait on its ``done`` event for the trace."""
        injection = Injection(text=text, wav=wav, quiet=quiet)
        self._injections.put(injection)
        return injection

    def _next_injection(self):
        try:
            return self._injections.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Speaking
    # ------------------------------------------------------------------

    def speak(self, text, msg="Speaking..."):
        clean_text = clean_text_for_speech(text)
        if not clean_text or not any(c.isalnum() for c in clean_text):
            return

        bmo_print("TTS", f"Speaking: {clean_text[:30]}...")
        try:
            synth = get_synthesizer()
            player = get_player()

            t0 = time.time()
            pcm = synth.synthesize(clean_text)
            synth_ms = (time.time() - t0) * 1000
            audio_dur = len(pcm) / 2 / 22050 if pcm else 0
            bmo_print("TTS", f"Piper: {synth_ms:.0f}ms synth -> {audio_dur:.1f}s audio")
            if not pcm:
                bmo_print("TTS", "Piper returned no audio")
                trace.add("spoken", {"text": clean_text, "error": "no audio"})
                return

            self.store.set_phase(Phase.SPEAKING, msg=msg or "")
            trace.mark("first_audio")
            trace.add("spoken", {
                "text": clean_text, "at_ms": trace.ms(),
                "synth_ms": int(synth_ms), "audio_s": round(audio_dur, 2),
            })

            if not self.store.muted:
                self.ui.speech_started(pcm, 22050)
                player.resume()
                player.play(pcm)
            else:
                time.sleep(1.5)

            self.store.mark_spoke()
            if self.store.phase == Phase.SPEAKING:
                self.store.set_phase(Phase.IDLE, msg="Ready...")
                time.sleep(0.3)

        except Exception as e:
            bmo_print("TTS", f"Hardware error: {e}")
            trace.add("spoken", {"text": clean_text, "error": repr(e)})
            trace.event("tts_error", text=clean_text[:80], error=repr(e))

    # ------------------------------------------------------------------
    # Thinking sounds
    # ------------------------------------------------------------------

    def stop_thinking_audio(self):
        if self.thinking_audio_process:
            try:
                self.thinking_audio_process.terminate()
            except Exception:
                pass
            self.thinking_audio_process = None

    def _play_thinking_loop(self):
        """Background thread: ack sound → thinking sounds until phase changes."""
        ack_proc = self.sounds.play("ack_sounds")
        if ack_proc:
            ack_proc.wait()
        while self.store.phase == Phase.THINKING:
            self.thinking_audio_process = self.sounds.play("thinking_sounds")
            if self.thinking_audio_process:
                self.thinking_audio_process.wait()
            for _ in range(80):
                if self.store.phase != Phase.THINKING:
                    break
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def active_timers(self):
        """Running timers, soonest first. Each is a dict with ``remaining_s``,
        ``total_s`` and ``message``. Safe to call from any thread."""
        now = time.time()
        with self._timers_lock:
            timers = [
                {"remaining_s": entry["ends_at"] - now, "total_s": entry["total_s"], "message": entry["message"]}
                for entry in self._timers
            ]
        return sorted(timers, key=lambda timer: timer["remaining_s"])

    def _add_timer(self, minutes, message):
        entry = {"ends_at": time.time() + minutes * 60, "total_s": minutes * 60, "message": message}
        with self._timers_lock:
            self._timers.append(entry)
        return entry

    def _remove_timer(self, entry):
        with self._timers_lock:
            if entry in self._timers:
                self._timers.remove(entry)

    def start_timer(self, minutes, message):
        entry = self._add_timer(minutes, message)

        def timer_worker():
            bmo_print("TIMER SET", f"for {minutes} minutes. Message: {message}")
            trace.event("timer_set", minutes=minutes, message=message)
            time.sleep(minutes * 60)
            self._remove_timer(entry)
            bmo_print("TIMER DONE", message)
            trace.event("timer_done", minutes=minutes, message=message)

            while self.store.phase in (Phase.SPEAKING, Phase.LISTENING):
                time.sleep(1)

            old_phase = self.store.phase
            self.store.set_phase(Phase.IDLE, expression="happy", msg="Reminder!")
            alert_proc = self.sounds.play("ack_sounds")
            if alert_proc:
                alert_proc.wait()

            self.speak(message, msg="Reminder!")

            time.sleep(1)
            if self.store.phase == Phase.IDLE:
                self.store.set_phase(
                    old_phase if old_phase != Phase.IDLE else Phase.IDLE,
                    msg="Ready",
                )

        threading.Thread(target=timer_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Music
    # ------------------------------------------------------------------

    def start_music(self):
        def music_worker():
            while self.store.phase in (Phase.SPEAKING, Phase.THINKING):
                time.sleep(0.5)
            self.speak(random.choice(t("music_intros")), msg="Getting ready to jam...")
            bmo_print("MUSIC", "Starting music playback...")
            music_proc = self.sounds.play("music")
            trace.event("music", started=bool(music_proc))
            if music_proc:
                self.store.set_phase(Phase.JAMMING, msg="Jamming!")
                bmo_print("MUSIC", "Now playing! State set to JAMMING")
                music_proc.wait()
                bmo_print("MUSIC", "Playback finished")
                time.sleep(1)
                if self.store.phase == Phase.JAMMING:
                    self.store.set_phase(Phase.IDLE, msg="Ready")
            else:
                bmo_print("MUSIC", "No music files found or muted!")
                self.speak(t("no_music"))

        threading.Thread(target=music_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Hearing
    # ------------------------------------------------------------------

    def wait_for_wakeword(self, oww):
        if not self._injections.empty():
            return True
        suppressed = {Phase.SPEAKING.value, Phase.JAMMING.value, Phase.LISTENING.value}
        # Suppress for 2s after last speech so the character's own audio does not false-trigger
        post_speech_cooldown = lambda: (time.time() - self.store.last_spoke_at) < 2.0
        result = wait_for_wakeword(oww, self.stop_event, lambda: self.store.phase.value, suppressed,
                                   extra_suppress_fn=post_speech_cooldown,
                                   interrupt_fn=lambda: not self._injections.empty())
        if not result and not self.stop_event.is_set():
            self.store.set_phase(Phase.ERROR)
            time.sleep(2)
        return result

    def record_audio(self):
        bmo_print("STT", "Recording...")
        return record_until_silence(
            self.stop_event,
            meter_cb=self.ui.meter_feed,
            grace_sec=1.5,
            timeout_sec=30.0,
            filename="input.wav",
        )

    def transcribe(self, filename):
        bmo_print("STT", "Transcribing...")
        return transcribe_audio(filename)

    # ------------------------------------------------------------------
    # One reply: think, speak, act
    # ------------------------------------------------------------------

    def _dispatch_and_speak(self, user_text):
        """Run LLM dispatch and speak all result chunks. Returns the ActionResult."""
        def _on_expression(expr):
            self.store.set_expression(expr)
            self.ui.set_status(f"Feeling {expr}...")

        # Looked up on the module so a hot-reload of core.dispatch takes effect
        result = dispatch.dispatch_stream(
            self.brain, user_text,
            on_expression=_on_expression,
            on_timer=self.start_timer,
            on_music=self.start_music,
            valid_expressions=VALID_EXPRESSIONS,
        )

        tts_start = time.time()
        for chunk in result.speak_chunks:
            self.speak(chunk)
        if result.speak_chunks:
            bmo_print("TTS", f"Total speak time: {time.time()-tts_start:.2f}s ({len(result.speak_chunks)} chunks)")

        return result

    def respond(self, user_text):
        """Answer *user_text*: generate, speak, then run any requested action."""
        result = self._dispatch_and_speak(user_text)
        trace.set("reply_chunks", len(result.speak_chunks))

        if result.timer_minutes is not None and not result.speak_chunks:
            # The timer was started without the model saying anything
            # (keyword route, or a bare action): confirm it out loud.
            self.speak(say("timer_set", duration=format_duration(result.timer_minutes * 60, LANGUAGE)))

        if result.take_photo:
            self.ui.take_photo(user_text)

        if result.voice_eq:
            if self.ui.voice_tuner_visible():
                bmo_print("EQ", "Voice command closing voice EQ tuner")
                self.speak(say("eq_close"))
                self.ui.close_voice_tuner()
            else:
                bmo_print("EQ", "Voice command opening voice EQ tuner")
                self.speak(say("eq_open"))
                self.ui.open_voice_tuner()

        if result.image_url:
            self.speak(t("draw_image"))
            self.store.set_phase(Phase.DISPLAY_IMAGE, msg="Showing Image...")
            bmo_print("IMAGE", f"Starting image display for: {result.image_url}")
            self.ui.show_image(result.image_url)

        return result

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def boot(self):
        """Load audio, models and the wake word. Returns the wake-word model,
        or ``None`` if it could not be loaded (phase is then ``ERROR``)."""
        boot_start = time.time()

        # --- Boot sequence (stays in WARMUP, status updates directly) ---
        self.ui.set_status("Booting...")
        t0 = time.time()
        init_audio(ALSA_DEVICE)
        atexit.register(shutdown_audio)
        bmo_print("BOOT", f"Audio subsystem: {time.time()-t0:.2f}s")

        t0 = time.time()
        precache_voice_sounds(self.sounds.get_voice_files())
        bmo_print("BOOT", f"Voice LPF cache: {time.time()-t0:.2f}s")

        self.sounds.play("boot_sounds")
        self.ui.set_boot_frame(1)

        self.ui.set_status("Loading Brain...")
        t0 = time.time()
        init_llm()
        bmo_print("BOOT", f"LLM (NPU): {time.time()-t0:.2f}s")

        self.ui.set_boot_frame(12)
        self.ui.set_status("Loading Ears...")
        t0 = time.time()
        init_stt()
        bmo_print("BOOT", f"STT (NPU): {time.time()-t0:.2f}s")

        self.ui.set_boot_frame(13)
        self.ui.set_status("Loading Wake Word...")
        t0 = time.time()
        try:
            from openwakeword.model import Model
            oww = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
        except Exception as e:
            bmo_print("WAKE", f"Failed to load model: {e}")
            self.store.set_phase(Phase.ERROR, msg="Wake Word Error")
            return None
        bmo_print("BOOT", f"Wake word (OWW): {time.time()-t0:.2f}s")

        self.ui.set_boot_frame(14)
        time.sleep(0.5)
        self.ui.set_boot_frame(15)
        time.sleep(0.5)

        bmo_print("BOOT", f"Total startup: {time.time()-boot_start:.2f}s")
        trace.event("boot", total_s=round(time.time() - boot_start, 1))
        self.store.set_phase(Phase.SPEAKING, msg="Ready!")
        greeting_proc = self.sounds.play("greeting_sounds")
        if greeting_proc:
            threading.Thread(
                target=lambda: (
                    greeting_proc.wait(),
                    self.store.set_phase(Phase.IDLE, msg="Waiting...") if self.store.phase == Phase.SPEAKING else None,
                ),
                daemon=True,
            ).start()
        else:
            self.store.set_phase(Phase.IDLE, msg="Waiting...")

        return oww

    # ------------------------------------------------------------------
    # Turn loop
    # ------------------------------------------------------------------

    def run_turn(self, oww):
        """Wait for the wake word (or an injected test turn) and run one
        full turn. Every turn is traced; see ``core/trace.py``."""
        if not self.wait_for_wakeword(oww):
            return

        injection = self._next_injection()
        if injection is None:
            trace.start_turn("voice")
            trace.set("wake_score", audio_input.last_wake_score)
        else:
            trace.start_turn("text" if injection.text else "wav")
        was_muted = self.store.muted
        if injection is not None and injection.quiet:
            self.store.set_muted(True)

        status, error = "ok", None
        try:
            status = self._turn(injection)
        except Exception as e:
            status, error = "error", repr(e)
            bmo_print("ERROR", f"LLM/TTS pipeline: {e}")
            traceback.print_exc()
            self.stop_thinking_audio()
        finally:
            if injection is not None and injection.quiet:
                self.store.set_muted(was_muted)
            self.store.set_phase(Phase.IDLE, msg="Ready")
            result = trace.finish_turn(status, error)
            if injection is not None:
                injection.result = result
                injection.done.set()

    def _turn(self, injection):
        """The body of one turn. Returns its status for the trace:
        ``ok``, ``no_speech`` or ``empty_transcript``."""
        if injection is not None and injection.text:
            # Injected text: nothing to record or transcribe
            user_text = injection.text
            trace.mark("record_done")
            self.store.set_phase(Phase.THINKING, msg="Thinking...")
        else:
            if injection is not None:
                wav_file = injection.wav
            else:
                # Record
                self.store.set_phase(Phase.LISTENING, msg="Listening...")
                wav_file = self.record_audio()
            trace.mark("record_done")

            if not wav_file:
                return "no_speech"
            if injection is None:
                trace.keep_audio(wav_file)

            # Transcribe
            self.store.set_phase(Phase.THINKING, msg="Transcribing...")
            threading.Thread(target=self._play_thinking_loop, daemon=True).start()

            user_text = self.transcribe(wav_file)
            trace.mark("stt_done")
            bmo_print("STT", f"User Transcribed: {user_text}")

        trace.set("transcript", user_text)
        self.ui.show_user_prompt(user_text)

        if len(user_text) < 2:
            self.stop_thinking_audio()
            return "empty_transcript"

        # LLM
        self.ui.set_status("Thinking...")  # same phase, msg-only
        self.stop_thinking_audio()

        self.respond(user_text)
        return "ok"

    def run(self):
        """Boot, then run turns until ``stop_event`` is set. Meant to be the
        target of a background thread."""
        time.sleep(1)  # Let UI settle
        try:
            oww = self.boot()
        except Exception as e:
            # Without this the thread would die silently and the display
            # would sit on its loading screen forever.
            bmo_print("BOOT", f"Startup failed: {e!r}")
            trace.event("boot_failed", error=repr(e))
            traceback.print_exc()
            self.store.set_phase(Phase.ERROR, msg="Startup failed")
            self.ui.boot_failed(e)
            return
        if oww is None:
            return
        while not self.stop_event.is_set():
            self.run_turn(oww)
