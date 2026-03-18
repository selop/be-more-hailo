# =========================================================================
#  Be More Agent (Hailo Optimized)
#  Simplified for Pi 5 + Hailo-10H + USB Mic
# =========================================================================

import tkinter as tk
import threading
import time
import subprocess
import random
import traceback
import atexit
import importlib
import os

# AI Engines
from openwakeword.model import Model

# Core modules
from core.llm import Brain, init_llm, is_llm_ready, _get_llm
from core.tts import init_audio, shutdown_audio, get_player, get_synthesizer, clean_text_for_speech, precache_voice_sounds
from core.stt import transcribe_audio, init_stt
from core.config import WAKE_WORD_MODEL, ALSA_DEVICE, t
from core.audio_input import wait_for_wakeword, record_until_silence
from core.dispatch import dispatch_stream
from core.screensaver import screensaver_loop
from core.meter import MicMeter
from core.bubble import ThoughtBubble
from core.voice_eq import VoiceEQOverlay
from core.sounds import SoundManager
from core.animation import AnimationEngine
from core.camera import CameraHandler
from core.image_display import download_and_display
from core.log import bmo_print, setup_logging

setup_logging()

# =========================================================================
# 1. HOT-RELOAD — auto-detect file changes and reload safe modules
# =========================================================================

# Modules that can be safely reloaded (no hardware/singleton state).
import core.actions, core.dispatch, core.config, core.voice_eq, core.screensaver, core.meter, core.bubble

_HOT_RELOAD_MODULES = [
    core.actions, core.dispatch, core.config,
    core.voice_eq, core.screensaver,
    core.meter, core.bubble,
]

def _hot_reload_watcher(stop_event, on_reload=None):
    """Background thread: poll mtimes every 2s, reload changed modules."""
    mtimes = {}
    for mod in _HOT_RELOAD_MODULES:
        try:
            path = mod.__file__
            mtimes[mod] = os.path.getmtime(path)
        except Exception:
            pass

    while not stop_event.is_set():
        stop_event.wait(2.0)
        if stop_event.is_set():
            break
        reloaded = []
        for mod in _HOT_RELOAD_MODULES:
            try:
                path = mod.__file__
                mtime = os.path.getmtime(path)
                if mtime != mtimes.get(mod):
                    mtimes[mod] = mtime
                    importlib.reload(mod)
                    reloaded.append(mod.__name__)
            except Exception as e:
                bmo_print("HOT-RELOAD", f"Error reloading {mod.__name__}: {e}")
        if reloaded:
            bmo_print("HOT-RELOAD", f"Reloaded: {', '.join(reloaded)}")
            # Re-import names that the main loop uses directly
            _refresh_imports()
            if on_reload:
                on_reload(reloaded)

def _refresh_imports():
    """Re-bind module-level names after hot-reload."""
    global dispatch_stream, screensaver_loop
    from core.dispatch import dispatch_stream
    from core.screensaver import screensaver_loop

# =========================================================================
# 2. GUI & STATE
# =========================================================================

class BotStates:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    CAPTURING = "capturing"
    WARMUP = "warmup"
    DISPLAY_IMAGE = "display_image"
    SCREENSAVER = "screensaver"
    # Expressions
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    SLEEPY = "sleepy"
    DIZZY = "dizzy"
    CHEEKY = "cheeky"
    HEART = "heart"
    STARRY_EYED = "starry_eyed"
    CONFUSED = "confused"
    SHHH = "shhh"
    JAMMING = "jamming"
    FOOTBALL = "football"
    DETECTIVE = "detective"
    SIR_MANO = "sir_mano"
    LOW_BATTERY = "low_battery"
    BEE = "bee"

    ALL = [
        IDLE, LISTENING, THINKING, SPEAKING, ERROR, CAPTURING, WARMUP,
        HAPPY, SAD, ANGRY, SURPRISED, SLEEPY, DIZZY, CHEEKY, HEART,
        STARRY_EYED, CONFUSED, SHHH, JAMMING, FOOTBALL, DETECTIVE,
        SIR_MANO, LOW_BATTERY, BEE,
    ]

    VALID_EXPRESSIONS = {
        HAPPY, SAD, ANGRY, SURPRISED, SLEEPY, DIZZY,
        CHEEKY, HEART, STARRY_EYED, CONFUSED,
    }


class BotGUI:

    BG_WIDTH, BG_HEIGHT = 800, 480

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        master.attributes('-fullscreen', True)
        master.configure(cursor='none')
        master.bind('<Escape>', self.exit_fullscreen)
        # Long-press (>1.5s) opens voice EQ; short tap mutes
        self._press_time = 0.0
        self._press_valid = False
        master.bind('<ButtonPress-1>', self._on_press)
        master.bind('<ButtonRelease-1>', self._on_release)

        # Events
        self.stop_event = threading.Event()
        self.current_state = BotStates.WARMUP
        self.last_state_change = time.time()

        # Audio State
        self.thinking_audio_process = None
        self.current_display_image = None  # Set when a photo/image is shown on screen
        self.is_muted = False
        self.last_spoke_at = 0.0  # timestamp of last TTS finish — used to suppress false wake words

        # Memory
        self.brain = Brain()

        # --- Sound Manager ---
        self.sounds = SoundManager(is_muted_fn=lambda: self.is_muted)

        # --- UI widgets ---
        self.background_label = tk.Label(master, bg='black')
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)

        self.meter = MicMeter(master)

        self.status_label = tk.Label(
            master,
            text="Initializing...",
            font=('Courier New', 14, 'bold'),
            fg='#1a5c2a',
            bg='#bdffcb',
            padx=12, pady=4,
            relief='flat',
            highlightthickness=0,
        )
        self.status_label.place(relx=0.5, rely=0.92, anchor=tk.S)

        self.bubble = ThoughtBubble(master)
        self.voice_eq = VoiceEQOverlay(master)

        self.mute_label = tk.Label(
            master,
            text="\U0001f507 Muted",
            font=('Courier New', 16, 'bold'),
            fg='#f44336',
            bg='#bdffcb',
            padx=10, pady=5,
            relief='flat',
            highlightthickness=0,
        )

        # --- Animation Engine ---
        self.anim = AnimationEngine(
            self.background_label,
            BotStates.ALL,
            get_state=lambda: self.current_state,
            get_last_state_change=lambda: self.last_state_change,
            get_display_image=lambda: self.current_display_image,
            on_screensaver=lambda: self.set_state(BotStates.SCREENSAVER, "Screensaver..."),
            status_label=self.status_label,
            bubble=self.bubble,
            screensaver_state=BotStates.SCREENSAVER,
            display_image_state=BotStates.DISPLAY_IMAGE,
        )
        self._start_animation_loop()

        # --- Camera Handler ---
        self.camera = CameraHandler(
            set_state=self.set_state,
            speak=self.speak,
            play_sound=self.sounds.play,
            brain=self.brain,
            background_label=self.background_label,
            bubble=self.bubble,
            set_display_image=self._set_display_image,
            clear_display_image=self._clear_display_image,
            stop_thinking_audio=self._stop_thinking_audio,
        )

        # Start Main Thread
        threading.Thread(target=self.main_loop, daemon=True).start()

        # Start Screensaver Audio Thread
        self.last_screensaver_audio_time = time.time()
        threading.Thread(target=self.screensaver_audio_loop, daemon=True).start()

        # Hot-reload watcher
        threading.Thread(
            target=_hot_reload_watcher,
            args=(self.stop_event, self._on_hot_reload),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Display-image helpers (prevent GC of PhotoImage references)
    # ------------------------------------------------------------------

    def _set_display_image(self, tk_img):
        self.current_display_image = tk_img

    def _clear_display_image(self):
        self.current_display_image = None

    # ------------------------------------------------------------------
    # Animation loop bridge
    # ------------------------------------------------------------------

    def _start_animation_loop(self):
        self.master.after(500, lambda: self.anim.update(self.master))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_state(self, state, msg=""):
        if state != self.current_state:
            self.current_state = state
            self.anim.current_frame = 0
            self.last_state_change = time.time()
            bmo_print("STATE", f"{state.upper()}: {msg}")
            if state == BotStates.LISTENING:
                self.meter.show()
            else:
                self.meter.hide()
        if msg:
            self.status_label.config(text=msg)

    def show_user_prompt(self, text):
        self.bubble.show(text)

    def exit_fullscreen(self, event=None):
        self.stop_event.set()
        self.master.quit()

    # ------------------------------------------------------------------
    # Mute / touch handling
    # ------------------------------------------------------------------

    def mute_bmo(self, event=None):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_label.place(relx=0.95, rely=0.05, anchor=tk.NE)
            player = get_player()
            if player:
                player.stop_playback()
                bmo_print("MUTE", "Stopped persistent audio player.")
            try:
                subprocess.run(["killall", "-9", "aplay"], capture_output=True)
            except Exception:
                pass

            old_state = self.current_state
            self.set_state(BotStates.SHHH, "Muted")
            self._stop_thinking_audio()

            def revert_state():
                if self.current_state == BotStates.SHHH:
                    self.set_state(old_state if old_state != BotStates.SHHH else BotStates.IDLE, "Muted")
            self.master.after(3000, revert_state)
        else:
            self.mute_label.place_forget()
            self.set_state(BotStates.HAPPY, "Unmuted!")
            def revert_state():
                if self.current_state == BotStates.HAPPY:
                    self.set_state(BotStates.IDLE, "Ready...")
            self.master.after(2000, revert_state)

    def _on_press(self, event=None):
        self._press_time = time.time()
        self._press_valid = True

    def _on_release(self, event=None):
        if not self._press_valid:
            return
        self._press_valid = False
        if self.voice_eq.is_visible():
            return
        held = time.time() - self._press_time
        if held >= 1.5:
            self._open_eq()
        else:
            self.mute_bmo(event)

    def _open_eq(self):
        bmo_print("EQ", "Opening voice EQ tuner")
        self.voice_eq.show()

    def _on_hot_reload(self, reloaded):
        if 'core.voice_eq' in reloaded:
            was_visible = self.voice_eq.is_visible()
            self.voice_eq.hide()
            from core.voice_eq import VoiceEQOverlay
            self.voice_eq = VoiceEQOverlay(self.master)
            if was_visible:
                self.master.after(0, self.voice_eq.show)

    # ------------------------------------------------------------------
    # Thinking audio (start/stop helper — was copy-pasted 4× before)
    # ------------------------------------------------------------------

    def _stop_thinking_audio(self):
        if self.thinking_audio_process:
            try:
                self.thinking_audio_process.terminate()
            except Exception:
                pass
            self.thinking_audio_process = None

    def _play_thinking_loop(self):
        """Background thread: ack sound → thinking sounds until state changes."""
        ack_proc = self.sounds.play("ack_sounds")
        if ack_proc:
            ack_proc.wait()
        while self.current_state == BotStates.THINKING:
            self.thinking_audio_process = self.sounds.play("thinking_sounds")
            if self.thinking_audio_process:
                self.thinking_audio_process.wait()
            for _ in range(80):
                if self.current_state != BotStates.THINKING:
                    break
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------

    def wait_for_wakeword(self, oww):
        suppressed = {BotStates.SPEAKING, BotStates.JAMMING, BotStates.LISTENING}
        # Suppress for 2s after last speech to prevent BMO's own audio from false-triggering
        post_speech_cooldown = lambda: (time.time() - self.last_spoke_at) < 2.0
        result = wait_for_wakeword(oww, self.stop_event, lambda: self.current_state, suppressed,
                                   extra_suppress_fn=post_speech_cooldown)
        if not result and not self.stop_event.is_set():
            self.set_state(BotStates.ERROR)
            time.sleep(2)
        return result

    def record_audio(self):
        bmo_print("STT", "Recording...")
        return record_until_silence(
            self.stop_event,
            meter_cb=self.meter.feed,
            grace_sec=1.5,
            timeout_sec=30.0,
            filename="input.wav",
        )

    # ------------------------------------------------------------------
    # STT & TTS
    # ------------------------------------------------------------------

    def transcribe(self, filename):
        bmo_print("STT", "Transcribing...")
        return transcribe_audio(filename)

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
                return

            if msg is not None:
                self.set_state(BotStates.SPEAKING, msg)
            else:
                self.current_state = BotStates.SPEAKING
                self.anim.current_frame = 0
                self.last_state_change = time.time()

            if not self.is_muted:
                player.resume()
                player.play(pcm)
            else:
                time.sleep(1.5)

            self.last_spoke_at = time.time()
            if self.current_state == BotStates.SPEAKING:
                if msg is not None:
                    self.set_state(BotStates.IDLE, "Ready...")
                else:
                    self.current_state = BotStates.IDLE
                    self.anim.current_frame = 0
                    self.last_state_change = time.time()
                time.sleep(0.3)

        except Exception as e:
            bmo_print("TTS", f"Hardware error: {e}")

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def start_timer_thread(self, minutes, message):
        def timer_worker():
            bmo_print("TIMER SET", f"for {minutes} minutes. Message: {message}")
            time.sleep(minutes * 60)
            bmo_print("TIMER DONE", message)

            while self.current_state in [BotStates.SPEAKING, BotStates.LISTENING]:
                time.sleep(1)

            old_state = self.current_state
            self.set_state(BotStates.HAPPY, "Reminder!")
            alert_proc = self.sounds.play("ack_sounds")
            if alert_proc:
                alert_proc.wait()

            self.speak(message, msg="Reminder!")

            time.sleep(1)
            if self.current_state == BotStates.IDLE:
                self.set_state(old_state if old_state != BotStates.HAPPY else BotStates.IDLE, "Ready")

        threading.Thread(target=timer_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Dispatch + speak helper (used by both primary and follow-up turns)
    # ------------------------------------------------------------------

    def _dispatch_and_speak(self, user_text, on_music):
        """Run LLM dispatch and speak all result chunks. Returns the ActionResult."""
        result = dispatch_stream(
            self.brain, user_text,
            on_expression=lambda expr: self.set_state(expr, f"Feeling {expr}..."),
            on_timer=self.start_timer_thread,
            on_music=on_music,
            valid_expressions=BotStates.VALID_EXPRESSIONS,
        )

        tts_start = time.time()
        for chunk in result.speak_chunks:
            self.speak(chunk)
        if result.speak_chunks:
            bmo_print("TTS", f"Total speak time: {time.time()-tts_start:.2f}s ({len(result.speak_chunks)} chunks)")

        return result

    # ------------------------------------------------------------------
    # Main conversation loop
    # ------------------------------------------------------------------

    def main_loop(self):
        time.sleep(1)  # Let UI settle
        boot_start = time.time()

        # --- Boot sequence ---
        self.set_state(BotStates.WARMUP, "Booting...")
        t0 = time.time()
        init_audio(ALSA_DEVICE)
        atexit.register(shutdown_audio)
        bmo_print("BOOT", f"Audio subsystem: {time.time()-t0:.2f}s")

        t0 = time.time()
        precache_voice_sounds(self.sounds.get_voice_files())
        bmo_print("BOOT", f"Voice LPF cache: {time.time()-t0:.2f}s")

        self.sounds.play("boot_sounds")
        self.anim.current_frame = 1

        self.set_state(BotStates.WARMUP, "Loading Brain...")
        t0 = time.time()
        init_llm()
        bmo_print("BOOT", f"LLM (NPU): {time.time()-t0:.2f}s")

        self.anim.current_frame = 12
        self.set_state(BotStates.WARMUP, "Loading Ears...")
        t0 = time.time()
        init_stt()
        bmo_print("BOOT", f"STT (NPU): {time.time()-t0:.2f}s")

        self.anim.current_frame = 13
        self.set_state(BotStates.WARMUP, "Loading Wake Word...")
        t0 = time.time()
        try:
            oww = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
        except Exception as e:
            bmo_print("WAKE", f"Failed to load model: {e}")
            self.set_state(BotStates.ERROR, "Wake Word Error")
            return
        bmo_print("BOOT", f"Wake word (OWW): {time.time()-t0:.2f}s")

        self.anim.current_frame = 14
        time.sleep(0.5)
        self.anim.current_frame = 15
        time.sleep(0.5)

        bmo_print("BOOT", f"Total startup: {time.time()-boot_start:.2f}s")
        self.set_state(BotStates.SPEAKING, "Ready!")
        greeting_proc = self.sounds.play("greeting_sounds")
        if greeting_proc:
            threading.Thread(
                target=lambda: (
                    greeting_proc.wait(),
                    self.set_state(BotStates.IDLE, "Waiting...") if self.current_state == BotStates.SPEAKING else None,
                ),
                daemon=True,
            ).start()
        else:
            self.set_state(BotStates.IDLE, "Waiting...")

        # --- Conversation loop ---
        while not self.stop_event.is_set():
            if not self.wait_for_wakeword(oww):
                continue

            # Record
            self.set_state(BotStates.LISTENING, "Listening...")
            wav_file = self.record_audio()

            if not wav_file:
                self.set_state(BotStates.IDLE, "Ready")
                continue

            # Transcribe
            self.set_state(BotStates.THINKING, "Transcribing...")
            threading.Thread(target=self._play_thinking_loop, daemon=True).start()

            user_text = self.transcribe(wav_file)
            bmo_print("STT", f"User Transcribed: {user_text}")
            self.show_user_prompt(user_text)

            if len(user_text) < 2:
                self.set_state(BotStates.IDLE, "Ready")
                self._stop_thinking_audio()
                continue

            # LLM
            self.set_state(BotStates.THINKING, "Thinking...")
            self._stop_thinking_audio()

            try:
                def on_music():
                    def music_worker():
                        while self.current_state in [BotStates.SPEAKING, BotStates.THINKING]:
                            time.sleep(0.5)
                        self.speak(random.choice(t("music_intros")), msg="Getting ready to jam...")
                        bmo_print("MUSIC", "Starting music playback...")
                        music_proc = self.sounds.play("music")
                        if music_proc:
                            self.set_state(BotStates.JAMMING, "Jamming!")
                            bmo_print("MUSIC", "Now playing! State set to JAMMING")
                            music_proc.wait()
                            bmo_print("MUSIC", "Playback finished")
                            time.sleep(1)
                            if self.current_state == BotStates.JAMMING:
                                self.set_state(BotStates.IDLE, "Ready")
                        else:
                            bmo_print("MUSIC", "No music files found or muted!")
                            self.speak(t("no_music"))
                    threading.Thread(target=music_worker, daemon=True).start()

                result = self._dispatch_and_speak(user_text, on_music)

                if result.take_photo:
                    self.camera.handle_photo(user_text, lambda: self.current_state)

                if result.voice_eq:
                    if self.voice_eq.is_visible():
                        bmo_print("EQ", "Voice command closing voice EQ tuner")
                        self.speak("Closing BMO's voice tuner!")
                        self.master.after(0, self.voice_eq.hide)
                    else:
                        bmo_print("EQ", "Voice command opening voice EQ tuner")
                        self.speak("Opening BMO's voice tuner!")
                        self.master.after(0, self._open_eq)

                if result.image_url:
                    self.speak(t("draw_image"))
                    self.set_state(BotStates.DISPLAY_IMAGE, "Showing Image...")
                    bmo_print("IMAGE", f"Starting image display for: {result.image_url}")
                    download_and_display(
                        result.image_url,
                        self.background_label,
                        self.master,
                        self._set_display_image,
                    )

            except Exception as e:
                bmo_print("ERROR", f"LLM/TTS pipeline: {e}")
                traceback.print_exc()

            self.set_state(BotStates.IDLE, "Ready")

    # ------------------------------------------------------------------
    # Screensaver audio
    # ------------------------------------------------------------------

    def screensaver_audio_loop(self):
        def display_image_cb(img_url):
            bmo_print("SCREENSAVER", f"Downloading image from: {img_url}")
            self.set_state(BotStates.DISPLAY_IMAGE, "Visualizing...")
            success = download_and_display(
                img_url,
                self.background_label,
                self.master,
                self._set_display_image,
            )
            if success:
                time.sleep(10)
            if self.current_state == BotStates.DISPLAY_IMAGE:
                self.set_state(BotStates.SCREENSAVER, "Sleeping...")

        screensaver_loop(
            stop_event=self.stop_event,
            get_state=lambda: self.current_state,
            set_state=self.set_state,
            speak_fn=self.speak,
            play_sound_fn=self.sounds.play,
            is_muted_fn=lambda: self.is_muted,
            display_image_fn=display_image_cb,
            get_last_state_change=lambda: self.last_state_change,
            get_last_audio_time=lambda: self.last_screensaver_audio_time,
            set_last_audio_time=lambda t: setattr(self, 'last_screensaver_audio_time', t),
            is_llm_ready_fn=is_llm_ready,
            get_llm_fn=_get_llm,
            screensaver_state=BotStates.SCREENSAVER,
            display_image_state=BotStates.DISPLAY_IMAGE,
            expression_states=[BotStates.HEART, BotStates.SLEEPY, BotStates.STARRY_EYED, BotStates.DIZZY],
            persona_states=[BotStates.FOOTBALL, BotStates.DETECTIVE, BotStates.SIR_MANO, BotStates.LOW_BATTERY, BotStates.BEE],
            alsa_device=ALSA_DEVICE,
        )


if __name__ == "__main__":
    import signal
    root = tk.Tk()
    app = BotGUI(root)
    signal.signal(signal.SIGINT, lambda *_: (app.stop_event.set(), root.quit()))
    root.mainloop()
