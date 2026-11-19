# =========================================================================
#  Be More Agent (Hailo Optimized)
#  Simplified for Pi 5 + Hailo-10H + USB Mic
# =========================================================================

import atexit
import tkinter as tk
import threading
import time
import subprocess
import importlib
import os

# Core modules
from core.llm import is_llm_ready, _get_llm
from core.tts import get_player
from core.config import CHARACTER, ALSA_DEVICE, FACE_DISPLAY
from core.turn import Conversation, TurnUI
from core.control import ControlServer
from core.display_server import DisplayServer, StateMirror, speech_envelope
from core import image_display
from core.screensaver import screensaver_loop
from core.meter import MicMeter
from core.bubble import ThoughtBubble
from core.voice_eq import VoiceEQOverlay
from core.sounds import SoundManager
from core.timer_clock import TimerClock
from core.animation import AnimationEngine
from core.camera import CameraHandler
from core.image_display import download_and_display
from core.state import StateStore, Phase, ALL_EXPRESSION_NAMES
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
    global screensaver_loop
    from core.screensaver import screensaver_loop

# =========================================================================
# 2. GUI & STATE
# =========================================================================

class BotGUI(TurnUI):
    """Tkinter shell: window, touch input, face animation and overlays.
    The turn loop itself lives in ``core.turn.Conversation``."""

    BG_WIDTH, BG_HEIGHT = 800, 480
    IMAGE_SECONDS = 30      # how long a generated picture stays on screen

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
        self.store = StateStore()
        self.store.on_phase_change(self._on_phase_change)

        # --- Sound Manager ---
        self.sounds = SoundManager(is_muted_fn=lambda: self.store.muted)

        # --- Conversation (turn loop, speech, timers, music) ---
        self.conversation = Conversation(
            store=self.store,
            sounds=self.sounds,
            ui=self,
            stop_event=self.stop_event,
        )
        # Local control channel for testing (scripts/bmoctl.py)
        ControlServer(self.conversation).start()

        # --- UI widgets ---
        # Fallback window: plain face colour with the status line. The face
        # itself is drawn by the live page on top (core/display_server.py).
        self.background_label = tk.Label(master, bg=CHARACTER.theme["face_bg"])
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)

        self.meter = MicMeter(master)

        self.status_label = tk.Label(
            master,
            text="Initializing...",
            font=('Courier New', 14, 'bold'),
            fg=CHARACTER.theme["status_fg"],
            bg=CHARACTER.theme["face_bg"],
            padx=12, pady=4,
            relief='flat',
            highlightthickness=0,
        )
        self.status_label.place(relx=0.5, rely=0.92, anchor=tk.S)

        self.bubble = ThoughtBubble(master)
        self.voice_eq = VoiceEQOverlay(master)

        # Countdown clock, visible while a timer runs. It steps aside for
        # photos, generated pictures and the voice tuner.
        self.timer_clock = TimerClock(
            master,
            get_timers=self.conversation.active_timers,
            is_hidden=lambda: (
                self.store.display_image is not None
                or self.store.phase in (Phase.CAPTURING, Phase.WARMUP)
                or self.voice_eq.is_visible()
            ),
        )

        self.mute_label = tk.Label(
            master,
            text="\U0001f507 Muted",
            font=('Courier New', 16, 'bold'),
            fg=CHARACTER.theme["alert_fg"],
            bg=CHARACTER.theme["face_bg"],
            padx=10, pady=5,
            relief='flat',
            highlightthickness=0,
        )

        # --- Animation Engine ---
        self.anim = AnimationEngine(
            self.background_label,
            sorted(ALL_EXPRESSION_NAMES),
            get_state=lambda: self.store.expression,
            get_phase=lambda: self.store.phase.value,
            get_last_phase_change=lambda: self.store.last_phase_change,
            get_display_image=lambda: self.store.display_image,
            on_screensaver=lambda: self.store.set_phase(Phase.SCREENSAVER, msg="Screensaver..."),
            status_label=self.status_label,
            bubble=self.bubble,
            screensaver_state=Phase.SCREENSAVER.value,
            display_image_state=Phase.DISPLAY_IMAGE.value,
        )
        self._start_animation_loop()

        # --- Camera Handler ---
        self.camera = CameraHandler(
            set_state=self.set_state,
            speak=self.conversation.speak,
            play_sound=self.sounds.play,
            brain=self.conversation.brain,
            background_label=self.background_label,
            bubble=self.bubble,
            set_display_image=self.store.set_display_image,
            clear_display_image=self.store.clear_display_image,
            stop_thinking_audio=self.conversation.stop_thinking_audio,
        )

        # --- Live face: a three.js page in a local browser on top of this window ---
        self.status_text = "Initializing..."
        self.display = None
        self._meter_sent = 0.0
        if FACE_DISPLAY == "web":
            self._start_live_face()

        # Start the conversation thread (boot + turn loop)
        threading.Thread(target=self.conversation.run, daemon=True).start()

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
    # Live face (core/display_server.py). It mirrors what this window
    # shows; if it cannot start, the window itself remains the display.
    # ------------------------------------------------------------------

    def _start_live_face(self):
        display = DisplayServer(on_action=self._on_face_action)
        if not display.start():
            return
        self.display = display
        atexit.register(display.stop)
        StateMirror(display, self.store, lambda: self.status_text, self.conversation.active_timers).start()
        image_display.image_listeners.append(display.show_image)

        # Mirror the speech bubble
        show, hide = self.bubble.show, self.bubble.hide

        def show_bubble(text, duration_ms=None):
            show(text, duration_ms)
            if text and len(text.strip()) >= 2:
                display.publish("bubble", text=text.strip(), hold_ms=duration_ms or self.bubble.HOLD_MS)

        def hide_bubble():
            hide()
            display.publish("bubble", text="", hold_ms=0)

        self.bubble.show, self.bubble.hide = show_bubble, hide_bubble
        display.open_browser()

    def _on_face_action(self, action):
        """A touch on the live face (it covers this window, so it gets them)."""
        kind = action.get("type")
        bmo_print("FACE", f"Touch: {kind}")
        if kind == "tap":
            self.master.after(0, self.mute_bmo)
        elif kind == "shutdown":
            self.master.after(0, self._shutdown)
        # "long_press" opened the voice tuner on the old display; the live
        # face has no tuner yet.

    # ------------------------------------------------------------------
    # Animation loop bridge
    # ------------------------------------------------------------------

    def _start_animation_loop(self):
        self.master.after(500, lambda: self.anim.update(self.master))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _on_phase_change(self, old_phase, new_phase, msg):
        """Hook fired by StateStore on every phase transition."""
        self.anim.current_frame = 0
        bmo_print("STATE", f"{new_phase.value.upper()}: {msg}")
        if msg:
            self.set_status(msg)
        if new_phase in (Phase.LISTENING, Phase.CAPTURING):
            # A new turn has started: a picture from an earlier turn goes away
            self._clear_image()
        if new_phase == Phase.LISTENING:
            self.meter.show()
        else:
            self.meter.hide()

    def set_state(self, state, msg=""):
        """String bridge — routes phase names to ``store.set_phase()``,
        expression names to ``store.set_expression()``.

        Prefer ``store.set_phase(Phase.X, msg=...)`` directly in new code.
        Kept for callbacks that pass plain strings (screensaver, camera).
        """
        try:
            phase = Phase(state)
            self.store.set_phase(phase, msg=msg)
        except ValueError:
            # Expression-only (e.g. "happy", "football") — face changes, phase stays
            self.store.set_expression(state)
            if msg:
                self.set_status(msg)

    def exit_fullscreen(self, event=None):
        self.stop_event.set()
        if self.display is not None:
            self.display.close_browser()
        self.master.quit()

    # ------------------------------------------------------------------
    # Mute / touch handling
    # ------------------------------------------------------------------

    def mute_bmo(self, event=None):
        self.store.set_muted(not self.store.muted)
        if self.store.muted:
            self.mute_label.place(relx=0.95, rely=0.05, anchor=tk.NE)
            player = get_player()
            if player:
                player.stop_playback()
                bmo_print("MUTE", "Stopped persistent audio player.")
            try:
                subprocess.run(["killall", "-9", "aplay"], capture_output=True)
            except Exception:
                pass

            old_phase = self.store.phase
            self.store.set_phase(Phase.SHHH, msg="Muted")
            self.conversation.stop_thinking_audio()

            def revert_state():
                if self.store.phase == Phase.SHHH:
                    self.store.set_phase(
                        old_phase if old_phase != Phase.SHHH else Phase.IDLE,
                        msg="Muted",
                    )
            self.master.after(3000, revert_state)
        else:
            self.mute_label.place_forget()
            self.store.set_phase(Phase.IDLE, expression="happy", msg="Unmuted!")
            def revert_state():
                if self.store.expression == "happy" and self.store.phase == Phase.IDLE:
                    self.store.set_expression("idle")
                    self.set_status("Ready...")
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
        if held >= 5.0:
            self._shutdown()
        elif held >= 1.5:
            self._open_eq()
        else:
            self.mute_bmo(event)

    def _shutdown(self):
        """Stop the agent from the touchscreen (5s long press)."""
        bmo_print("SHUTDOWN", "Long press shutdown triggered")
        self.store.set_phase(Phase.ERROR, msg="Shutting down...")
        self.stop_event.set()
        if self.display is not None:
            self.display.close_browser()
        self.master.after(500, self.master.quit)

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
    # TurnUI — what the conversation asks the display to do.
    # Called from the conversation's worker threads.
    # ------------------------------------------------------------------

    def set_status(self, text):
        self.status_text = text
        self.status_label.config(text=text)

    def show_user_prompt(self, text):
        self.bubble.show(text)

    def meter_feed(self, volume):
        self.meter.feed(volume)
        if self.display is not None:
            now = time.time()
            if now - self._meter_sent >= 0.05:
                self._meter_sent = now
                self.display.publish("meter", level=round(min(volume / MicMeter.VOL_MAX, 1.0), 3))

    def speech_started(self, pcm, sample_rate):
        if self.display is not None:
            self.display.publish("speak", envelope=speech_envelope(pcm, sample_rate), step_ms=20)

    def set_boot_frame(self, frame):
        self.anim.current_frame = frame
        if self.display is not None:
            self.display.publish("boot", stage=frame)

    def take_photo(self, user_text):
        self.camera.handle_photo(user_text, lambda: self.store.phase.value)

    def voice_tuner_visible(self):
        return self.voice_eq.is_visible()

    def open_voice_tuner(self):
        self.master.after(0, self._open_eq)

    def close_voice_tuner(self):
        self.master.after(0, self.voice_eq.hide)

    def boot_failed(self, error):
        """Exit with a failure status so systemd (Restart=on-failure)
        starts the agent again. A cold-boot model load that timed out
        usually succeeds on the second attempt."""
        bmo_print("BOOT", "Exiting so the service manager can restart the agent")
        time.sleep(3)   # leave the error face visible briefly
        os._exit(1)

    def show_image(self, image_url):
        shown = download_and_display(
            image_url,
            self.background_label,
            self.master,
            self.store.set_display_image,
        )
        if shown:
            # The bubble would cover the picture
            self.bubble.hide()
            # Bring the face back after a while; a new turn clears it sooner
            image = self.store.display_image
            self.master.after(self.IMAGE_SECONDS * 1000, lambda: self._clear_image(image))

    def _clear_image(self, image=None):
        """Remove a displayed picture so the face animates again. With
        *image* given, only if that picture is still the one on screen."""
        if image is None or self.store.display_image is image:
            self.store.clear_display_image()

    # ------------------------------------------------------------------
    # Screensaver audio
    # ------------------------------------------------------------------

    def screensaver_audio_loop(self):
        def display_image_cb(img_url):
            bmo_print("SCREENSAVER", f"Downloading image from: {img_url}")
            self.store.set_phase(Phase.DISPLAY_IMAGE, msg="Visualizing...")
            success = download_and_display(
                img_url,
                self.background_label,
                self.master,
                self.store.set_display_image,
            )
            if success:
                time.sleep(10)
            if self.store.phase == Phase.DISPLAY_IMAGE:
                self.store.set_phase(Phase.SCREENSAVER, msg="Sleeping...")

        screensaver_loop(
            stop_event=self.stop_event,
            get_state=lambda: self.store.phase.value,
            set_state=self.set_state,
            speak_fn=self.conversation.speak,
            play_sound_fn=self.sounds.play,
            is_muted_fn=lambda: self.store.muted,
            display_image_fn=display_image_cb,
            get_last_state_change=lambda: self.store.last_phase_change,
            get_expression=lambda: self.store.expression,
            get_last_audio_time=lambda: self.last_screensaver_audio_time,
            set_last_audio_time=lambda t: setattr(self, 'last_screensaver_audio_time', t),
            is_llm_ready_fn=is_llm_ready,
            get_llm_fn=_get_llm,
            screensaver_state=Phase.SCREENSAVER.value,
            display_image_state=Phase.DISPLAY_IMAGE.value,
            alsa_device=ALSA_DEVICE,
        )


if __name__ == "__main__":
    import signal
    root = tk.Tk()
    app = BotGUI(root)
    signal.signal(signal.SIGINT, lambda *_: (app.stop_event.set(), root.quit()))
    root.mainloop()
