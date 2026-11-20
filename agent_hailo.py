# =========================================================================
#  Be More Agent (Hailo Optimized)
#  Simplified for Pi 5 + Hailo-10H + USB Mic
# =========================================================================

import atexit
import threading
import time
import subprocess
import importlib
import os

# Core modules
from core.llm import is_llm_ready, _get_llm
from core.tts import get_player
from core.config import ALSA_DEVICE
from core.turn import Conversation, TurnUI
from core.control import ControlServer
from core.display_server import DisplayServer, StateMirror, speech_envelope
from core import image_display
from core.screensaver import screensaver_loop
from core.sounds import SoundManager
from core.camera import CameraHandler
from core.image_display import download_and_display
from core.state import StateStore, Phase
from core.log import bmo_print, setup_logging

setup_logging()

# =========================================================================
# 1. HOT-RELOAD — auto-detect file changes and reload safe modules
# =========================================================================

# Modules that can be safely reloaded (no hardware/singleton state).
import core.actions, core.dispatch, core.config, core.screensaver

_HOT_RELOAD_MODULES = [
    core.actions, core.dispatch, core.config, core.screensaver,
]

def _hot_reload_watcher(stop_event):
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

def _refresh_imports():
    """Re-bind module-level names after hot-reload."""
    global screensaver_loop
    from core.screensaver import screensaver_loop

# =========================================================================
# 2. AGENT
# =========================================================================

class Agent(TurnUI):
    """Builds the engine and connects it to the live face (``face/``), the
    only display. The turn loop itself lives in ``core.turn.Conversation``."""

    IMAGE_SECONDS = 30      # how long a generated picture stays on screen
    IDLE_SECONDS = 60       # idle time before the screensaver starts
    BUBBLE_HOLD_MS = 8000
    METER_MAX = 3000.0      # mic level (RMS) that fills the meter

    def __init__(self):
        self.stop_event = threading.Event()
        self.store = StateStore()
        self.store.on_phase_change(self._on_phase_change)
        self.sounds = SoundManager(is_muted_fn=lambda: self.store.muted)
        self.conversation = Conversation(
            store=self.store, sounds=self.sounds, ui=self, stop_event=self.stop_event,
        )
        # Local control channel for testing (scripts/bmoctl.py)
        ControlServer(self.conversation).start()

        self.status_text = "Initializing..."
        self._meter_sent = 0.0
        self.display = DisplayServer(on_action=self._on_face_action)
        if not self.display.start():
            raise SystemExit(1)     # no display: let the service manager retry
        atexit.register(self.display.stop)
        StateMirror(self.display, self.store, lambda: self.status_text, self.conversation.active_timers).start()
        image_display.image_listeners.append(self.display.show_image)
        self.display.open_browser()

        self.camera = CameraHandler(
            set_state=self.set_state,
            speak=self.conversation.speak,
            play_sound=self.sounds.play,
            brain=self.conversation.brain,
            hide_bubble=self._hide_bubble,
            set_display_image=self.store.set_display_image,
            clear_display_image=self.store.clear_display_image,
            stop_thinking_audio=self.conversation.stop_thinking_audio,
        )

        # The face first: reading the models from disk starves Chromium's own
        # start, which at boot left a blank window for minutes.
        deadline = time.time() + 90
        while self.display.connected == 0 and time.time() < deadline:
            time.sleep(0.5)
        bmo_print("FACE", "Page connected" if self.display.connected else "Page not connected after 90 s; loading the models anyway")

        threading.Thread(target=self.conversation.run, daemon=True).start()
        self.last_screensaver_audio_time = time.time()
        threading.Thread(target=self.screensaver_audio_loop, daemon=True).start()
        threading.Thread(target=_hot_reload_watcher, args=(self.stop_event,), daemon=True).start()

    def run(self):
        """Block until shutdown; start the screensaver when idle and keep the browser up."""
        while not self.stop_event.wait(1):
            if (self.store.phase == Phase.IDLE and self.store.display_image is None
                    and time.time() - self.store.last_phase_change > self.IDLE_SECONDS):
                self.store.set_phase(Phase.SCREENSAVER, msg="Screensaver...")
            self.display.ensure_browser()

    @staticmethod
    def _later(seconds, fn):
        timer = threading.Timer(seconds, fn)
        timer.daemon = True
        timer.start()

    def _hide_bubble(self):
        self.display.publish("bubble", text="", hold_ms=0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _on_phase_change(self, old_phase, new_phase, msg):
        """Hook fired by StateStore on every phase transition."""
        bmo_print("STATE", f"{new_phase.value.upper()}: {msg}")
        if msg:
            self.set_status(msg)
        if new_phase in (Phase.LISTENING, Phase.CAPTURING):
            # A new turn has started: a picture from an earlier turn goes away
            self._clear_image()
        if new_phase == Phase.SCREENSAVER:
            self._hide_bubble()

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

    # ------------------------------------------------------------------
    # Touch: tap = mute, 5 s press = shut down (sent by the face page)
    # ------------------------------------------------------------------

    def _on_face_action(self, action):
        kind = action.get("type")
        bmo_print("FACE", f"Touch: {kind}")
        if kind == "tap":
            self.mute_bmo()
        elif kind == "shutdown":
            self._shutdown()

    def mute_bmo(self):
        self.store.set_muted(not self.store.muted)
        if self.store.muted:
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
            self._later(3, revert_state)
        else:
            self.store.set_phase(Phase.IDLE, expression="happy", msg="Unmuted!")
            def revert_state():
                if self.store.expression == "happy" and self.store.phase == Phase.IDLE:
                    self.store.set_expression("idle")
                    self.set_status("Ready...")
            self._later(2, revert_state)

    def _shutdown(self):
        bmo_print("SHUTDOWN", "Long press shutdown triggered")
        self.store.set_phase(Phase.ERROR, msg="Shutting down...")
        self.stop_event.set()

    # ------------------------------------------------------------------
    # TurnUI — what the conversation asks the display to do.
    # Called from the conversation's worker threads.
    # ------------------------------------------------------------------

    def set_status(self, text):
        self.status_text = text

    def show_user_prompt(self, text):
        if text and len(text.strip()) >= 2:
            self.display.publish("bubble", text=text.strip(), hold_ms=self.BUBBLE_HOLD_MS)

    def meter_feed(self, volume):
        now = time.time()
        if now - self._meter_sent >= 0.05:
            self._meter_sent = now
            self.display.publish("meter", level=round(min(volume / self.METER_MAX, 1.0), 3))

    def speech_started(self, pcm, sample_rate):
        self.display.publish("speak", envelope=speech_envelope(pcm, sample_rate), step_ms=20)

    def set_boot_frame(self, frame):
        self.display.publish("boot", stage=frame)

    def take_photo(self, user_text):
        self.camera.handle_photo(user_text, lambda: self.store.phase.value)

    def boot_failed(self, error):
        """Exit with a failure status so systemd (Restart=on-failure)
        starts the agent again. A cold-boot model load that timed out
        usually succeeds on the second attempt."""
        bmo_print("BOOT", "Exiting so the service manager can restart the agent")
        time.sleep(3)   # leave the error face visible briefly
        os._exit(1)

    def show_image(self, image_url):
        if download_and_display(image_url, self.store.set_display_image):
            # The bubble would cover the picture
            self._hide_bubble()
            # Bring the face back after a while; a new turn clears it sooner
            image = self.store.display_image
            self._later(self.IMAGE_SECONDS, lambda: self._clear_image(image))

    def _clear_image(self, image=None):
        """Remove a displayed picture so the face shows again. With
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
            if download_and_display(img_url, self.store.set_display_image):
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
    app = Agent()
    signal.signal(signal.SIGINT, lambda *_: app.stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: app.stop_event.set())
    app.run()
