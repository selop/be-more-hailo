"""Animation engine — loads face frames and drives the Tkinter animation loop."""

import os
import random
import time

from PIL import Image, ImageTk

from .log import bmo_print


class AnimationEngine:
    """Loads PNG face animations and cycles frames on a Tkinter label.

    Parameters
    ----------
    background_label : tk.Label
        The label widget that displays the current frame.
    all_states : list[str]
        Every BotStates value that has a ``faces/<state>/`` directory.
    get_state : callable() -> str
        Returns the current bot state string.
    get_last_state_change : callable() -> float
        Returns ``time.time()`` of the most recent state change.
    on_screensaver : callable()
        Called when idle timeout triggers the screensaver.
    status_label : tk.Label
        The status text label (hidden during screensaver).
    bubble : ThoughtBubble
        The thought-bubble overlay (hidden during screensaver).
    width, height : int
        Target frame dimensions (default 800×480).
    """

    SCREENSAVER_STATES = [
        "idle", "happy", "sleepy", "heart", "starry_eyed",
        "cheeky", "dizzy", "confused",
        "daydream", "bored", "jamming", "curious",
        "football", "detective", "sir_mano", "low_battery", "bee",
    ]

    def __init__(
        self,
        background_label,
        all_states,
        *,
        get_state,
        get_last_state_change,
        get_display_image,
        on_screensaver,
        status_label,
        bubble,
        screensaver_state,
        display_image_state,
        width=800,
        height=480,
    ):
        self.label = background_label
        self.get_state = get_state
        self.get_last_state_change = get_last_state_change
        self.get_display_image = get_display_image
        self.on_screensaver = on_screensaver
        self.status_label = status_label
        self.bubble = bubble
        self.screensaver_state = screensaver_state
        self.display_image_state = display_image_state
        self.w = width
        self.h = height

        self.animations = {}
        self.current_frame = 0
        self.screensaver_sequences = []

        self._load(all_states)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, all_states):
        base = "faces"
        for state in all_states:
            path = os.path.join(base, state)
            self.animations[state] = []
            if os.path.exists(path):
                files = sorted(f for f in os.listdir(path) if f.lower().endswith(".png"))
                for f in files:
                    img = Image.open(os.path.join(path, f)).resize((self.w, self.h))
                    self.animations[state].append(ImageTk.PhotoImage(img))

        # Screensaver sequences — only expressions that make sense without audio
        self.screensaver_sequences = []
        for state_dir in self.SCREENSAVER_STATES:
            path = os.path.join(base, state_dir)
            if not os.path.isdir(path):
                continue
            files = sorted(f for f in os.listdir(path) if f.lower().endswith(".png"))
            if not files:
                continue
            seq_frames = []
            for f in files:
                try:
                    img = Image.open(os.path.join(path, f)).resize((self.w, self.h))
                    seq_frames.append(ImageTk.PhotoImage(img))
                except Exception as e:
                    bmo_print("SCREENSAVER", f"Failed to load image {f}: {e}")
            if seq_frames:
                self.screensaver_sequences.append((state_dir, seq_frames))

        self._rebuild_screensaver_frames()

    def _rebuild_screensaver_frames(self):
        random.shuffle(self.screensaver_sequences)
        self.animations[self.screensaver_state] = []
        for _name, seq in self.screensaver_sequences:
            self.animations[self.screensaver_state].extend(seq * 2)

    # ------------------------------------------------------------------
    # Per-tick update (called via master.after)
    # ------------------------------------------------------------------

    def update(self, master):
        """Advance one animation frame. Call from the Tkinter after-loop."""
        state = self.get_state()

        # Don't animate while a photo/image is on screen
        if state == self.display_image_state or self.get_display_image() is not None:
            master.after(500, lambda: self.update(master))
            return

        # Idle → screensaver after 60s
        if state == "idle" and (time.time() - self.get_last_state_change()) > 60:
            self.on_screensaver()
            state = self.get_state()

        # Hide/show status label during screensaver
        if state == self.screensaver_state:
            if self.status_label.winfo_ismapped():
                self.status_label.place_forget()
            self.bubble.hide()
        else:
            if not self.status_label.winfo_ismapped():
                self.status_label.place(relx=0.5, rely=0.92, anchor="s")

        frames = self.animations.get(state, []) or self.animations.get("idle", [])
        if frames:
            if state == "warmup":
                if self.current_frame < 12:
                    self.current_frame = (self.current_frame + 1) % 12
            else:
                self.current_frame = (self.current_frame + 1) % len(frames)

            # Re-shuffle screensaver on loop completion
            if state == self.screensaver_state and self.current_frame == 0:
                self._rebuild_screensaver_frames()

            self.label.config(image=frames[min(self.current_frame, len(frames) - 1)])

        speed = self._speed_for_state(state)
        master.after(speed, lambda: self.update(master))

    @staticmethod
    def _speed_for_state(state):
        speeds = {
            "warmup": 1000,
            "speaking": 90,
            "thinking": 500,
            "listening": 250,
            "capturing": 150,
            "screensaver": 400,
            "shhh": 400,
        }
        return speeds.get(state, 500)
