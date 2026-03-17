"""Sound loading and playback manager for BMO."""

import os
import random
import subprocess

from .config import ALSA_DEVICE, LANGUAGE
from .tts import get_cached_path
from .log import bmo_print


# Voice-line categories get the BMO lo-fi LPF applied.
# These are pre-filtered at boot into sounds/.cache/ so play_sound()
# just plays the cached WAV via aplay — zero runtime overhead.
VOICE_SOUND_CATEGORIES = {
    "boot_sounds", "greeting_sounds", "ack_sounds",
    "thinking_sounds", "analyzing_sounds",
}


class SoundManager:
    """Loads categorized WAV files and plays them via aplay."""

    def __init__(self, is_muted_fn):
        self._is_muted = is_muted_fn
        self.sounds = {
            "boot_sounds": [],
            "greeting_sounds": [],
            "ack_sounds": [],
            "thinking_sounds": [],
            "analyzing_sounds": [],
            "camera_sounds": [],
            "music": [],
        }
        self._load()

    def _load(self):
        base = "sounds"
        lang_suffix = f"_{LANGUAGE}" if LANGUAGE != "en" else ""
        for category in self.sounds:
            # Try language-specific directory first (e.g. greeting_sounds_de), fall back to default
            path = os.path.join(base, category + lang_suffix)
            if not os.path.exists(path):
                path = os.path.join(base, category)
            if os.path.exists(path):
                self.sounds[category] = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(".wav")
                ]

    def get_voice_files(self):
        """Return all WAV paths in voice categories (for LPF pre-caching)."""
        files = []
        for cat in VOICE_SOUND_CATEGORIES:
            files.extend(self.sounds.get(cat, []))
        return files

    def play(self, category):
        """Play a random sound from *category*. Returns the Popen handle or None."""
        if self._is_muted():
            return None
        sounds = self.sounds.get(category, [])
        if not sounds:
            return None
        sound_file = random.choice(sounds)
        bmo_print("AUDIO", f"Playing {category}: {os.path.basename(sound_file)}")
        try:
            if category in VOICE_SOUND_CATEGORIES:
                sound_file = get_cached_path(sound_file)
            return subprocess.Popen(["aplay", "-D", ALSA_DEVICE, "-q", sound_file])
        except Exception as e:
            bmo_print("AUDIO", f"Error playing sound {sound_file}: {e}")
            return None
