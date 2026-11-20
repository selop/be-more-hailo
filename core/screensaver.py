"""Screensaver idle-thought generation and audio loop.
"""

import datetime
import json
import logging
import os
import random
import re
import subprocess
import time

from .actions import clean_llm_response
from .config import ALSA_DEVICE, CHARACTER, SOUNDS_DIR
from .log import bmo_print
from .search import search_web
from .state import Phase

logger = logging.getLogger(__name__)

def generate_thought(search_result, get_llm):
    """Generate an idle musing using the direct NPU LLM API.

    Parameters
    ----------
    search_result : str
        Web search result text to muse about.
    get_llm : callable
        Returns the LLM instance (e.g. ``core.npu._get_llm``).

    Returns the thought string, or None on failure.
    """
    messages = [
        {"role": "system", "content": CHARACTER.prompt("idle_system")},
        {"role": "user", "content": CHARACTER.prompt("idle_thought", info=search_result[:1200])},
    ]
    try:
        llm = get_llm()
        content = llm.generate_all(prompt=messages, temperature=0.8, max_generated_tokens=300)
        content = clean_llm_response(content)
        if content and "connect" not in content.lower() and "error" not in content.lower():
            return content
    except Exception as e:
        bmo_print("SCREENSAVER", f"LLM request failed: {e}")
    return None


def extract_image_url(phrase):
    """Extract a display_image URL from *phrase*, returning ``(cleaned_phrase, url | None)``."""
    json_match = re.search(r'\{.*?\}', phrase, re.DOTALL)
    if json_match:
        try:
            action_data = json.loads(json_match.group(0))
            if action_data.get("action") == "display_image" and action_data.get("image_url"):
                img_url = action_data["image_url"]
                img_url = img_url.replace("gen.pollinations.ai/image/", "image.pollinations.ai/prompt/")
                cleaned = phrase.replace(json_match.group(0), '').strip()
                return cleaned, img_url
        except Exception as e:
            bmo_print("SCREENSAVER", f"JSON parse error in thought: {e}")
    return phrase, None


class Screensaver:
    """What the assistant does by itself while the screensaver shows: now
    and then another face, a persona gag, or a spoken thought.

    Parameters
    ----------
    store : StateStore
    set_state : callable(str, str)
        Takes a phase or expression name and a status text.
    speak : callable(str, msg=str)
    show_image : callable(str)
        Download and display an image URL.
    get_llm : callable() -> LLM or None
        The language model, or None while it is not loaded.
    stop_event : threading.Event
    """

    CHECK_SECONDS = 30
    THOUGHT_GAP_SECONDS = 1200      # at least this long between spoken thoughts

    def __init__(self, *, store, set_state, speak, show_image, get_llm, stop_event):
        self.store = store
        self.set_state = set_state
        self.speak = speak
        self.show_image = show_image
        self.get_llm = get_llm
        self.stop_event = stop_event
        self.last_audio_time = time.time()

    def run(self):
        while not self.stop_event.is_set():
            time.sleep(self.CHECK_SECONDS)
            self.tick()

    def tick(self):
        """One look at the idle assistant; most of the time nothing happens."""
        store, screensaver = self.store, Phase.SCREENSAVER.value
        if store.phase != Phase.SCREENSAVER:
            return

        # Quiet Hours: 10 PM to 8 AM
        hour = datetime.datetime.now().hour
        if hour >= 22 or hour < 8:
            return

        # Skip if user was recently interacting
        if time.time() - store.last_phase_change < 60:
            return

        # Random visual-only boredom animations (~10% chance every 30s)
        if random.random() < 0.10:
            expr = random.choice(CHARACTER.idle_expressions)
            self.set_state(expr, "Zzz..." if expr == "sleepy" else "...")
            time.sleep(4)
            if store.expression == expr:
                self.set_state(screensaver, "Screensaver...")

        # Random Persona Gags (~5% chance every 30s)
        elif random.random() < 0.05:
            self._persona_gag()
            return

        # ~2% chance every 30 seconds for audio vocalizations
        if random.random() < 0.02:
            current_hour = datetime.datetime.now().hour
            if current_hour >= 22 or current_hour < 7:
                return
            if time.time() - self.last_audio_time > self.THOUGHT_GAP_SECONDS:
                self._speak_thought()

    def _persona_gag(self):
        persona = random.choice(sorted(CHARACTER.personas))
        self.set_state(persona, "...")

        sound_file = os.path.join(SOUNDS_DIR, "personas", f"{persona}.wav")
        if not self.store.muted and os.path.exists(sound_file) and ALSA_DEVICE:
            try:
                from .tts import get_cached_path
                subprocess.Popen(['aplay', '-D', ALSA_DEVICE, '-q', get_cached_path(sound_file)])
            except Exception:
                pass

        time.sleep(8)
        if self.store.expression == persona:
            self.set_state(Phase.SCREENSAVER.value, "Screensaver...")

    def _speak_thought(self):
        phrase = None

        if self.get_llm() is not None:
            try:
                topic = random.choice(CHARACTER.search_topics)
                bmo_print("SCREENSAVER", f"Searching for: {topic}")
                search_result = search_web(topic)

                if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR"):
                    for attempt in range(2):
                        phrase = generate_thought(search_result, self.get_llm)
                        if phrase:
                            bmo_print("SCREENSAVER", f"{CHARACTER.name} muses: {phrase}")

                            phrase, img_url = extract_image_url(phrase)

                            if phrase:
                                self.speak(phrase, msg="Pondering...")

                            if img_url:
                                self.show_image(img_url)

                            self.last_audio_time = time.time()
                            break
                        bmo_print("SCREENSAVER", f"Attempt {attempt + 1} failed, retrying...")
                        time.sleep(5)
            except Exception as e:
                bmo_print("SCREENSAVER", f"Dynamic thought failed: {e}")
        else:
            bmo_print("SCREENSAVER", "LLM server not reachable, skipping thought")

        if not phrase:
            phrase = random.choice(CHARACTER.fallback_phrases)
            bmo_print("SCREENSAVER", f"Fallback: {phrase}")

        if self.store.phase == Phase.SCREENSAVER:
            self.speak(phrase, msg="")
            self.set_state(Phase.SCREENSAVER.value, "")
            self.last_audio_time = time.time()
