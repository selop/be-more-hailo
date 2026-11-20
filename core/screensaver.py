"""Screensaver idle-thought generation and audio loop.
"""

import datetime
import json
import logging
import random
import re
import time

from .actions import clean_llm_response
from .config import CHARACTER, SOUNDS_DIR
from .log import bmo_print
from .search import search_web

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


def screensaver_loop(
    stop_event,
    get_state,
    set_state,
    speak_fn,
    play_sound_fn,
    is_muted_fn,
    display_image_fn,
    get_last_state_change,
    get_last_audio_time,
    set_last_audio_time,
    is_llm_ready_fn,
    get_llm_fn,
    *,
    get_expression=None,
    screensaver_state="screensaver",
    display_image_state="display_image",
    expression_states=None,
    persona_states=None,
    alsa_device=None,
):
    """Main screensaver audio/animation loop.

    All GUI interaction is done through callbacks so this module has no
    Tkinter dependency.

    Parameters
    ----------
    stop_event : threading.Event
    get_state : callable() -> str
        Returns the current phase string.
    set_state : callable(str, str)
    speak_fn : callable(str, msg=str)
    play_sound_fn : callable(str) -> handle
    is_muted_fn : callable() -> bool
    display_image_fn : callable(str) -> None
        Download and display an image URL on screen.
    get_last_state_change : callable() -> float
    get_last_audio_time : callable() -> float
    set_last_audio_time : callable(float)
    is_llm_ready_fn : callable() -> bool
    get_llm_fn : callable
    get_expression : callable() -> str or None
        Returns the current expression string.  When provided, revert
        guards check expression instead of phase.
    screensaver_state, display_image_state : str
    expression_states : list[str] or None
    persona_states : list[str] or None
    alsa_device : str or None
    """
    import os

    if expression_states is None:
        expression_states = CHARACTER.idle_expressions
    if persona_states is None:
        persona_states = sorted(CHARACTER.personas)

    while not stop_event.is_set():
        time.sleep(30)
        if get_state() != screensaver_state:
            continue

        now = datetime.datetime.now()
        hour = now.hour

        # Quiet Hours: 10 PM to 8 AM
        if hour >= 22 or hour < 8:
            continue

        # Skip if user was recently interacting
        if time.time() - get_last_state_change() < 60:
            continue

        # Helper: check current expression (falls back to phase if no expression callback)
        _get_expr = get_expression if get_expression is not None else get_state

        # Random visual-only boredom animations (~10% chance every 30s)
        if random.random() < 0.10:
            expr = random.choice(expression_states)
            set_state(expr, "Zzz..." if expr == "sleepy" else "...")
            time.sleep(4)
            if _get_expr() == expr:
                set_state(screensaver_state, "Screensaver...")

        # Random Persona Gags (~5% chance every 30s)
        elif random.random() < 0.05:
            persona = random.choice(persona_states)
            set_state(persona, "...")

            sound_file = os.path.join(SOUNDS_DIR, "personas", f"{persona}.wav")
            if not is_muted_fn() and os.path.exists(sound_file) and alsa_device:
                try:
                    from .tts import get_cached_path
                    import subprocess
                    cached = get_cached_path(sound_file)
                    subprocess.Popen(['aplay', '-D', alsa_device, '-q', cached])
                except Exception:
                    pass

            time.sleep(8)
            if _get_expr() == persona:
                set_state(screensaver_state, "Screensaver...")
            continue

        # ~2% chance every 30 seconds for audio vocalizations
        if random.random() < 0.02:
            current_hour = datetime.datetime.now().hour
            if current_hour >= 22 or current_hour < 7:
                continue

            if time.time() - get_last_audio_time() > 1200:
                phrase = None

                if is_llm_ready_fn():
                    try:
                        topic = random.choice(CHARACTER.search_topics)
                        bmo_print("SCREENSAVER", f"Searching for: {topic}")
                        search_result = search_web(topic)

                        if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR"):
                            for attempt in range(2):
                                phrase = generate_thought(search_result, get_llm_fn)
                                if phrase:
                                    bmo_print("SCREENSAVER", f"{CHARACTER.name} muses: {phrase}")

                                    phrase, img_url = extract_image_url(phrase)

                                    if phrase:
                                        speak_fn(phrase, msg="Pondering...")

                                    if img_url:
                                        display_image_fn(img_url)

                                    set_last_audio_time(time.time())
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

                if get_state() == screensaver_state:
                    old_state = get_state()
                    speak_fn(phrase, msg="")
                    set_state(old_state, "")
                    set_last_audio_time(time.time())
