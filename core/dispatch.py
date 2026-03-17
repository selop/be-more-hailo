"""Stream action dispatch — consume ``Brain.stream_think()`` chunks and
route JSON actions (camera, music, timer, expression, image) into
callbacks or result flags.

Extracted from the 200-line chunk-parsing loop inside ``BotGUI.main_loop``.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Aggregated result of consuming a ``stream_think()`` generator."""
    take_photo: bool = False
    music_triggered: bool = False
    voice_eq: bool = False
    image_url: str | None = None
    speak_chunks: list[str] = field(default_factory=list)


def dispatch_stream(
    brain,
    user_text: str,
    *,
    on_expression=None,
    on_timer=None,
    on_music=None,
    valid_expressions=None,
) -> ActionResult:
    """Consume ``brain.stream_think(user_text)`` and return an :class:`ActionResult`.

    Parameters
    ----------
    brain : Brain
        The ``core.llm.Brain`` instance.
    user_text : str
        User's input text.
    on_expression : callable(str) or None
        Called when a ``set_expression`` action is detected (value = emotion string).
    on_timer : callable(float, str) or None
        Called when a ``set_timer`` action is detected (minutes, message).
    on_music : callable() or None
        Called when a ``play_music`` action is detected.
    valid_expressions : set[str] or None
        Set of valid expression/emotion strings.  If None, all are accepted.
    """
    result = ActionResult()
    if valid_expressions is None:
        valid_expressions = set()

    stream_start = time.time()
    for chunk in brain.stream_think(user_text):
        if not chunk.strip():
            continue

        from .log import bmo_print
        bmo_print("AGENT", f"Chunk received: '{chunk[:80]}'")

        # Direct take_photo check
        if '{"action": "take_photo"}' in chunk:
            bmo_print("AGENT", "take_photo action detected!")
            result.take_photo = True
            break

        # Try to parse JSON action from chunk
        json_match = re.search(r'\{.*?\}', chunk, re.DOTALL)
        if json_match:
            bmo_print("AGENT", f"JSON regex matched: '{json_match.group(0)[:80]}'")
            try:
                action_data = json.loads(json_match.group(0))
                action = action_data.get("action")
                bmo_print("AGENT", f"Parsed action: {action or 'unknown'}")

                if action == "display_image" and action_data.get("image_url"):
                    result.image_url = action_data["image_url"]
                    bmo_print("AGENT", f"display_image URL set: {result.image_url[:80]}")
                    chunk = chunk.replace(json_match.group(0), '').strip()

                elif action == "set_expression" and action_data.get("value"):
                    expr = action_data["value"].lower()
                    if expr in valid_expressions and on_expression:
                        on_expression(expr)
                    chunk = chunk.replace(json_match.group(0), '').strip()

                elif action == "set_timer" and action_data.get("minutes") is not None:
                    mins = float(action_data["minutes"])
                    msg = action_data.get("message", "Timer is up!")
                    if on_timer:
                        on_timer(mins, msg)
                    chunk = chunk.replace(json_match.group(0), '').strip()

                elif action == "play_music":
                    result.music_triggered = True
                    if on_music:
                        on_music()
                    chunk = chunk.replace(json_match.group(0), '').strip()

                elif action == "voice_eq":
                    # voice_eq is only valid from pre-LLM keyword detection
                    # (which short-circuits before the LLM runs and yields
                    # exactly one chunk).  If we've already seen other chunks,
                    # the LLM is hallucinating — just strip the JSON silently.
                    if not result.speak_chunks:
                        result.voice_eq = True
                    chunk = chunk.replace(json_match.group(0), '').strip()

            except Exception as e:
                bmo_print("AGENT", f"JSON Parse Error: {e} for: '{json_match.group(0)[:50]}'")

        if chunk.strip():
            result.speak_chunks.append(chunk)

    ttlt = time.time() - stream_start
    from .log import bmo_print
    bmo_print("LLM", f"TTLT: {ttlt:.2f}s ({len(result.speak_chunks)} chunks)")
    return result
