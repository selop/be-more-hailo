"""Brain: conversation history + LLM inference orchestration.

NPU lifecycle lives in ``core.npu``; keyword matching and response
cleaning live in ``core.actions``.  This module ties them together.
"""

import contextlib
import logging
import time

from . import cloud_llm, trace
from .config import CHARACTER, LLM_BACKEND, get_system_prompt, say
from .tts import add_pronunciation
from .search import search_web
from .log import bmo_print
from .actions import (
    detect_pre_llm_action_json,
    needs_web_search,
    inject_search_context,
    strip_prompt_leakage,
    extract_pronunciation,
    strip_pronunciation_tag,
)

# Re-export NPU public API so existing imports keep working
from .npu import (          # noqa: F401
    init_llm,
    is_llm_ready,
    _get_llm,
    _release_llm,
    reload_after_vlm,
    _resolve_hef,
    _vlm_worker,
    _prepare_prompt,
)

logger = logging.getLogger(__name__)

# Keep at most this many messages (plus the system prompt) to avoid
# unbounded memory growth on memory-constrained devices like a Pi.
MAX_HISTORY_MESSAGES = 20

# If the last N assistant responses share this much text, the LLM is stuck.
STUCK_CHECK_COUNT = 2
STUCK_SIMILARITY_LEN = 40

# Phrases that indicate the LLM broke character or entered a refusal loop.
_REFUSAL_PATTERNS = [
    "i can't continue",
    "i cannot continue",
    "i can't disclose",
    "i cannot disclose",
    "i can't assist with",
    "i cannot assist with",
    "as an ai",
    "as a language model",
    "don't ask me to repeat",
]
_name = CHARACTER.name.lower()
_REFUSAL_PATTERNS += [f"please allow {_name} to respond", f"i'm not {_name}", f"i am not {_name}"]


def _reply_tokens(history):
    """The reply to *history*, piece by piece: from the hosted model when
    ``LLM_BACKEND=cloud``, otherwise from the local one. If the hosted model
    fails before saying anything (no network, no key, a refusal), the local
    model answers instead."""
    if LLM_BACKEND == "cloud":
        said_something = False
        try:
            for piece in cloud_llm.stream(history):
                said_something = True
                yield piece
            if said_something:
                trace.set("llm_backend", "cloud")
                return
            raise RuntimeError("empty reply")
        except Exception as e:  # noqa: BLE001 - any cloud failure means: answer locally
            if said_something:
                raise
            bmo_print("LLM", f"Cloud model failed ({e!r}); using the local model")
            trace.set("cloud_error", repr(e))

    trace.set("llm_backend", "local")
    logger.info("Stream request to Hailo LLM (direct NPU)")
    with _get_llm().generate(prompt=_prepare_prompt(history), temperature=0.4, max_generated_tokens=100) as gen:
        yield from gen


class Brain:
    def __init__(self):
        self.history = [{"role": "system", "content": get_system_prompt()}]

    def _trim_history(self):
        """Keep the system prompt + the most recent MAX_HISTORY_MESSAGES messages."""
        non_system = self.history[1:]
        if len(non_system) > MAX_HISTORY_MESSAGES:
            self.history = [self.history[0]] + non_system[-MAX_HISTORY_MESSAGES:]

    def _check_stuck(self, latest_response: str) -> bool:
        """Detect if the LLM is stuck in a refusal/repetition loop.
        Returns True if history was reset."""
        lower = latest_response.lower()

        # Check for known refusal patterns
        is_refusal = any(pat in lower for pat in _REFUSAL_PATTERNS)

        # Check if recent assistant messages are near-identical
        recent = [m["content"] for m in self.history
                  if m["role"] == "assistant"][-STUCK_CHECK_COUNT:]
        is_repetition = (
            len(recent) >= STUCK_CHECK_COUNT
            and all(
                r[:STUCK_SIMILARITY_LEN] == recent[0][:STUCK_SIMILARITY_LEN]
                for r in recent
            )
        )

        if is_refusal or is_repetition:
            reason = "refusal" if is_refusal else "repetition"
            bmo_print("BRAIN", f"Stuck detected ({reason}), resetting history")
            self.history = [{"role": "system", "content": get_system_prompt()}]
            return True
        return False

    def think(self, user_text: str) -> str:
        """The whole reply as one string, for callers that do not stream."""
        return " ".join(chunk.strip() for chunk in self.stream_think(user_text))

    # ── Streaming inference ───────────────────────────────────────────────

    def stream_think(self, user_text: str):
        """Send text to Hailo LLM and yield full sentences as they are generated."""
        self.history.append({"role": "user", "content": user_text})

        # Pre-LLM shortcut actions
        action_json = detect_pre_llm_action_json(user_text)
        if action_json is not None:
            bmo_print("LLM-STREAM", f"Pre-LLM action: {action_json[:80]}")
            trace.set("route", "keyword")
            self.history.append({"role": "assistant", "content": action_json})
            yield action_json
            return

        bmo_print("LLM-STREAM", f"No pre-LLM action matched for: '{user_text[:60].lower()}'")
        trace.set("route", "llm")

        # Pre-LLM web search
        search_injected = False
        if needs_web_search(user_text):
            search_start = time.time()
            search_result = None
            try:
                search_result = search_web(user_text)
                if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR") and len(search_result) > 50:
                    self.history[-1]["content"] = inject_search_context(user_text, search_result)
                    search_injected = True
            except Exception as e:
                logger.warning(f"Pre-LLM web search failed: {e}")
            trace.set("search", {
                "ms": int((time.time() - search_start) * 1000),
                "used": search_injected,
                "result": (search_result or "")[:600],
            })
            trace.mark("search_done")

        full_content = ""
        buffer = ""
        sentences_yielded = 0
        tag_seen = False        # the pronunciation tag ends the spoken reply
        MAX_SENTENCES = 4

        try:
            trace.set("prompt_messages", len(self.history) - 1)
            stream_start = time.time()
            first_token_logged = False

            with contextlib.closing(_reply_tokens(self.history)) as gen:
                for token in gen:
                    if not token:
                        continue

                    # Strip stop tokens inline
                    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                        token = token.replace(tok, "")
                    if not token:
                        continue

                    if not first_token_logged:
                        ttft = time.time() - stream_start
                        bmo_print("LLM", f"TTFT: {ttft:.2f}s")
                        trace.set("ttft_ms", int(ttft * 1000))
                        trace.mark("first_token")
                        first_token_logged = True

                    # Replace smart quotes
                    token = token.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")

                    buffer += token
                    full_content += token

                    # If buffer ends with punctuation or newline, yield it
                    # An action's JSON stays in one piece: cut in two it cannot be parsed and gets spoken
                    in_json = buffer.count("{") > buffer.count("}")
                    carry = ""
                    if not in_json and "\n\n" in buffer and not buffer.endswith(('.', '!', '?', '\n')):
                        # A blank line ends the chunk before it; what follows starts the next one
                        buffer, carry = buffer.rsplit("\n\n", 1)
                        buffer += "\n\n"
                    if not in_json and any(buffer.endswith(punc) for punc in ['.', '!', '?', '\n']):
                        cleaned = strip_pronunciation_tag(strip_prompt_leakage(buffer))
                        out_chunk = "" if tag_seen else CHARACTER.normalize_name(cleaned)
                        tag_seen = tag_seen or strip_pronunciation_tag(buffer) != buffer.strip()
                        # Clear before yielding: breaking out at the sentence
                        # limit used to leave the last sentence in the buffer,
                        # and it was then yielded (and spoken) a second time.
                        buffer = carry
                        if any(ch.isalnum() for ch in out_chunk):    # a lone '!' is not a sentence
                            yield out_chunk
                            sentences_yielded += 1
                            if sentences_yielded >= MAX_SENTENCES:
                                break

            # Yield any remaining buffer
            if buffer.strip():
                cleaned = strip_pronunciation_tag(strip_prompt_leakage(buffer))
                out_chunk = "" if tag_seen else CHARACTER.normalize_name(cleaned)
                if any(ch.isalnum() for ch in out_chunk):
                    yield out_chunk

            self.history.append({"role": "assistant", "content": full_content})
            trace.set("llm_raw", full_content)
            trace.set("llm_ms", int((time.time() - stream_start) * 1000))

            # The user corrected a pronunciation: remember it for the voice
            # Only when the user spoke about it: a stray or parroted tag must not rewrite speech for good
            pron = extract_pronunciation(full_content)
            if pron and "pronounc" in user_text.lower():
                word, phonetic = pron
                logger.info(f"Learned new pronunciation from LLM: {word} -> {phonetic}")
                add_pronunciation(word, phonetic)
                trace.set("pronunciation", f"{word}={phonetic}")

            # Detect stuck LLM (refusal loops / repetition) — the bad
            # response was already yielded, but we reset history so the
            # NEXT interaction starts clean.
            if self._check_stuck(full_content):
                trace.set("history_reset", True)

            # Clean injected search context from history
            if search_injected:
                for msg in reversed(self.history):
                    if msg.get("role") == "user" and msg.get("content", "").startswith("[LIVE DATA:"):
                        msg["content"] = user_text
                        break

            self._trim_history()

        except Exception as e:
            logger.error(f"Brain Exception: {e}", exc_info=True)
            trace.set("llm_error", repr(e))
            yield say("stream_error")

    def set_history(self, new_history):
        if not new_history or new_history[0].get("role") != "system":
            new_history.insert(0, {"role": "system", "content": get_system_prompt()})
        else:
            new_history[0]["content"] = get_system_prompt()
        self.history = new_history

    # ── VLM (vision) ─────────────────────────────────────────────────────

    def analyze_image(self, image_base64: str, user_text: str) -> str:
        """Analyse an image using the Hailo VLM in a child process."""
        import multiprocessing as mp
        from .config import VLM_HEF_PATH

        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        self.history.append({"role": "user", "content": user_text})

        try:
            vlm_hef = _resolve_hef(VLM_HEF_PATH)
        except FileNotFoundError as e:
            logger.warning(f"VLM HEF not found: {e}")
            return say("vision_missing")

        _release_llm()

        try:
            result_queue = mp.Queue()
            proc = mp.Process(
                target=_vlm_worker,
                args=(image_base64, user_text, vlm_hef, result_queue),
            )
            logger.info("Starting VLM subprocess ...")
            proc.start()
            proc.join(timeout=90)

            if proc.is_alive():
                logger.warning("VLM subprocess timed out, killing it")
                proc.kill()
                proc.join(timeout=5)
                return say("vision_timeout")

            if proc.exitcode != 0:
                logger.error(f"VLM subprocess exited with code {proc.exitcode}")
                return say("vision_error")

            try:
                content = result_queue.get_nowait()
            except Exception:
                return say("vision_error")

            if content.startswith("ERROR:"):
                logger.error(f"VLM subprocess error: {content}")
                return say("vision_error")

            logger.info(f"VLM response ({len(content)} chars): {content[:120]}...")
            self.history.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            logger.error(f"VLM Exception: {e}", exc_info=True)
            return say("vision_error")
