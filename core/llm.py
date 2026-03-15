"""Brain: conversation history + LLM inference orchestration.

NPU lifecycle lives in ``core.npu``; keyword matching and response
cleaning live in ``core.actions``.  This module ties them together.
"""

import json
import logging
import re
import time

from .config import get_system_prompt
from .tts import add_pronunciation
from .search import search_web
from .log import bmo_print
from .actions import (
    detect_pre_llm_action_json,
    needs_web_search,
    inject_search_context,
    clean_llm_response,
    strip_prompt_leakage,
    extract_json_action,
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


class Brain:
    def __init__(self):
        self.history = [{"role": "system", "content": get_system_prompt()}]

    def _trim_history(self):
        """Keep the system prompt + the most recent MAX_HISTORY_MESSAGES messages."""
        non_system = self.history[1:]
        if len(non_system) > MAX_HISTORY_MESSAGES:
            self.history = [self.history[0]] + non_system[-MAX_HISTORY_MESSAGES:]

    # ── Non-streaming inference ───────────────────────────────────────────

    def think(self, user_text: str) -> str:
        """Send text to Hailo LLM (direct NPU API) and get response."""
        self.history.append({"role": "user", "content": user_text})

        # Pre-LLM shortcut actions (camera / image / music)
        action_json = detect_pre_llm_action_json(user_text)
        if action_json is not None:
            bmo_print("LLM", f"Pre-LLM action: {action_json[:80]}")
            self.history.append({"role": "assistant", "content": action_json})
            return action_json

        bmo_print("LLM", f"No pre-LLM action matched for: '{user_text[:60].lower()}'")

        # Pre-LLM web search
        search_injected = False
        if needs_web_search(user_text):
            try:
                search_result = search_web(user_text)
                if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR") and len(search_result) > 50:
                    self.history[-1]["content"] = inject_search_context(user_text, search_result)
                    search_injected = True
            except Exception as e:
                logger.warning(f"Pre-LLM web search failed: {e}")

        try:
            llm = _get_llm()
            prompt = _prepare_prompt(self.history)
            logger.info("Sending request to Hailo LLM (direct NPU)")
            content = llm.generate_all(prompt=prompt, temperature=0.4, max_generated_tokens=100)

            # Strip stop tokens
            for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                content = content.replace(tok, "")
            content = content.strip()

            # Check if the LLM outputted a JSON action (like search_web)
            action_data = extract_json_action(content)
            if action_data:
                if action_data.get("action") == "take_photo":
                    logger.info("LLM requested to take a photo.")
                    return json.dumps({"action": "take_photo"})

                elif action_data.get("action") == "search_web":
                    query = action_data.get("query", "")
                    logger.info(f"LLM requested web search for: {query}")
                    search_result = search_web(query)
                    summary_messages = [
                        {"role": "system", "content": "Summarize this search result in one short, conversational sentence as BMO. Do not use markdown."},
                        {"role": "user", "content": f"RESULT: {search_result}\nUser Question: {user_text}"}
                    ]
                    content = llm.generate_all(prompt=summary_messages, temperature=0.4, max_generated_tokens=100)
                    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                        content = content.replace(tok, "")
                    content = content.strip()

            # Check for pronunciation learning tag
            pron = extract_pronunciation(content)
            if pron:
                word, phonetic = pron
                logger.info(f"Learned new pronunciation from LLM: {word} -> {phonetic}")
                add_pronunciation(word, phonetic)
                content = strip_pronunciation_tag(content)

            content = clean_llm_response(content)

            self.history.append({"role": "assistant", "content": content})

            # Clean injected search context from history
            if search_injected:
                for msg in reversed(self.history):
                    if msg.get("role") == "user" and msg.get("content", "").startswith("[LIVE DATA:"):
                        msg["content"] = user_text
                        break

            self._trim_history()
            return content

        except Exception as e:
            logger.error(f"Brain Exception: {e}", exc_info=True)
            return "I'm having trouble thinking right now."

    # ── Streaming inference ───────────────────────────────────────────────

    def stream_think(self, user_text: str):
        """Send text to Hailo LLM and yield full sentences as they are generated."""
        self.history.append({"role": "user", "content": user_text})

        # Pre-LLM shortcut actions
        action_json = detect_pre_llm_action_json(user_text)
        if action_json is not None:
            bmo_print("LLM-STREAM", f"Pre-LLM action: {action_json[:80]}")
            self.history.append({"role": "assistant", "content": action_json})
            yield action_json
            return

        bmo_print("LLM-STREAM", f"No pre-LLM action matched for: '{user_text[:60].lower()}'")

        # Pre-LLM web search
        search_injected = False
        if needs_web_search(user_text):
            try:
                search_result = search_web(user_text)
                if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR") and len(search_result) > 50:
                    self.history[-1]["content"] = inject_search_context(user_text, search_result)
                    search_injected = True
            except Exception as e:
                logger.warning(f"Pre-LLM web search failed: {e}")

        full_content = ""
        buffer = ""
        sentences_yielded = 0
        MAX_SENTENCES = 4

        try:
            llm = _get_llm()
            prompt = _prepare_prompt(self.history)
            logger.info("Stream request to Hailo LLM (direct NPU)")
            stream_start = time.time()
            first_token_logged = False

            with llm.generate(prompt=prompt, temperature=0.4, max_generated_tokens=100) as gen:
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
                        first_token_logged = True

                    # Replace smart quotes
                    token = token.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")

                    buffer += token
                    full_content += token

                    # If buffer ends with punctuation or newline, yield it
                    if any(buffer.endswith(punc) for punc in ['.', '!', '?', '\n']) or "\n\n" in buffer:
                        cleaned = strip_prompt_leakage(buffer)
                        out_chunk = re.sub(r'\bBeemo\b', 'BMO', cleaned, flags=re.IGNORECASE)
                        if out_chunk.strip():
                            yield out_chunk
                            sentences_yielded += 1
                            if sentences_yielded >= MAX_SENTENCES:
                                break
                        buffer = ""

            # Yield any remaining buffer
            if buffer.strip():
                cleaned = strip_prompt_leakage(buffer)
                out_chunk = re.sub(r'\bBeemo\b', 'BMO', cleaned, flags=re.IGNORECASE)
                if out_chunk.strip():
                    yield out_chunk

            self.history.append({"role": "assistant", "content": full_content})

            # Clean injected search context from history
            if search_injected:
                for msg in reversed(self.history):
                    if msg.get("role") == "user" and msg.get("content", "").startswith("[LIVE DATA:"):
                        msg["content"] = user_text
                        break

            self._trim_history()

        except Exception as e:
            logger.error(f"Brain Exception: {e}", exc_info=True)
            yield "I'm having trouble right now."

    # ── History management ────────────────────────────────────────────────

    def get_history(self):
        return self.history

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
            return "BMO's vision model isn't installed yet. Run setup.sh to download it!"

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
                return "BMO looked too long and got dizzy! Try again?"

            if proc.exitcode != 0:
                logger.error(f"VLM subprocess exited with code {proc.exitcode}")
                return "I tried to look, but my eyes aren't working right now."

            try:
                content = result_queue.get_nowait()
            except Exception:
                return "I tried to look, but my eyes aren't working right now."

            if content.startswith("ERROR:"):
                logger.error(f"VLM subprocess error: {content}")
                return "I tried to look, but my eyes aren't working right now."

            logger.info(f"VLM response ({len(content)} chars): {content[:120]}...")
            self.history.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            logger.error(f"VLM Exception: {e}", exc_info=True)
            return "I tried to look, but my eyes aren't working right now."
