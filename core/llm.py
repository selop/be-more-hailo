import base64
import os
import time
import logging
import re
import json
import urllib.parse
import numpy as np
from .config import LLM_HEF_PATH, VLM_HEF_PATH, get_system_prompt
from .tts import add_pronunciation
from .search import search_web
from .log import bmo_print

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  VDevice + LLM + STT singletons (main process — shared VDevice)
#  VLM runs in a child process with its own VDevice since the NPU can't hold
#  both large generative models (LLM 1.5B + VLM 2B) simultaneously.
#  Whisper (125MB) is small enough to coexist with LLM on the shared VDevice.
# --------------------------------------------------------------------------- #
_vdevice = None
_llm_instance = None
_system_context = None  # Cached KV state for the system prompt


def _resolve_hef(hef_path: str) -> str:
    """Resolve a HEF path, making relative paths relative to the project root."""
    if not os.path.isabs(hef_path):
        hef_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), hef_path)
    if not os.path.exists(hef_path):
        raise FileNotFoundError(f"HEF not found at {hef_path}")
    return hef_path


def _get_vdevice():
    """Return the shared VDevice singleton, creating it on first call.
    Uses group_id='SHARED' so LLM + Speech2Text can coexist on the NPU."""
    global _vdevice
    if _vdevice is not None:
        return _vdevice

    from hailo_platform import VDevice

    params = VDevice.create_params()
    params.group_id = "SHARED"
    _vdevice = VDevice(params)
    logger.info("VDevice initialised (group_id=SHARED)")
    return _vdevice


def _get_llm():
    """Return the LLM singleton, initialising on first call."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    from hailo_platform.genai import LLM

    hef = _resolve_hef(LLM_HEF_PATH)
    vdevice = _get_vdevice()

    logger.info(f"Initialising Hailo LLM from {hef} ...")
    _llm_instance = LLM(vdevice, hef)
    logger.info("LLM ready")
    return _llm_instance


def _release_llm():
    """Release STT + LLM + VDevice to free the NPU for VLM subprocess."""
    global _llm_instance, _vdevice
    # Release STT first (it shares the VDevice)
    from .stt import release_stt
    release_stt()
    if _llm_instance is not None:
        try:
            _llm_instance.release()
        except Exception as e:
            logger.warning(f"Error releasing LLM: {e}")
        _llm_instance = None
    if _vdevice is not None:
        try:
            _vdevice.release()
        except Exception as e:
            logger.warning(f"Error releasing VDevice: {e}")
        _vdevice = None
    logger.info("STT + LLM + VDevice released for VLM subprocess")


def init_llm():
    """Eagerly initialise VDevice + LLM at startup, and cache the system prompt
    KV state so it doesn't need to be re-processed on every turn."""
    global _system_context
    llm = _get_llm()

    if _system_context is None:
        # Prime the KV cache with the system prompt
        system_prompt = [{"role": "system", "content": get_system_prompt()}]
        logger.info("Caching system prompt context ...")
        try:
            llm.generate_all(prompt=system_prompt, max_generated_tokens=1)
            _system_context = llm.save_context()
            logger.info("System prompt context cached")
        except Exception as e:
            logger.warning(f"Failed to cache system prompt context: {e}")
            _system_context = None


def is_llm_ready() -> bool:
    """Return True if the LLM singleton has been initialised."""
    return _llm_instance is not None


def _prepare_prompt(history: list) -> list:
    """Load cached system prompt context if available, and return the
    prompt messages (without the system message, since it's in the KV cache)."""
    if _system_context is not None and _llm_instance is not None:
        try:
            _llm_instance.load_context(_system_context)
            # Skip the system message — it's already baked into the KV cache
            return [msg for msg in history if msg.get("role") != "system"]
        except Exception as e:
            logger.warning(f"Failed to load cached context, using full prompt: {e}")
    return history


# --------------------------------------------------------------------------- #
#  Hailo VLM (Vision Language Model) — runs in a child process
# --------------------------------------------------------------------------- #
# The Hailo-10H NPU cannot hold LLM (2.3GB) + VLM (2.3GB) simultaneously,
# and in-process .release() doesn't fully free NPU driver state.  The proven
# pattern (from hailo-apps vlm_chat) is to run VLM in a separate process —
# process exit guarantees clean NPU release.
#
# Flow: release LLM → fork child → child creates VDevice + VLM → inference →
#       child exits (NPU freed) → parent reloads LLM.

_CAMERA_TRIGGER_PHRASES = [
    "take a photo", "take a picture", "take photo", "take picture",
    "look at", "what do you see", "what can you see", "use your camera",
    "photograph", "snap a photo",
]

def _vlm_question(user_text: str) -> str:
    """Convert camera-trigger phrases into an actual VLM question."""
    if not user_text:
        return "What do you see in this image? Describe it briefly."
    lower = user_text.lower()
    # If the text is just a camera trigger (or dominated by one), ask for a description
    if any(lower.strip().rstrip(".!?") == p for p in _CAMERA_TRIGGER_PHRASES) or \
       any(lower.startswith(p) for p in ["take a photo", "take a picture", "take photo",
                                          "take picture", "snap a photo", "photograph"]):
        return "What do you see in this image? Describe it briefly."
    return user_text


def _vlm_worker(image_b64: str, user_text: str, vlm_hef: str, result_queue):
    """Run VLM inference in a child process with its own VDevice.
    Puts the result string into result_queue."""
    import cv2
    try:
        from hailo_platform import VDevice
        from hailo_platform.genai import VLM

        vdevice = VDevice()
        vlm = VLM(vdevice, vlm_hef)

        shape = vlm.input_frame_shape()
        dtype = vlm.input_frame_format_type()

        # Decode image
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            result_queue.put("ERROR:Failed to decode image")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target_h, target_w = shape[0], shape[1]
        src_h, src_w = img.shape[:2]
        if src_h != target_h or src_w != target_w:
            scale = max(target_w / src_w, target_h / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            x_start = (new_w - target_w) // 2
            y_start = (new_h - target_h) // 2
            img = img[y_start:y_start + target_h, x_start:x_start + target_w]
        frame = img.astype(dtype)

        prompt = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are BMO, a helpful robot assistant. Describe what you see concisely and conversationally in English."}
            ]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": _vlm_question(user_text)}
            ]}
        ]

        vlm.clear_context()
        content = vlm.generate_all(
            prompt=prompt, frames=[frame],
            max_generated_tokens=200, temperature=0.4, seed=42,
        )

        # Clean up stop tokens
        for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            content = content.replace(tok, "")
        content = content.replace('\u201c', '"').replace('\u201d', '"')
        content = content.replace('\u2018', "'").replace('\u2019', "'")
        content = content.strip()

        vlm.clear_context()
        vlm.release()
        vdevice.release()

        result_queue.put(content)
    except Exception as e:
        result_queue.put(f"ERROR:{e}")


# Keep at most this many messages (plus the system prompt) to avoid
# unbounded memory growth on memory-constrained devices like a Pi.
MAX_HISTORY_MESSAGES = 20

_DISPLAY_IMAGE_KEYWORDS = [
    "show me a picture", "show me an image", "show me a photo",
    "show a picture", "show an image", "show a photo",
    "display a picture", "display an image", "display a photo",
    "picture of", "image of", "photo of",
    "generate an image", "generate a picture",
    "draw me", "draw a",
]

_MUSIC_KEYWORDS = [
    "play music", "play a song", "play me a song", "play some music",
    "sing a song", "sing me a song", "sing for me", "sing something",
    "play a tune", "play me a tune", "play some tunes",
    "can you sing", "will you sing", "do you sing",
    "play your music", "jam out", "dance for me",
    "sing for bmo", "bmo sing", "play me some music",
]

# Phrases that indicate the model is regurgitating system prompt instructions
_PROMPT_LEAK_PATTERNS = [
    re.compile(r'CRIT(?:IC)?AL:.*?JSON', re.IGNORECASE | re.DOTALL),
    re.compile(r'you MUST output.*?JSON', re.IGNORECASE | re.DOTALL),
    re.compile(r'output (?:exactly )?this JSON', re.IGNORECASE),
    re.compile(r'If the user asks.*?(?:output|JSON)', re.IGNORECASE | re.DOTALL),
    re.compile(r'Replace YOUR_PROMPT_HERE', re.IGNORECASE),
    re.compile(r'Valid emotions are:', re.IGNORECASE),
]


def _build_display_image_action(user_text: str) -> str:
    """Extract the subject from user text and return a display_image JSON action."""
    # Strip common prefixes to get the image subject
    subject = user_text
    for prefix in ["show me a picture of", "show me an image of", "show me a photo of",
                    "show a picture of", "show an image of", "show a photo of",
                    "display a picture of", "display an image of", "display a photo of",
                    "generate an image of", "generate a picture of",
                    "draw me a", "draw me an", "draw me", "draw a", "draw an",
                    "picture of", "image of", "photo of"]:
        lower = subject.lower()
        if lower.startswith(prefix):
            subject = subject[len(prefix):].strip()
            break
    # Remove trailing punctuation
    subject = subject.rstrip("?.!")
    prompt = urllib.parse.quote(subject.strip() or "something fun")
    return json.dumps({"action": "display_image", "image_url": f"https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true"})


def _strip_prompt_leakage(content: str) -> str:
    """Remove text that looks like the model echoing system prompt instructions."""
    for pat in _PROMPT_LEAK_PATTERNS:
        # Remove the matched pattern and everything after it (it's usually trailing noise)
        match = pat.search(content)
        if match:
            content = content[:match.start()].strip()
    return content



class Brain:
    def __init__(self):
        self.history = [{"role": "system", "content": get_system_prompt()}]

    def _trim_history(self):
        """Keep the system prompt + the most recent MAX_HISTORY_MESSAGES messages."""
        # history[0] is always the system prompt
        non_system = self.history[1:]
        if len(non_system) > MAX_HISTORY_MESSAGES:
            self.history = [self.history[0]] + non_system[-MAX_HISTORY_MESSAGES:]

    def think(self, user_text: str) -> str:
        """
        Send text to Hailo LLM (direct NPU API) and get response.
        """
        self.history.append({"role": "user", "content": user_text})

        lower_text = user_text.lower()

        # Pre-LLM camera check — same logic as stream_think
        camera_keywords = [
            "take a photo", "take a picture", "take photo", "take picture",
            "look at", "what do you see", "what can you see", "use your camera",
            "photograph", "snap a photo",
        ]
        if any(kw in lower_text for kw in camera_keywords):
            action = '{"action": "take_photo"}'
            self.history.append({"role": "assistant", "content": action})
            return action

        # Pre-LLM display_image check — handle image generation requests
        # directly instead of relying on the small model to emit correct JSON
        if any(kw in lower_text for kw in _DISPLAY_IMAGE_KEYWORDS):
            action = _build_display_image_action(user_text)
            matched_kw = next(kw for kw in _DISPLAY_IMAGE_KEYWORDS if kw in lower_text)
            bmo_print("LLM", f"Image keyword MATCHED: '{matched_kw}' in '{lower_text[:60]}'")
            bmo_print("LLM", f"Emitting display_image action: {action[:80]}")
            self.history.append({"role": "assistant", "content": action})
            return action

        # Pre-LLM music check — emit play_music directly rather than
        # relying on the small model to emit correct JSON
        if any(kw in lower_text for kw in _MUSIC_KEYWORDS):
            action = '{"action": "play_music"}'
            matched_kw = next(kw for kw in _MUSIC_KEYWORDS if kw in lower_text)
            bmo_print("LLM", f"Music keyword MATCHED: '{matched_kw}' in '{lower_text[:60]}'")
            bmo_print("LLM", "Emitting play_music action")
            self.history.append({"role": "assistant", "content": action})
            return action

        bmo_print("LLM", f"No pre-LLM action matched for: '{lower_text[:60]}'")

        # Pre-LLM web search — same logic as stream_think
        realtime_keywords = [
            "weather", "forecast", "temperature", "tonight", "tomorrow",
            "news", "latest", "right now", "score", "stocks", "bitcoin",
            "crypto", "price of", "happening", "recently", "live",
        ]
        question_markers = [
            "what", "who", "when", "where", "find", "search", "tell me",
            "look up", "check", "is there", "did", "?",
        ]
        has_realtime_kw = any(kw in lower_text for kw in realtime_keywords)
        has_question = any(q in lower_text for q in question_markers)
        search_injected = False
        if has_realtime_kw and has_question:
            try:
                search_result = search_web(user_text)
                if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR") and len(search_result) > 50:
                    # Strip the verbose "SEARCH RESULTS for '...':" header from search.py
                    clean_result = re.sub(r"^SEARCH RESULTS for '.*?':\n?", "", search_result).strip()
                    # Inject as a tight [LIVE DATA] block — clearer than the previous format
                    self.history[-1]["content"] = (
                        f"[LIVE DATA: {clean_result}] "
                        f"Using only the above live data, answer in one or two sentences as BMO: {user_text}"
                    )
                    search_injected = True
            except Exception as e:
                logger.warning(f"Pre-LLM web search failed: {e}")

        try:
            llm = _get_llm()
            prompt = _prepare_prompt(self.history)
            logger.info("Sending request to Hailo LLM (direct NPU)")
            content = llm.generate_all(prompt=prompt, temperature=0.4, max_generated_tokens=100)

            # Strip stop tokens the model may emit
            for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                content = content.replace(tok, "")
            content = content.strip()

            # Check if the LLM outputted a JSON action (like search_web)
            try:
                # Replace smart quotes with standard quotes before parsing
                clean_content = content.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
                json_match = re.search(r'\{.*?\}', clean_content, re.DOTALL)
                if json_match:
                    action_data = json.loads(json_match.group(0))

                    if action_data.get("action") == "take_photo":
                        logger.info("LLM requested to take a photo.")
                        return json.dumps({"action": "take_photo"})

                    elif action_data.get("action") == "search_web":
                        query = action_data.get("query", "")
                        logger.info(f"LLM requested web search for: {query}")

                        # Perform the search
                        search_result = search_web(query)

                        # Feed the result back to the LLM to summarize
                        summary_messages = [
                            {"role": "system", "content": "Summarize this search result in one short, conversational sentence as BMO. Do not use markdown."},
                            {"role": "user", "content": f"RESULT: {search_result}\nUser Question: {user_text}"}
                        ]
                        content = llm.generate_all(prompt=summary_messages, temperature=0.4, max_generated_tokens=100)
                        for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                            content = content.replace(tok, "")
                        content = content.strip()
            except json.JSONDecodeError:
                pass  # Not valid JSON, just treat as normal text

            # Check for pronunciation learning tag
            pronounce_match = re.search(r'!PRONOUNCE:\s*([a-zA-Z0-9_-]+)\s*=\s*([a-zA-Z0-9_-]+)', content, re.IGNORECASE)
            if pronounce_match:
                word = pronounce_match.group(1).strip()
                phonetic = pronounce_match.group(2).strip()
                logger.info(f"Learned new pronunciation from LLM: {word} -> {phonetic}")
                add_pronunciation(word, phonetic)
                # Remove the tag from the spoken content
                content = re.sub(r'!PRONOUNCE:.*', '', content, flags=re.IGNORECASE).strip()

            # Strip any system prompt leakage from the response
            content = _strip_prompt_leakage(content)

            # Ensure BMO is spelled correctly in text responses
            content = re.sub(r'\bBeemo\b', 'BMO', content, flags=re.IGNORECASE)

            # Fallback if filtering left nothing useful
            if not content.strip():
                content = "BMO is here! How can I help?"

            self.history.append({"role": "assistant", "content": content})

            # Clean injected search context from history so it doesn't
            # accumulate and confuse the model on future turns.
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

    def get_history(self):
        return self.history

    def stream_think(self, user_text: str):
        """
        Send text to Hailo LLM and yield full sentences as they are generated.
        Useful for TTS chunking (speaking while generating).
        """
        self.history.append({"role": "user", "content": user_text})

        lower_text = user_text.lower()

        # Pre-LLM camera check: if user asks to take a photo / look at something,
        # emit the action JSON directly without calling the LLM.
        camera_keywords = [
            "take a photo", "take a picture", "take photo", "take picture",
            "look at", "what do you see", "what can you see", "use your camera",
            "photograph", "snap a photo",
        ]
        if any(kw in lower_text for kw in camera_keywords):
            action = '{"action": "take_photo"}'
            self.history.append({"role": "assistant", "content": action})
            yield action
            return

        # Pre-LLM display_image check
        if any(kw in lower_text for kw in _DISPLAY_IMAGE_KEYWORDS):
            action = _build_display_image_action(user_text)
            matched_kw = next(kw for kw in _DISPLAY_IMAGE_KEYWORDS if kw in lower_text)
            bmo_print("LLM-STREAM", f"Image keyword MATCHED: '{matched_kw}' in '{lower_text[:60]}'")
            self.history.append({"role": "assistant", "content": action})
            yield action
            return

        # Pre-LLM music check — emit play_music directly
        if any(kw in lower_text for kw in _MUSIC_KEYWORDS):
            action = '{"action": "play_music"}'
            matched_kw = next(kw for kw in _MUSIC_KEYWORDS if kw in lower_text)
            bmo_print("LLM-STREAM", f"Music keyword MATCHED: '{matched_kw}' in '{lower_text[:60]}'")
            self.history.append({"role": "assistant", "content": action})
            yield action
            return

        bmo_print("LLM-STREAM", f"No pre-LLM action matched for: '{lower_text[:60]}'")

        # Pre-LLM keyword check: if the question likely needs real-time info,
        # do the web search now rather than relying on the model to emit JSON.
        realtime_keywords = [
            "weather", "forecast", "temperature", "tonight", "tomorrow",
            "news", "latest", "right now", "score", "stocks", "bitcoin",
            "crypto", "price of", "happening", "recently", "live",
        ]
        question_markers = [
            "what", "who", "when", "where", "find", "search", "tell me",
            "look up", "check", "is there", "did", "?",
        ]
        has_realtime_kw = any(kw in lower_text for kw in realtime_keywords)
        has_question = any(q in lower_text for q in question_markers)
        needs_search = has_realtime_kw and has_question
        search_injected = False
        if needs_search:
            try:
                search_result = search_web(user_text)
                # Only inject if we got a real result (not empty/error sentinel)
                if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR") and len(search_result) > 50:
                    # Strip verbose "SEARCH RESULTS for '...':" prefix from search.py
                    clean_result = re.sub(r"^SEARCH RESULTS for '.*?':\n?", "", search_result).strip()
                    self.history[-1]["content"] = (
                        f"[LIVE DATA: {clean_result}] "
                        f"Using only the above live data, answer in one or two sentences as BMO: {user_text}"
                    )
                    search_injected = True
            except Exception as e:
                logger.warning(f"Pre-LLM web search failed: {e}")

        full_content = ""
        buffer = ""
        sentences_yielded = 0
        MAX_SENTENCES = 4  # Cap to keep BMO concise (system prompt says 2-4)

        try:
            llm = _get_llm()
            prompt = _prepare_prompt(self.history)
            logger.info("Stream request to Hailo LLM (direct NPU)")

            with llm.generate(prompt=prompt, temperature=0.4, max_generated_tokens=100) as gen:
                for token in gen:
                    if not token:
                        continue

                    # Strip stop tokens inline
                    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                        token = token.replace(tok, "")
                    if not token:
                        continue

                    # Replace smart quotes
                    token = token.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")

                    buffer += token
                    full_content += token

                    # If buffer ends with punctuation or newline, yield it
                    if any(buffer.endswith(punc) for punc in ['.', '!', '?', '\n']) or "\n\n" in buffer:
                        # Strip system prompt leakage
                        cleaned = _strip_prompt_leakage(buffer)
                        # Ensure BMO spelling before yielding
                        out_chunk = re.sub(r'\bBeemo\b', 'BMO', cleaned, flags=re.IGNORECASE)
                        if out_chunk.strip():
                            yield out_chunk
                            sentences_yielded += 1
                            if sentences_yielded >= MAX_SENTENCES:
                                break
                        buffer = ""

            # Yield any remaining buffer
            if buffer.strip():
                cleaned = _strip_prompt_leakage(buffer)
                out_chunk = re.sub(r'\bBeemo\b', 'BMO', cleaned, flags=re.IGNORECASE)
                if out_chunk.strip():
                    yield out_chunk

            # Handle json actions at the very end if applicable
            json_match = re.search(r'\{.*?\}', full_content, re.DOTALL)
            if json_match and "action" in json_match.group(0):
                # For advanced tool use we won't yield the json action to TTS
                pass

            self.history.append({"role": "assistant", "content": full_content})

            # Clean injected search context from history so it doesn't
            # accumulate and confuse the model on future turns.
            if search_injected:
                for msg in reversed(self.history):
                    if msg.get("role") == "user" and msg.get("content", "").startswith("[LIVE DATA:"):
                        msg["content"] = user_text
                        break

            self._trim_history()

        except Exception as e:
            logger.error(f"Brain Exception: {e}", exc_info=True)
            yield "I'm having trouble right now."

    def set_history(self, new_history):
        # Ensure system prompt is always present and up to date
        if not new_history or new_history[0].get("role") != "system":
            new_history.insert(0, {"role": "system", "content": get_system_prompt()})
        else:
            new_history[0]["content"] = get_system_prompt()
        self.history = new_history

    def analyze_image(self, image_base64: str, user_text: str) -> str:
        """
        Analyse an image using the Hailo VLM (Qwen2-VL-2B) in a child process.
        The NPU can only hold one large model at a time, so we release the LLM,
        fork a child for VLM inference, then reload the LLM when it exits.
        """
        import multiprocessing as mp

        # Strip data URI prefix if present (browser sends "data:image/jpeg;base64,...")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        self.history.append({"role": "user", "content": user_text})

        try:
            vlm_hef = _resolve_hef(VLM_HEF_PATH)
        except FileNotFoundError as e:
            logger.warning(f"VLM HEF not found: {e}")
            return "BMO's vision model isn't installed yet. Run setup.sh to download it!"

        # Release LLM + VDevice so the child process can claim the NPU
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
        finally:
            # Reload LLM + STT + re-cache system prompt so inference is ready again
            global _system_context
            logger.info("Reloading LLM + STT after VLM subprocess ...")
            _system_context = None
            init_llm()
            from .stt import init_stt
            init_stt()
