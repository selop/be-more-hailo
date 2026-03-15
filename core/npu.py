"""NPU device lifecycle: VDevice, LLM, and VLM management.

Isolates all Hailo hardware interaction from conversation logic so that
``core.llm.Brain`` only deals with history and inference orchestration.
"""

import base64
import logging
import os

import numpy as np

from .config import LLM_HEF_PATH, VLM_HEF_PATH, get_system_prompt
from .actions import vlm_question, clean_llm_response

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────
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
            return [msg for msg in history if msg.get("role") != "system"]
        except Exception as e:
            logger.warning(f"Failed to load cached context, using full prompt: {e}")
    return history


def reload_after_vlm():
    """Reload LLM + STT after VLM subprocess. Can be called in a background thread
    so TTS can speak the VLM response while models reload."""
    global _system_context
    logger.info("Reloading LLM + STT after VLM subprocess ...")
    _system_context = None
    init_llm()
    from .stt import init_stt
    init_stt()
    logger.info("LLM + STT reload complete")


# ── VLM subprocess worker ────────────────────────────────────────────────

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
                {"type": "text", "text": vlm_question(user_text)}
            ]}
        ]

        vlm.clear_context()
        content = vlm.generate_all(
            prompt=prompt, frames=[frame],
            max_generated_tokens=200, temperature=0.4, seed=42,
        )

        content = clean_llm_response(content)

        vlm.clear_context()
        vlm.release()
        vdevice.release()

        result_queue.put(content)
    except Exception as e:
        result_queue.put(f"ERROR:{e}")
