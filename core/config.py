import datetime
import os
from dotenv import load_dotenv

from .character import get_character

load_dotenv()

# Shared configuration.
#
# Everything character-specific (name, prompts, spoken lines, voice, faces,
# theme) comes from the active character pack under characters/<id>/ — see
# core/character.py. Select a pack with the CHARACTER environment variable.
CHARACTER = get_character()

# Language — switch the voice and LLM output language
# Supported: "en" (English, default), "de" (German)
LANGUAGE = os.environ.get("BMO_LANGUAGE", "en")

# LLM HEF — direct NPU inference via hailo_platform.genai.LLM (no hailo-ollama needed)
LLM_HEF_PATH = os.environ.get("LLM_HEF_PATH", "./models/Qwen2.5-1.5B-Instruct.hef")

# VLM (Vision Language Model) Settings — uses HailoRT Python API directly
# The HEF file is a precompiled model binary from Hailo's model zoo
VLM_HEF_PATH = os.environ.get("VLM_HEF_PATH", "./models/Qwen2-VL-2B-Instruct.hef")

_LANGUAGE_RULES = {
    "de": "You MUST respond ONLY in German (Deutsch) at all times. Never use Chinese characters or English, regardless of the prompt. ",
    "en": "You MUST respond ONLY in English at all times. Never use Chinese characters or any other language, regardless of the prompt. ",
}


def get_system_prompt():
    """Build the system prompt from the character pack with the current
    time, date and language rule filled in."""
    now = datetime.datetime.now()
    return CHARACTER.prompt(
        "system",
        time=now.strftime("%I:%M %p"),
        date=now.strftime("%A, %B %d, %Y"),
        language_rule=_LANGUAGE_RULES["de" if LANGUAGE == "de" else "en"],
    )


SYSTEM_PROMPT = get_system_prompt()

# TTS Settings
PIPER_CMD = "./piper/piper"
_voice = CHARACTER.voice(LANGUAGE)
PIPER_MODEL = _voice["model"]
# Speech rate: >1.0 = slower.
PIPER_LENGTH_SCALE = _voice["length_scale"]

# Low-pass filter for the character's speaker sound. Defaults come from the
# character pack; env vars override them.
_voice_filter = CHARACTER.voice_filter
VOICE_LPF_ENABLED = os.environ.get("VOICE_LPF_ENABLED", "1" if _voice_filter["enabled"] else "0") != "0"
VOICE_LPF_CUTOFF = int(os.environ.get("VOICE_LPF_CUTOFF", str(_voice_filter["cutoff"])))   # Hz (try 4000–8000)
VOICE_LPF_ORDER = int(os.environ.get("VOICE_LPF_ORDER", str(_voice_filter["order"])))      # Butterworth order (2=gentle, 6=steep)

PRONUNCIATION_FILE = CHARACTER.asset("pronunciations")

# Asset folder of the active character
SOUNDS_DIR = CHARACTER.asset("sounds")

# ALSA output device for hardware audio playback (aplay -D).
# The USB combo device (mic+speaker) exposes two ALSA cards:
#   card 2: UACDemoV10 -> speaker/playback output
#   card 3: Device     -> microphone/capture input (held by sounddevice while agent runs)
# Use the playback card (UACDemoV10) so aplay doesn't conflict with the mic stream.
# Run 'aplay -l' to check your device names if this changes.
# ALSA_DEVICE = os.environ.get("ALSA_DEVICE", "plughw:UACDemoV10,0")
ALSA_DEVICE = os.environ.get("ALSA_DEVICE", "default")

# STT Settings
# NPU Speech2Text (preferred — 7x faster than CPU whisper.cpp)
WHISPER_HEF_PATH = os.environ.get("WHISPER_HEF_PATH", "./models/Whisper-Base.hef")
# CPU fallback (whisper.cpp subprocess)
WHISPER_CMD = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODELS = {
    "en": "./models/ggml-base.en.bin",
    "de": "./models/ggml-base.bin",  # Multilingual model for non-English
}
WHISPER_MODEL = WHISPER_MODELS.get(LANGUAGE, WHISPER_MODELS["en"])

# Audio Settings
MIC_DEVICE_INDEX = int(os.environ.get("MIC_DEVICE_INDEX", "1"))
MIC_SAMPLE_RATE = 48000
WAKE_WORD_MODEL = CHARACTER.asset("wakeword")
WAKE_WORD_THRESHOLD = CHARACTER.wake_threshold if CHARACTER.wake_threshold is not None else 0.35

# UI Settings
# Which face to show: "web" = the live three.js face in a local browser on top
# of the window, "tk" = the PNG frames only.

def t(key):
    """Get a spoken line from the character pack in the current language.
    Returns a list (alternatives) or a single string depending on the key."""
    return CHARACTER.line(key, LANGUAGE)

# Which model writes the replies: "local" (the NPU) or "cloud" (a hosted model,
# see core/cloud_llm.py). To change the hosted model, check that it accepts
# the options cloud_llm.py sends.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")
CLOUD_LLM_MODEL = "claude-haiku-5-5"      # the smallest and cheapest current Claude model


def say(key, **values):
    """Like :func:`t`, but always one string (a random pick from a list).
    Keyword arguments fill ``${placeholders}`` in the line."""
    return CHARACTER.say(key, LANGUAGE, **values)
