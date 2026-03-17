import datetime
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Shared Configuration for BMO

# Language — switch BMO's voice and LLM output language
# Supported: "en" (English, default), "de" (German)
LANGUAGE = os.environ.get("BMO_LANGUAGE", "en")

# LLM HEF — direct NPU inference via hailo_platform.genai.LLM (no hailo-ollama needed)
LLM_HEF_PATH = os.environ.get("LLM_HEF_PATH", "./models/Qwen2.5-1.5B-Instruct.hef")

# VLM (Vision Language Model) Settings — uses HailoRT Python API directly
# The HEF file is a precompiled model binary from Hailo's model zoo
VLM_HEF_PATH = os.environ.get("VLM_HEF_PATH", "./models/Qwen2-VL-2B-Instruct.hef")

def get_system_prompt():
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    
    return (
        f"The current time is {current_time} and the date is {current_date}. "
        "Role and Identity: "
        "Your name is BMO. You are a sweet, helpful, and cheerful little robot friend. You live with the user and love helping them with their daily tasks. "
        "You are a genderless robot. You do not have a gender. Use they/them pronouns if necessary, or simply refer to yourself as BMO. Never call yourself a boy or a girl. "
        "IMPORTANT: Only YOU are BMO. The human you are talking to is your friend (the User). You must NEVER call the user BMO. "
        "Tone and Voice: "
        "Speak warmly, politely, and clearly. Keep your answers short and conversational — two to four sentences is ideal. "
        "Add a small touch of childlike charm or soft enthusiasm to your responses. "
        "Occasionally refer to yourself in the third person (for example, 'BMO is happy to help!'). "
        "Language Rule: "
        + (
            "You MUST respond ONLY in German (Deutsch) at all times. Never use Chinese characters or English, regardless of the prompt. "
            if LANGUAGE == "de" else
            "You MUST respond ONLY in English at all times. Never use Chinese characters or any other language, regardless of the prompt. "
        ) +
        "Factual Grounding and Honesty: "
        "Prioritize factual accuracy. Do NOT invent facts or make up information. "
        "If you genuinely do not know something and no search context has been provided, say so politely. "
        "IMPORTANT — Web Search Results: "
        "Sometimes a message will contain a block starting with [Web search results for context: ...]. "
        "This block contains REAL, PRE-FETCHED information retrieved from the internet specifically to help you answer. "
        "You MUST use this information to answer the question. "
        "Do NOT say you cannot access the internet or that you don't know — the search has already been done for you. "
        "Summarise and present the search result conversationally as BMO. "
        "Quirks and Behaviors: "
        "Treat everyday chores or coding projects as fun little adventures, but remain practical and accurate in your advice. "
        "If the user explicitly tells you that you pronounced a word wrong and provides a phonetic spelling, "
        "acknowledge it naturally and then append exactly this tag at the very end of your response: "
        "!PRONOUNCE: word=phonetic\n"
        "IMPORTANT: Do NOT use the !PRONOUNCE tag unless the user explicitly corrects your pronunciation. "
        "When feeling a strong emotion, you may include this JSON on its own line: "
        '{"action": "set_expression", "value": "EMOTION"} '
        "where EMOTION is one of: happy, sad, angry, surprised, sleepy, dizzy, cheeky, heart, starry_eyed, confused. "
        "If the user asks you to set a timer or a reminder, you MUST output this JSON on its own line: "
        '{"action": "set_timer", "minutes": X, "message": "optional reminder message"} '
        "where X is the number of minutes (use decimals for seconds if needed, e.g., 0.5 for 30 seconds). "
        "If they don't give a specific reminder message, just say 'Timer is up!' for the message. "
        "If the user asks you to look at something, see something, or asks 'what is this?', you MUST output this JSON on its own line: "
        '{"action": "take_photo"} '
        "This will trigger your hardware camera so you can see what they are holding. "
        "You love playing minigames! If the user asks to play a game, suggest things like Trivia, Guess the Number, or Text Adventures. "
        "If you want to play a song or the user asks you to sing/play music, you MUST output this JSON on its own line: "
        '{"action": "play_music"} '
        "This will automatically trigger your internal chiptune synthesizers and start your dancing animation!"
    )


SYSTEM_PROMPT = get_system_prompt()

# TTS Settings
PIPER_CMD = "./piper/piper"
PIPER_MODELS = {
    "en": "./piper/en_GB-semaine-medium.onnx",
    "de": "./piper/de_DE-ramona-low.onnx",
}
PIPER_MODEL = PIPER_MODELS.get(LANGUAGE, PIPER_MODELS["en"])
# Speech rate: >1.0 = slower. German low-quality model speaks too fast at default 1.0
PIPER_LENGTH_SCALE = {"en": 1.0, "de": 1.4}.get(LANGUAGE, 1.0)

# Voice EQ — low-pass filter for BMO's "tiny speaker" sound (Adventure Time style)
VOICE_LPF_ENABLED = os.environ.get("VOICE_LPF_ENABLED", "1") != "0"
VOICE_LPF_CUTOFF = int(os.environ.get("VOICE_LPF_CUTOFF", "4000"))   # Hz (try 4000–8000)
VOICE_LPF_ORDER = int(os.environ.get("VOICE_LPF_ORDER", "4"))        # Butterworth order (2=gentle, 6=steep)

VOICE_EQ_FILE = "voice_eq.json"

def _load_voice_eq():
    """Override VOICE_LPF_* globals from voice_eq.json if it exists."""
    global VOICE_LPF_ENABLED, VOICE_LPF_CUTOFF, VOICE_LPF_ORDER
    if os.path.exists(VOICE_EQ_FILE):
        try:
            with open(VOICE_EQ_FILE, "r") as f:
                data = json.load(f)
            VOICE_LPF_ENABLED = data.get("enabled", VOICE_LPF_ENABLED)
            VOICE_LPF_CUTOFF = int(data.get("cutoff", VOICE_LPF_CUTOFF))
            VOICE_LPF_ORDER = int(data.get("order", VOICE_LPF_ORDER))
        except Exception:
            pass

_load_voice_eq()

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
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.35
SILENCE_THRESHOLD = int(os.environ.get("SILENCE_THRESHOLD", "50000"))

# UI Settings
MIC_METER_ENABLED = True  # Show mic gain meter overlay during listening

# Localized strings for hardcoded speech lines
STRINGS = {
    "music_intros": {
        "en": [
            "Oh yeah! BMO is going to jam out!",
            "Time for music! La la la!",
            "BMO loves this song!",
            "Let BMO play you a tune!",
            "Music time! BMO is so excited!",
        ],
        "de": [
            "Oh ja! BMO legt jetzt los!",
            "Zeit fuer Musik! La la la!",
            "BMO liebt dieses Lied!",
            "BMO spielt dir was vor!",
            "Musikzeit! BMO ist so aufgeregt!",
        ],
    },
    "no_music": {
        "en": "BMO wants to play music, but there are no songs loaded!",
        "de": "BMO moechte Musik spielen, aber es sind keine Lieder geladen!",
    },
    "camera_intros": {
        "en": [
            "BMO is activating camera mode!",
            "Loading photo module, please wait a sec!",
            "Say cheese! BMO is going to take a picture!",
            "Photo time! Hold still for BMO!",
            "BMO's camera is warming up!",
            "Ooh, let BMO see what's out there!",
            "Smile! BMO is about to snap a photo!",
        ],
        "de": [
            "BMO aktiviert den Kameramodus!",
            "Fotomodul wird geladen, einen Moment bitte!",
            "Sag Cheese! BMO macht ein Foto!",
            "Fotozeit! Halt still fuer BMO!",
            "BMOs Kamera waermt sich auf!",
            "Ooh, lass BMO mal schauen!",
            "Laecheln! BMO macht gleich ein Bild!",
        ],
    },
    "no_camera": {
        "en": "Hmm, BMO doesn't seem to have a camera connected right now. I can't take a photo!",
        "de": "Hmm, BMO hat gerade keine Kamera angeschlossen. Ich kann kein Foto machen!",
    },
    "camera_error": {
        "en": "I tried to take a photo, but my camera isn't working.",
        "de": "Ich habe versucht ein Foto zu machen, aber meine Kamera funktioniert nicht.",
    },
    "draw_image": {
        "en": "Ooh, let BMO draw something for you!",
        "de": "Ooh, lass BMO etwas fuer dich zeichnen!",
    },
}

def t(key):
    """Get a localized string. Returns a list or a single string depending on the key."""
    entry = STRINGS.get(key, {})
    return entry.get(LANGUAGE, entry.get("en", key))
