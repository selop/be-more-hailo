import subprocess
import logging
import os
import re
import json
import struct
import threading
import wave
from .config import PIPER_CMD, PIPER_MODEL, ALSA_DEVICE, PIPER_LENGTH_SCALE

logger = logging.getLogger(__name__)

PRONUNCIATION_FILE = "pronunciations.json"

# --- Singletons (initialized by init_audio) ---
_player = None
_synthesizer = None


# =========================================================================
# Pronunciation helpers (unchanged)
# =========================================================================

def load_pronunciations() -> dict:
    """Loads the pronunciation dictionary from a JSON file."""
    if os.path.exists(PRONUNCIATION_FILE):
        try:
            with open(PRONUNCIATION_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading pronunciations: {e}")

    # Default dictionary if file doesn't exist or fails to load
    default_dict = {
        "cheesy": "cheezy",
        "poutine": "poo-teen",
        "bmo": "beemo"
    }
    save_pronunciations(default_dict)
    return default_dict

def save_pronunciations(pronunciations: dict):
    """Saves the pronunciation dictionary to a JSON file."""
    try:
        with open(PRONUNCIATION_FILE, "w") as f:
            json.dump(pronunciations, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving pronunciations: {e}")

def add_pronunciation(word: str, phonetic: str):
    """Adds a new pronunciation rule and saves it."""
    pronunciations = load_pronunciations()
    pronunciations[word.lower()] = phonetic
    save_pronunciations(pronunciations)

def clean_text_for_speech(text: str) -> str:
    """Removes markdown and special characters that shouldn't be spoken."""
    # Remove JSON blocks
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
    # Replace newlines with spaces to prevent shell line breaks during Piper TTS
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove asterisks used for emphasis or actions (e.g., *beep boop*)
    text = text.replace('*', '')
    # Remove other common markdown like bold/italics, headers, and list bullets
    text = re.sub(r'[_~`#\-]', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove emojis and other symbols (keep ASCII, common punctuation, and accents)
    text = re.sub(r'[^\x00-\x7F\xC0-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF\u2018-\u201F\u2028-\u202F]', '', text)

    # Apply pronunciation fixes (case-insensitive)
    pronunciations = load_pronunciations()
    for word, replacement in pronunciations.items():
        # Use regex word boundaries (\b) to ensure we only replace whole words
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text.strip()


# =========================================================================
# AudioPlayer — persistent sounddevice output stream
# =========================================================================

class AudioPlayer:
    """Plays raw int16 PCM audio through a persistent sounddevice OutputStream."""

    SAMPLE_RATE = 22050
    CHANNELS = 1
    CHUNK_FRAMES = 2048  # frames per write

    def __init__(self, alsa_device_name: str = None):
        import sounddevice as sd

        self._sd = sd
        self._stop_event = threading.Event()
        self._device_index = self._find_device(alsa_device_name)

        # Open a persistent output stream
        self._stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype='int16',
            device=self._device_index,
        )
        self._stream.start()
        logger.info(f"AudioPlayer: opened persistent stream on device {self._device_index} @ {self.SAMPLE_RATE}Hz")

    def _find_device(self, alsa_device_name: str):
        """Find sounddevice output device matching the ALSA card name."""
        if not alsa_device_name:
            return None  # use default

        import sounddevice as sd
        # Extract card name from ALSA device string like "plughw:UACDemoV10,0"
        card_name = alsa_device_name
        if ':' in card_name:
            card_name = card_name.split(':')[1].split(',')[0]

        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if card_name in dev['name'] and dev['max_output_channels'] > 0:
                    logger.info(f"AudioPlayer: matched ALSA card '{card_name}' to device {i}: {dev['name']}")
                    return i
        except Exception as e:
            logger.warning(f"AudioPlayer: device query failed: {e}")

        logger.info(f"AudioPlayer: no match for '{card_name}', using default output device")
        return None

    def play(self, pcm_bytes: bytes):
        """Play raw int16 PCM bytes. Can be interrupted via stop_playback()."""
        self._stop_event.clear()
        chunk_bytes = self.CHUNK_FRAMES * 2  # int16 = 2 bytes per sample

        import numpy as np
        offset = 0
        while offset < len(pcm_bytes):
            if self._stop_event.is_set():
                logger.info("AudioPlayer: playback interrupted")
                break
            end = min(offset + chunk_bytes, len(pcm_bytes))
            chunk = np.frombuffer(pcm_bytes[offset:end], dtype='int16')
            # Reshape for single-channel write
            self._stream.write(chunk.reshape(-1, 1))
            offset = end

    def stop_playback(self):
        """Interrupt current playback immediately."""
        self._stop_event.set()

    def resume(self):
        """Clear stop event so next play() works normally."""
        self._stop_event.clear()

    def close(self):
        """Shut down the persistent stream."""
        try:
            self._stream.stop()
            self._stream.close()
            logger.info("AudioPlayer: stream closed")
        except Exception as e:
            logger.warning(f"AudioPlayer: close error: {e}")


# =========================================================================
# PiperSynthesizer — Piper library with subprocess fallback
# =========================================================================

class PiperSynthesizer:
    """Synthesize text to raw int16 PCM at 22050Hz using Piper."""

    def __init__(self):
        self._piper_voice = None
        self._use_library = False

        # Try piper-tts Python package first
        try:
            from piper import PiperVoice
            self._piper_voice = PiperVoice.load(PIPER_MODEL)
            self._use_library = True
            logger.info(f"PiperSynthesizer: using piper-tts library with model {PIPER_MODEL}")
        except ImportError:
            logger.info("PiperSynthesizer: piper-tts not installed, falling back to subprocess")
        except Exception as e:
            logger.warning(f"PiperSynthesizer: piper-tts load failed ({e}), falling back to subprocess")

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw int16 PCM bytes at 22050Hz."""
        if self._use_library:
            return self._synthesize_library(text)
        return self._synthesize_subprocess(text)

    def _synthesize_library(self, text: str) -> bytes:
        """Use piper-tts Python library."""
        import io
        import wave as wave_mod

        audio_buf = io.BytesIO()
        with wave_mod.open(audio_buf, 'wb') as wf:
            self._piper_voice.synthesize(text, wf, length_scale=PIPER_LENGTH_SCALE)

        # Extract raw PCM from the WAV (skip header)
        audio_buf.seek(0)
        with wave_mod.open(audio_buf, 'rb') as wf:
            return wf.readframes(wf.getnframes())

    def _synthesize_subprocess(self, text: str) -> bytes:
        """Fall back to Piper CLI subprocess (still eliminates aplay)."""
        safe_text = text.replace("'", "'\\''")
        cmd = f"echo '{safe_text}' | {PIPER_CMD} --model {PIPER_MODEL} --length-scale {PIPER_LENGTH_SCALE} --output_raw"
        res = subprocess.run(cmd, shell=True, capture_output=True)
        if res.returncode != 0:
            logger.error(f"Piper subprocess error: {res.stderr}")
            return b""
        return res.stdout


# =========================================================================
# Module-level init / shutdown / accessors
# =========================================================================

def init_audio(alsa_device_name: str = None):
    """Initialize the persistent audio player and Piper synthesizer."""
    global _player, _synthesizer
    _player = AudioPlayer(alsa_device_name)
    _synthesizer = PiperSynthesizer()
    logger.info("Audio subsystem initialized")

def shutdown_audio():
    """Close the persistent audio stream."""
    global _player
    if _player:
        _player.close()
        _player = None
    logger.info("Audio subsystem shut down")

def get_player() -> AudioPlayer:
    """Get the singleton AudioPlayer instance."""
    return _player

def get_synthesizer() -> PiperSynthesizer:
    """Get the singleton PiperSynthesizer instance."""
    return _synthesizer


# =========================================================================
# Public API (used by web_app.py and agent_hailo.py)
# =========================================================================

def play_audio_on_hardware(text: str):
    """Plays audio directly out of the Pi's speakers using Piper + persistent stream."""
    try:
        clean_text = clean_text_for_speech(text)
        if not clean_text or not any(c.isalnum() for c in clean_text):
            return

        logger.info(f"Playing audio on hardware: {clean_text[:30]}...")

        synth = get_synthesizer()
        player = get_player()

        if synth and player:
            pcm = synth.synthesize(clean_text)
            if pcm:
                player.play(pcm)
        else:
            # Fallback: original shell pipeline (init_audio not called, e.g. web_app)
            safe_text = clean_text.replace("'", "'\\''")
            piper_cmd = f"echo '{safe_text}' | {PIPER_CMD} --model {PIPER_MODEL} --output_raw | aplay -D {ALSA_DEVICE} -r 22050 -f S16_LE -t raw"
            subprocess.run(piper_cmd, shell=True, check=True)
    except Exception as e:
        logger.error(f"Hardware TTS Error: {e}")

def generate_audio_file(text: str, filename: str) -> str:
    """Generates a WAV file using Piper for the browser to play."""
    try:
        clean_text = clean_text_for_speech(text)
        if not clean_text or not any(c.isalnum() for c in clean_text):
            return None

        logger.info(f"Generating audio file: {filename}")
        filepath = os.path.join("static", "audio", filename)

        synth = get_synthesizer()
        if synth:
            pcm = synth.synthesize(clean_text)
            if pcm:
                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(pcm)
                return f"/static/audio/{filename}"
            return None

        # Fallback: original subprocess
        safe_text = clean_text.replace("'", "'\\''")
        piper_cmd = f"echo '{safe_text}' | {PIPER_CMD} --model {PIPER_MODEL} --output_file {filepath}"
        subprocess.run(piper_cmd, shell=True, check=True)
        return f"/static/audio/{filename}"
    except Exception as e:
        logger.error(f"File TTS Error: {e}")
        return None
