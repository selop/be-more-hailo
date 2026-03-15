import subprocess
import logging
import os
import re
import wave
import numpy as np
from .config import WHISPER_CMD, WHISPER_MODEL, WHISPER_HEF_PATH, LANGUAGE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  NPU Speech2Text singleton (shares VDevice with LLM)
# --------------------------------------------------------------------------- #
_stt_instance = None


def init_stt():
    """Initialise Speech2Text on the shared VDevice. Safe to call multiple times.
    Falls back silently to CPU whisper.cpp if the HEF is missing or init fails."""
    global _stt_instance
    if _stt_instance is not None:
        return

    try:
        from .llm import _get_vdevice, _resolve_hef
        from hailo_platform.genai import Speech2Text

        hef = _resolve_hef(WHISPER_HEF_PATH)
        vdevice = _get_vdevice()

        logger.info(f"Initialising Speech2Text from {hef} ...")
        _stt_instance = Speech2Text(vdevice, hef)
        logger.info("Speech2Text ready (NPU)")
    except FileNotFoundError:
        logger.info("Whisper HEF not found — will use CPU whisper.cpp fallback")
    except Exception as e:
        logger.warning(f"Speech2Text init failed, using CPU fallback: {e}")
        _stt_instance = None


def release_stt():
    """Release Speech2Text (e.g. before VLM subprocess claims the NPU)."""
    global _stt_instance
    if _stt_instance is not None:
        try:
            _stt_instance.release()
        except Exception as e:
            logger.warning(f"Error releasing Speech2Text: {e}")
        _stt_instance = None
        logger.info("Speech2Text released")


def is_stt_npu() -> bool:
    """Return True if NPU Speech2Text is active."""
    return _stt_instance is not None


# --------------------------------------------------------------------------- #
#  Audio loading helpers
# --------------------------------------------------------------------------- #

def _load_audio_as_float32(filepath: str) -> np.ndarray:
    """Load a WAV file and return float32 mono audio at 16kHz."""
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    # Convert to numpy
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    # Mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)[:, 0]

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        from scipy.signal import resample
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = resample(audio, num_samples).astype(np.float32)

    return audio


def _is_16khz_mono_wav(filepath: str) -> bool:
    """Check if file is already a 16kHz mono WAV (no conversion needed)."""
    try:
        with wave.open(filepath, 'rb') as wf:
            return wf.getframerate() == 16000 and wf.getnchannels() == 1
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Post-processing (shared between NPU and CPU paths)
# --------------------------------------------------------------------------- #

def _clean_transcription(text: str) -> str:
    """Clean up raw STT output: remove timestamps, fix BMO spelling, filter hallucinations."""
    # Remove timestamps like [00:00:00.000 --> 00:00:02.000] or [BLANK_AUDIO]
    text = re.sub(r'\[.*?\]', '', text).strip()

    # Fix BMO spelling
    text = re.sub(r'\b[Bb]emo\b', 'BMO', text)
    text = re.sub(r'\b[Bb]eemo\b', 'BMO', text)

    # Filter hallucinations
    lowered = text.lower()
    hallucinations = [
        "[silence]", "(silence)", "you", "thanks for watching!",
        "[blank_audio]", "thank you.", "thank you", "thanks."
    ]

    # Reject output that is entirely inside parentheses or brackets
    is_parenthetical = bool(re.match(r'^\s*[\(\[].*[\)\]]\s*$', text.strip()))

    if is_parenthetical or lowered in hallucinations or not re.search(r'[a-zA-Z0-9]', lowered):
        logger.info(f"Hallucination filtered: {repr(text)}")
        return ""

    return text


# --------------------------------------------------------------------------- #
#  NPU transcription
# --------------------------------------------------------------------------- #

def _transcribe_npu(audio_filepath: str) -> str:
    """Transcribe using Hailo NPU Speech2Text."""
    from hailo_platform.genai import Speech2TextTask

    audio = _load_audio_as_float32(audio_filepath)
    logger.info(f"NPU STT: {len(audio)} samples ({len(audio)/16000:.1f}s)")

    segments = _stt_instance.generate_all_segments(
        audio,
        task=Speech2TextTask.TRANSCRIBE,
        language="en" if LANGUAGE == "en" else LANGUAGE,
        timeout_ms=15000,
    )
    text = " ".join(seg.text for seg in segments).strip()
    return _clean_transcription(text)


# --------------------------------------------------------------------------- #
#  CPU fallback transcription (whisper.cpp subprocess)
# --------------------------------------------------------------------------- #

def _transcribe_cpu(audio_filepath: str) -> str:
    """Transcribe using whisper.cpp on CPU (fallback)."""
    temp_wav = None
    whisper_input = audio_filepath

    try:
        # Convert if not already 16kHz mono WAV
        if not _is_16khz_mono_wav(audio_filepath):
            temp_wav = f"{audio_filepath}_16k.wav"
            logger.info(f"Converting {audio_filepath} to 16kHz WAV for whisper.cpp ...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_filepath, "-ar", "16000", "-ac", "1", temp_wav],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            whisper_input = temp_wav
        else:
            logger.info(f"Input is already 16kHz mono WAV, skipping FFmpeg.")

        cmd = [WHISPER_CMD, "-m", WHISPER_MODEL, "-f", whisper_input, "-nt"]
        if LANGUAGE != "en":
            cmd += ["-l", LANGUAGE]
        logger.info(f"Running whisper.cpp on CPU: {' '.join(cmd)}")

        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return _clean_transcription(output)

    except subprocess.CalledProcessError as e:
        logger.error(f"Whisper CPU process failed with exit code {e.returncode}")
        return ""
    except Exception as e:
        logger.error(f"CPU transcription error: {e}")
        return ""
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
#  Public API (same signature as before — drop-in replacement)
# --------------------------------------------------------------------------- #

def transcribe_audio(audio_filepath: str) -> str:
    """Transcribe audio to text. Uses NPU Speech2Text if available, else CPU whisper.cpp."""
    if not os.path.exists(audio_filepath):
        logger.error(f"Audio file not found: {audio_filepath}")
        return ""

    if _stt_instance is not None:
        try:
            return _transcribe_npu(audio_filepath)
        except Exception as e:
            logger.warning(f"NPU STT failed, falling back to CPU: {e}")

    return _transcribe_cpu(audio_filepath)
