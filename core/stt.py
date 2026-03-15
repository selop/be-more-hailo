import subprocess
import logging
import os
import re
import wave
from .config import WHISPER_CMD, WHISPER_MODEL, LANGUAGE

logger = logging.getLogger(__name__)

def _is_16khz_mono_wav(filepath: str) -> bool:
    """Check if file is already a 16kHz mono WAV (no conversion needed)."""
    try:
        with wave.open(filepath, 'rb') as wf:
            return wf.getframerate() == 16000 and wf.getnchannels() == 1
    except Exception:
        return False

def transcribe_audio(audio_filepath: str) -> str:
    """
    Transcribes audio using whisper.cpp.
    Skips FFmpeg conversion if the input is already 16kHz mono WAV.
    """
    if not os.path.exists(audio_filepath):
        logger.error(f"Audio file not found: {audio_filepath}")
        return ""

    temp_wav = None
    whisper_input = audio_filepath

    try:
        # Only convert if not already 16kHz mono WAV (e.g. WebM uploads from browser)
        if not _is_16khz_mono_wav(audio_filepath):
            temp_wav = f"{audio_filepath}_16k.wav"
            logger.info(f"Converting {audio_filepath} to 16kHz WAV for whisper.cpp CPU inference...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_filepath, "-ar", "16000", "-ac", "1", temp_wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            whisper_input = temp_wav
        else:
            logger.info(f"Input {audio_filepath} is already 16kHz mono WAV, skipping FFmpeg.")

        # Run whisper.cpp
        cmd = [WHISPER_CMD, "-m", WHISPER_MODEL, "-f", whisper_input, "-nt"]
        if LANGUAGE != "en":
            cmd += ["-l", LANGUAGE]
        logger.info(f"Running whisper.cpp transcription on the CPU... CMD: {' '.join(cmd)}")
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Whisper CPU process failed with exit code {e.returncode}")
            return ""

        # Clean up output (remove timestamps like [00:00:00.000 --> 00:00:02.000] or [BLANK_AUDIO])
        output = re.sub(r'\[.*?\]', '', output).strip()

        # Fix capitalization of BMO
        output = re.sub(r'\b[Bb]emo\b', 'BMO', output)
        output = re.sub(r'\b[Bb]eemo\b', 'BMO', output)

        # Clean hallucinated whispers from silence
        lowered = output.lower()
        hallucinations = [
            "[silence]", "(silence)", "you", "thanks for watching!",
            "[blank_audio]", "thank you.", "thank you", "thanks."
        ]

        # Whisper often hallucinates sound descriptions when mic picks up silence or
        # ambient audio — e.g. "(eerie music)", "(background music)", "[SOUND]".
        # Reject any output that is entirely inside parentheses or brackets.
        is_parenthetical = bool(re.match(r'^\s*[\(\[].*[\)\]]\s*$', output.strip()))

        # If output is purely punctuation/noise (no letters or numbers) or a known hallucination
        if is_parenthetical or lowered in hallucinations or not re.search(r'[a-zA-Z0-9]', lowered):
            logger.info(f"Whisper hallucination filtered: {repr(output)}")
            return ""

        return output

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg or Whisper CPU process failed: {e}")
        return ""
    except Exception as e:
        logger.error(f"Transcription Error: {e}")
        return ""
    finally:
        # Only clean up the temp file if FFmpeg was actually used
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_wav}: {e}")
