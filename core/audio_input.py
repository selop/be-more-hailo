"""Microphone recording and wake-word detection.

Extracted from ``BotGUI`` so the logic is reusable and the GUI class
doesn't need to own mic-level concerns directly.
"""

import time
import wave
import logging

import numpy as np
import sounddevice as sd

from .config import (
    MIC_DEVICE_INDEX, MIC_SAMPLE_RATE, WAKE_WORD_THRESHOLD,
    SILENCE_RMS_THRESHOLD, NOISE_FLOOR_MULTIPLIER, MIN_SPEECH_CHUNKS, SILENCE_TIERS,
)
from .log import bmo_print

logger = logging.getLogger(__name__)


def wait_for_wakeword(oww, stop_event, get_state, suppressed_states, *, extra_suppress_fn=None):
    """Block until the wake word is heard, returning ``True``.

    Parameters
    ----------
    oww : openwakeword.Model
        Loaded wake-word model.
    stop_event : threading.Event
        Set to abort.
    get_state : callable
        Returns the current bot state string.
    suppressed_states : set[str]
        States during which wake-word detection is suppressed.
    extra_suppress_fn : callable or None
        Additional suppression check (e.g. post-speech cooldown).
        Returns True to suppress.
    """
    CHUNK = 1280
    capture_rate = MIC_SAMPLE_RATE
    target_rate = 16000
    downsample_factor = capture_rate // target_rate

    try:
        with sd.InputStream(samplerate=capture_rate, device=MIC_DEVICE_INDEX, channels=1, dtype='int16') as stream:
            while not stop_event.is_set():
                data, _ = stream.read(CHUNK * downsample_factor)
                audio_16k = data[::downsample_factor].flatten()

                if get_state() in suppressed_states or (extra_suppress_fn and extra_suppress_fn()):
                    oww.reset()
                    continue

                oww.predict(audio_16k)

                for key in oww.prediction_buffer.keys():
                    if oww.prediction_buffer[key][-1] > WAKE_WORD_THRESHOLD:
                        bmo_print("WAKE", f"Detected: {key}")
                        oww.reset()
                        return True
    except Exception as e:
        bmo_print("AUDIO", f"Input Error: {e}")
        return False

    return False


def _silence_timeout_for_duration(speech_duration):
    """Return the required silence duration (seconds) based on how long the user has spoken."""
    for max_dur, silence_sec in SILENCE_TIERS:
        if speech_duration < max_dur:
            return silence_sec
    return SILENCE_TIERS[-1][1]


def record_until_silence(
    stop_event,
    meter_cb=None,
    *,
    grace_sec=1.5,
    timeout_sec=30.0,
    silence_chunks_no_speech=100,
    silence_chunks_after_speech=40,
    ignore_sec=0.0,
    filename="input.wav",
):
    """Record from the microphone until silence, returning the WAV path or ``None``.

    Parameters
    ----------
    stop_event : threading.Event
        Set to abort recording.
    meter_cb : callable or None
        Called with the volume level each chunk (for VU meter display).
    grace_sec : float
        Seconds to wait before checking for silence (lets the user start speaking).
    timeout_sec : float
        Hard maximum recording time.
    silence_chunks_no_speech : int
        How many consecutive silent chunks to wait before giving up if no
        speech was ever detected.  (Legacy parameter — kept for API compat.)
    silence_chunks_after_speech : int
        Legacy parameter — silence timeout is now tier-based.
    ignore_sec : float
        Seconds at the start during which audio is completely ignored
        (used by follow-up recording to let BMO's own TTS echo die down).
    filename : str
        Output WAV filename.
    """
    frames = []
    silent_chunks = 0
    speech_chunks = 0
    has_spoken = False
    max_vol_seen = 0.0
    noise_floor = float(SILENCE_RMS_THRESHOLD)
    speech_start_time = None
    ignore_until = time.time() + ignore_sec

    # Approximate chunk duration — sounddevice default blocksize is ~2048 samples
    # at 48 kHz ≈ 42.7 ms per callback.  We compute exact duration from sample count.
    chunk_duration = None  # set on first callback

    def callback(indata, frames_count, time_info, status):
        nonlocal silent_chunks, speech_chunks, has_spoken, max_vol_seen
        nonlocal noise_floor, speech_start_time, chunk_duration

        if time.time() < ignore_until:
            return

        if chunk_duration is None:
            chunk_duration = frames_count / MIC_SAMPLE_RATE

        # RMS volume — chunk-size-independent measure of average power
        rms = np.sqrt(np.mean(indata.astype(np.float64) ** 2))

        # Feed VU meter (scale to match MicMeter.VOL_MAX range ~150,000)
        if meter_cb is not None:
            meter_cb(rms * 100)

        max_vol_seen = max(max_vol_seen, rms)
        frames.append(indata.copy())

        # Adaptive noise floor — only update when likely not speech
        if rms < noise_floor * 2.0:
            noise_floor = noise_floor * 0.98 + rms * 0.02

        threshold = max(noise_floor * NOISE_FLOOR_MULTIPLIER, SILENCE_RMS_THRESHOLD)

        if rms < threshold:
            silent_chunks += 1
            speech_chunks = 0  # reset — require consecutive speech chunks
        else:
            silent_chunks = 0
            speech_chunks += 1
            # Require MIN_SPEECH_CHUNKS above-threshold chunks before arming
            if not has_spoken and speech_chunks >= MIN_SPEECH_CHUNKS:
                has_spoken = True
                speech_start_time = time.time()
                logger.info("Speech armed: RMS=%.0f floor=%.0f thresh=%.0f",
                            rms, noise_floor, threshold)

    try:
        record_start = time.time()
        with sd.InputStream(samplerate=MIC_SAMPLE_RATE, device=MIC_DEVICE_INDEX,
                            channels=1, dtype='int16', callback=callback):
            while not stop_event.is_set():
                sd.sleep(50)
                elapsed = time.time() - record_start

                if elapsed < grace_sec:
                    continue

                # No speech detected yet — use legacy chunk count for "give up" timeout
                if not has_spoken and silent_chunks > silence_chunks_no_speech:
                    if ignore_sec > 0:
                        bmo_print("FOLLOW-UP",
                                  f"Timeout. Max RMS: {max_vol_seen:.0f} "
                                  f"(floor {noise_floor:.0f}, threshold {SILENCE_RMS_THRESHOLD})")
                    break

                # Tiered silence timeout after speech
                if has_spoken and chunk_duration is not None:
                    speech_duration = time.time() - speech_start_time
                    required_silence = _silence_timeout_for_duration(speech_duration)
                    silence_elapsed = silent_chunks * chunk_duration
                    if silence_elapsed >= required_silence:
                        logger.info("Silence stop: spoke=%.1fs, silent=%.1fs/%.1fs, floor=%.0f",
                                    speech_duration, silence_elapsed, required_silence, noise_floor)
                        break

                if elapsed > timeout_sec:
                    if ignore_sec > 0:
                        bmo_print("FOLLOW-UP", f"Max deadline hit. Max RMS: {max_vol_seen:.0f}")
                    break
    except Exception as e:
        bmo_print("STT", f"Recording Error: {e}")
        return None

    if ignore_sec > 0:
        # Give ALSA/PortAudio time to fully close the stream context at OS level
        time.sleep(0.5)

    if not has_spoken or not frames:
        return None

    # Filter out any empty arrays from the callback race
    valid_frames = [f for f in frames if f is not None and len(f) > 0]
    if not valid_frames:
        return None

    try:
        data = np.concatenate(valid_frames, axis=0)
    except Exception as e:
        bmo_print("STT", f"Audio concat error: {e}")
        return None

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(data.tobytes())

    if ignore_sec > 0:
        bmo_print("STT", f"Speech finished! Max RMS: {max_vol_seen:.0f}")

    return filename
