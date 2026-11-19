"""Microphone recording and wake-word detection.

Extracted from ``BotGUI`` so the logic is reusable and the GUI class
doesn't need to own mic-level concerns directly.
"""

import time
import wave
import logging

import numpy as np
import sounddevice as sd

from . import trace
from .config import MIC_DEVICE_INDEX, MIC_SAMPLE_RATE, WAKE_WORD_THRESHOLD, SILENCE_THRESHOLD
from .log import bmo_print

logger = logging.getLogger(__name__)


# Score of the most recent wake-word detection (read by the turn trace)
last_wake_score = None


def wait_for_wakeword(oww, stop_event, get_state, suppressed_states, *, extra_suppress_fn=None,
                      interrupt_fn=None):
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
    interrupt_fn : callable or None
        Checked on every chunk; when it returns True the wait ends and
        ``True`` is returned although no wake word was heard (used to
        start an injected test turn).
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

                if interrupt_fn and interrupt_fn():
                    oww.reset()
                    return True

                if get_state() in suppressed_states or (extra_suppress_fn and extra_suppress_fn()):
                    oww.reset()
                    continue

                oww.predict(audio_16k)

                for key in oww.prediction_buffer.keys():
                    if oww.prediction_buffer[key][-1] > WAKE_WORD_THRESHOLD:
                        bmo_print("WAKE", f"Detected: {key}")
                        global last_wake_score
                        last_wake_score = round(float(oww.prediction_buffer[key][-1]), 3)
                        oww.reset()
                        return True
    except Exception as e:
        bmo_print("AUDIO", f"Input Error: {e}")
        return False

    return False


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
        speech was ever detected.
    silence_chunks_after_speech : int
        How many consecutive silent chunks to wait after speech was detected.
    ignore_sec : float
        Seconds at the start during which audio is completely ignored
        (used by follow-up recording to let BMO's own TTS echo die down).
    filename : str
        Output WAV filename.
    """
    frames = []
    silent_chunks = 0
    has_spoken = False
    max_vol_seen = 0.0
    block_frames = 0        # frames per callback, as delivered by the driver
    n_blocks = 0
    ignore_until = time.time() + ignore_sec

    def callback(indata, frames_count, time_info, status):
        nonlocal silent_chunks, has_spoken, max_vol_seen, block_frames, n_blocks

        if time.time() < ignore_until:
            return
        block_frames = frames_count
        n_blocks += 1
        vol = np.linalg.norm(indata) * 10
        if meter_cb is not None:
            meter_cb(vol)
        max_vol_seen = max(max_vol_seen, vol)
        frames.append(indata.copy())
        if vol < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0
            has_spoken = True

    try:
        record_start = time.time()
        with sd.InputStream(samplerate=MIC_SAMPLE_RATE, device=MIC_DEVICE_INDEX,
                            channels=1, dtype='int16', callback=callback):
            while not stop_event.is_set():
                sd.sleep(50)
                elapsed = time.time() - record_start

                if elapsed < grace_sec:
                    continue

                if not has_spoken and silent_chunks > silence_chunks_no_speech:
                    if ignore_sec > 0:
                        bmo_print("FOLLOW-UP", f"Timeout. Max mic volume: {max_vol_seen:.2f} (threshold {SILENCE_THRESHOLD})")
                    break

                if has_spoken and silent_chunks > silence_chunks_after_speech:
                    break

                if elapsed > timeout_sec:
                    if ignore_sec > 0:
                        bmo_print("FOLLOW-UP", f"Max deadline hit. Max volume: {max_vol_seen:.2f}")
                    break
    except Exception as e:
        bmo_print("STT", f"Recording Error: {e}")
        return None

    trace.set("mic", {
        "block_frames": block_frames,
        "blocks": n_blocks,
        "seconds": round(n_blocks * block_frames / MIC_SAMPLE_RATE, 2),
        "max_volume": round(float(max_vol_seen)),
        "silence_threshold": SILENCE_THRESHOLD,
        "spoke": has_spoken,
    })

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
        bmo_print("STT", f"Speech finished! Max mic volume: {max_vol_seen:.2f}")

    return filename
