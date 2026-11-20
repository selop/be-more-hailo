"""Microphone recording and wake-word detection."""

import time
import wave
import logging

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from . import trace
from .health import mic_monitor
from .config import MIC_DEVICE_INDEX, MIC_SAMPLE_RATE, WAKE_WORD_THRESHOLD
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
    mic_monitor.threshold = WAKE_WORD_THRESHOLD
    capture_rate = MIC_SAMPLE_RATE
    target_rate = 16000
    downsample_factor = capture_rate // target_rate

    try:
        with sd.InputStream(samplerate=capture_rate, device=MIC_DEVICE_INDEX, channels=1, dtype='int16') as stream:
            while not stop_event.is_set():
                data, _ = stream.read(CHUNK * downsample_factor)
                # Filtered resampling: plain decimation folds noise above 8 kHz (the fan) into the speech band
                audio_16k = np.clip(resample_poly(data[:, 0].astype(np.float32), 1, downsample_factor), -32768, 32767).astype(np.int16)

                if interrupt_fn and interrupt_fn():
                    oww.reset()
                    return True

                rms = float(np.sqrt(np.mean(audio_16k.astype(np.float32) ** 2)))
                if get_state() in suppressed_states or (extra_suppress_fn and extra_suppress_fn()):
                    mic_monitor.heard(rms, None, True)
                    oww.reset()
                    continue

                oww.predict(audio_16k)
                mic_monitor.heard(rms, max(float(v[-1]) for v in oww.prediction_buffer.values()), False)

                for key in oww.prediction_buffer.keys():
                    if oww.prediction_buffer[key][-1] > WAKE_WORD_THRESHOLD:
                        bmo_print("WAKE", f"Detected: {key}")
                        global last_wake_score
                        last_wake_score = round(float(oww.prediction_buffer[key][-1]), 3)
                        mic_monitor.detected(last_wake_score)
                        oww.reset()
                        return True
    except Exception as e:
        bmo_print("AUDIO", f"Input Error: {e}")
        return False

    return False


BLOCK_SEC = 0.03
STALL_GRACE_SEC = 3.0     # how long past the longest recording to wait for a silent microphone stream


class Endpointer:
    """Decides when a recording is over, from the loudness of each block.

    Speech is anything clearly above the room's noise floor, so a louder
    room (a fan, a fridge) raises the bar instead of keeping the recording
    open. Levels are RMS of 16-bit samples, times are seconds.
    """

    def __init__(self, noise_floor, *, no_speech_sec=4.0, hang_sec=0.9, max_sec=15.0):
        self.threshold = max(2.0 * noise_floor, 300.0)     # 300: quiet-room rustle is not speech
        self.no_speech_sec, self.hang_sec, self.max_sec = no_speech_sec, hang_sec, max_sec
        self.spoke = False
        self.ended = None           # "silence", "no_speech" or "max"
        self.max_level = 0.0
        self._elapsed = self._quiet = 0.0
        self._loud_blocks = 0

    def feed(self, level, seconds=BLOCK_SEC) -> bool:
        """Take one block; True when the recording should stop."""
        self._elapsed += seconds
        self.max_level = max(self.max_level, level)
        if level > self.threshold:
            self._loud_blocks += 1
            self._quiet = 0.0
            if self._loud_blocks >= 3:      # a click is not speech
                self.spoke = True
        else:
            self._loud_blocks = 0
            self._quiet += seconds
        if self.spoke and self._quiet >= self.hang_sec:
            self.ended = "silence"
        elif not self.spoke and self._elapsed >= self.no_speech_sec:
            self.ended = "no_speech"
        elif self._elapsed >= self.max_sec:
            self.ended = "max"
        return self.ended is not None


def record_until_silence(stop_event, meter_cb=None, *, filename="input.wav", max_sec=15.0):
    """Record from the microphone until the speaker stops, returning the WAV
    path, or ``None`` when nobody spoke. ``meter_cb`` gets each block's level."""
    floor = mic_monitor.snapshot().get("noise_floor") or 250      # the room just before the wake word
    endpointer = Endpointer(floor, max_sec=max_sec)
    frames = []

    def callback(indata, frames_count, time_info, status):
        if endpointer.ended:
            return
        level = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
        if meter_cb is not None:
            meter_cb(level)
        frames.append(indata.copy())
        endpointer.feed(level, frames_count / MIC_SAMPLE_RATE)

    try:
        with sd.InputStream(samplerate=MIC_SAMPLE_RATE, device=MIC_DEVICE_INDEX, channels=1, dtype='int16',
                            blocksize=int(MIC_SAMPLE_RATE * BLOCK_SEC), callback=callback):
            # The endpointer counts audio, not time: a stream that opens but delivers nothing
            # (seen on the device, agent stuck on "Listening") would otherwise wait forever.
            deadline = time.time() + max_sec + STALL_GRACE_SEC
            while not stop_event.is_set() and not endpointer.ended:
                if time.time() > deadline:
                    endpointer.ended = "stalled"
                    bmo_print("STT", f"Microphone stream delivered {len(frames)} blocks in {max_sec + STALL_GRACE_SEC:.0f} s; giving up")
                    trace.event("mic_stalled", blocks=len(frames))
                    break
                sd.sleep(30)
    except Exception as e:
        bmo_print("STT", f"Recording Error: {e}")
        return None

    trace.set("mic", {
        "seconds": round(sum(len(f) for f in frames) / MIC_SAMPLE_RATE, 2),
        "ended": endpointer.ended,
        "noise_floor": round(floor),
        "threshold": round(endpointer.threshold),
        "max_level": round(endpointer.max_level),
        "spoke": endpointer.spoke,
    })

    if not endpointer.spoke or not frames:
        return None

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(np.concatenate(frames, axis=0).tobytes())
    return filename
