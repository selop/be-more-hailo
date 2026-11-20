"""Microphone recording and wake-word detection."""

import queue
import threading
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


class Microphone:
    """Microphone audio in blocks of *block_frames*, handed over through a queue.

    A reader thread does blocking reads, which deliver every sample. (A
    callback stream with a pinned block size delivered 9.5 instead of 12.5
    blocks per second on the device: a quarter of the audio was lost.) The
    consumer waits with a timeout, so a stream that goes silent is noticed
    instead of blocking forever.
    """

    def __init__(self, block_frames: int):
        self.block_frames = block_frames
        self._blocks = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._stream = None

    def __enter__(self):
        self._stream = sd.InputStream(samplerate=MIC_SAMPLE_RATE, device=MIC_DEVICE_INDEX, channels=1, dtype='int16')
        self._stream.start()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        return self

    def _read(self):
        try:
            while not self._stop.is_set():
                data, _ = self._stream.read(self.block_frames)
                try:
                    self._blocks.put_nowait(data)
                except queue.Full:
                    pass        # the consumer is behind; the oldest audio is what it still holds
        except Exception as e:  # noqa: BLE001 - the stream was closed under us or the device failed
            if not self._stop.is_set():
                bmo_print("AUDIO", f"Microphone read failed: {e!r}")

    def get(self, timeout: float):
        """Next block; raises ``queue.Empty`` when none arrives within *timeout* seconds."""
        return self._blocks.get(timeout=timeout)

    def __exit__(self, *exc):
        # The reader must be out of its read before the stream is closed: closing it under a
        # running read crashed the whole process (SIGSEGV in PortAudio, twice on the device).
        self._stop.set()
        self._reader.join(timeout=1.0)
        if self._reader.is_alive():
            # Still inside a read: the stream is dead. It is left open on purpose and the
            # agent has to be restarted (see microphone_stuck()).
            _stuck.append(self._stream)
            bmo_print("AUDIO", "Microphone stream is stuck in a read; it cannot be closed safely")
            return False
        try:
            self._stream.close()
        except Exception as e:  # noqa: BLE001
            bmo_print("AUDIO", f"Closing the microphone stream failed: {e!r}")
        return False


_stuck = []     # streams that could not be closed; kept so they are never garbage-collected (which closes them)


def microphone_stuck() -> bool:
    """True once a microphone stream had to be abandoned. Only a restart of the process recovers the device."""
    return bool(_stuck)


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
        with Microphone(CHUNK * downsample_factor) as mic:
            while not stop_event.is_set():
                try:
                    data = mic.get(timeout=STALL_GRACE_SEC)
                except queue.Empty:
                    bmo_print("AUDIO", f"No microphone audio for {STALL_GRACE_SEC:.0f} s; reopening the stream")
                    trace.event("mic_stalled", where="wake")
                    if interrupt_fn and interrupt_fn():
                        return True         # an injected test turn does not need the microphone
                    return False            # the caller comes back and the stream is opened afresh
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
STALL_GRACE_SEC = 3.0     # no microphone audio for this long means the stream has stalled


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

    try:
        with Microphone(int(MIC_SAMPLE_RATE * BLOCK_SEC)) as mic:
            while not stop_event.is_set() and not endpointer.ended:
                try:
                    block = mic.get(timeout=STALL_GRACE_SEC)
                except queue.Empty:
                    # A stream that opens but delivers nothing (seen on the device, agent stuck on "Listening")
                    endpointer.ended = "stalled"
                    bmo_print("STT", f"No microphone audio for {STALL_GRACE_SEC:.0f} s; giving up on this recording")
                    trace.event("mic_stalled", where="record", blocks=len(frames))
                    break
                level = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
                if meter_cb is not None:
                    meter_cb(level)
                frames.append(block)
                endpointer.feed(level, len(block) / MIC_SAMPLE_RATE)
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
