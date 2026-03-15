# Pipeline Architecture Review: Current BMO vs Hailo-Apps Patterns

## Current BMO Pipeline

```
Mic (48kHz) → sounddevice → WAV file → Speech2Text (NPU, Python API)
    → text → LLM.generate() (NPU, direct Python API, streaming)
    → sentence chunker → piper Python lib → AudioPlayer (persistent stream)
```

Tiers 1-3A eliminated hailo-ollama, whisper.cpp, and FFmpeg from the hot path. Remaining subprocesses: `aplay` (sound effects only — not latency-critical), `rpicam-still` (camera capture).

## Hailo-Apps Reference Pipeline

```
Mic (16kHz) → sounddevice → numpy array → Speech2Text (NPU, Python API)
    → text → LLM.generate() (NPU, Python API, streaming context manager)
    → token callback → sentence chunker → piper Python lib → AudioPlayer (persistent stream)
```

**Zero subprocesses.** Everything is in-process Python with direct NPU API calls.

Reference implementation: `hailo-apps/python/gen_ai_apps/voice_assistant/voice_assistant.py`

---

## Key Differences

| Aspect | BMO Now | Hailo-Apps Pattern | Impact |
|--------|---------|-------------------|--------|
| **STT** | whisper.cpp on CPU (subprocess) | `Speech2Text` on NPU (Python API) | Frees CPU, ~2x faster |
| **LLM** | ✅ `LLM.generate()` direct Python API | `LLM.generate()` direct Python API | Done (Tier 2) |
| **TTS** | ✅ Piper Python lib + persistent OutputStream | Piper Python lib + persistent `sounddevice.OutputStream` | Done (Tier 1) |
| **VAD** | OpenWakeWord only (wake word, not VAD) | `webrtcvad` for speech endpoint detection | Better silence detection |
| **Audio capture** | 48kHz → file → FFmpeg → 16kHz file | 16kHz direct to numpy array | No temp files, no FFmpeg |
| **NPU sharing** | N/A — one model at a time (see below) | `VDevice(group_id="SHARED")` | STT + LLM may coexist |
| **TTS interruption** | None | `generation_id` invalidation | User can interrupt BMO |
| **Audio playback** | ✅ Persistent OutputStream | Persistent OutputStream (writes silence when idle) | Done (Tier 1) |
| **Context caching** | ✅ `llm.save_context()` / `llm.load_context()` | `llm.save_context()` / `llm.load_context()` | Done (Tier 2) |

---

## Prerequisites

Before starting any tier, verify:

- **SDK version**: `hailo_platform.genai` (`Speech2Text`, `LLM`, `VLM` classes) requires HailoRT ≥ X.Y. Confirm installed version on the Pi with `pip show hailo-platform` or `hailortcli fw-control identify`.
- **Model files**: Direct Python API uses HEF files. LLM HEF is symlinked from hailo-ollama's blob storage (`~/.local/share/hailo-ollama/models/blob/`). VLM HEF is in `./models/`.

---

## Modernization Tiers

### Tier 1 — Safe Wins (low risk, high impact) ✅ COMPLETE

- ~~Capture mic at 16kHz directly instead of 48kHz → FFmpeg → 16kHz~~ **DROPPED** — mic must stay at 48kHz because OpenWakeWord requires integer decimation from 48kHz→16kHz, and USB mic hardware 16kHz support is unverified. FFmpeg cost (~100-300ms) is not the bottleneck. STT now skips FFmpeg when input is already 16kHz mono WAV (e.g. pre-converted files).
- ✅ Replace aplay subprocess with persistent `sounddevice.OutputStream` (commit `ac27405`)
- ✅ Replace Piper subprocess shell pipe with Piper Python bindings, subprocess fallback (commit `ac27405`)

> **Rollback**: All Tier 1 changes are isolated to audio I/O. Revert by restoring the original subprocess calls — no model or API changes involved.

### Tier 2 — Architecture Shift (medium risk, high impact) ✅ COMPLETE

- ✅ Replace hailo-ollama HTTP API with `hailo_platform.genai.LLM` direct Python API
- ✅ Reimplement conversation history — LLM API natively accepts `[{"role": ..., "content": ...}]` message lists (no manual ChatML formatting needed)
- ✅ Drop model routing (FAST_LLM_MODEL vs LLM_MODEL — were already the same model)
- ✅ Use `llm.save_context()` / `llm.load_context()` for system prompt KV cache — system prompt is processed once at startup, cached, and restored before each turn
- ✅ VLM runs in a child process (`multiprocessing.Process`) — eliminates `pkill hailo-ollama` hack
- ✅ All consumers updated: agent_hailo.py, web_app.py, cli_chat.py
- ~~Use `VDevice(group_id="SHARED")` to run LLM + VLM concurrently~~ **NOT POSSIBLE** — Hailo-10H NPU cannot hold Qwen2.5-1.5B (2.3GB) + Qwen2-VL-2B (2.3GB) simultaneously. `HAILO_INTERNAL_FAILURE(8)` on any attempt. The `SHARED` group_id works for smaller model pairs (e.g. Whisper + LLM in hailo-apps voice_assistant) but not two 1.5B+ generative models.
- Token-level TTS callback and `generation_id` interruption → deferred to Tier 3 (requires async queue architecture)

**VLM swap pattern**: Main process releases LLM + VDevice → forks child process → child creates its own VDevice + VLM → inference → child exits (OS guarantees clean NPU release) → parent reloads LLM + re-caches system prompt. Total swap: ~16s (vs 5-15s with old pkill/restart, but no crash risk).

**Benchmark results** (direct NPU vs hailo-ollama HTTP):
- Time to first token: **0.37s vs 0.55s** (33% faster)
- Total generation time: roughly equal (~10s avg, varies by response length)
- LLM init: 6.6s (once at startup)

> **Rollback**: `hailo-ollama` is still installed. Re-enable with `sudo systemctl enable bmo-ollama && sudo systemctl start bmo-ollama`, then revert `core/llm.py`, `core/config.py`, `agent_hailo.py`, `web_app.py`, `cli_chat.py` to pre-Tier-2 state from git.

### Tier 3 — Full Pipeline Rewrite (high risk, highest impact) — IN PROGRESS

#### Phase 3A — STT on NPU ✅ COMPLETE
- ✅ `Speech2Text` on NPU via shared VDevice (`group_id="SHARED"`) — Whisper (125MB) coexists with LLM (2.3GB)
- ✅ 7.3x faster transcription: **0.26s** vs 1.91s (whisper.cpp CPU)
- ✅ CPU fallback if Whisper HEF is missing or init fails
- ✅ STT released/reloaded during VLM subprocess swap
- ✅ PCIe Gen 3 confirmed active (8GT/s) — no timeout issues on HailoRT 5.1.1
- ✅ Whisper HEF: `./models/Whisper-Base.hef` (downloaded from `dev-public.hailo.ai/v5.1.1/blob/`)

#### Phase 3C — Eliminate subprocesses ⚠️ PARTIAL
- ✅ `shutil.which()` replaces `subprocess.run(['which', ...])` for camera detection
- ❌ Sound effects stay on `aplay` subprocess — sharing the persistent AudioPlayer stream between TTS and sound effects causes ALSA assertion crashes and audio stuttering. `aplay` is reliable for fire-and-forget sound effects and isn't on the latency-critical path.
- ✅ Wake word suppression during SPEAKING/JAMMING states (prevents BMO's own audio from triggering false detections)

#### Phase 3B — webrtcvad for endpoint detection (not started)
- Add `webrtcvad` for proper silence/speech endpoint detection
- Changes the recording state machine in `record_audio()` — not a drop-in
- Note: chunk-based silence detection works well enough; time-based approach was tested and reverted (background noise kept resetting the silence timer)

#### Phase 3D — generation_id + TTS interruption (not started)
- Add `generation_id` for TTS interruption support
- Adopt token-level TTS callback pattern (queue sentences during streaming)

#### Phase 3E — Async pipeline (not started)
- Adopt `VoiceInteractionManager` pattern from hailo-apps
- Proper async architecture with `queue.Queue` between stages

> **Rollback**: Tier 3 is incremental — each phase can be rolled back independently. STT on NPU falls back to whisper.cpp CPU automatically if the HEF is missing.

---

## Critical Consideration: hailo-ollama vs Direct Python API

The hailo-apps use `hailo_platform.genai.LLM` which is **mutually exclusive** with `hailo-ollama`. ~~Moving to the direct Python API means:~~ **DONE (Tier 2).**

- ✅ Models come from HEF files (LLM HEF symlinked from hailo-ollama blob storage)
- ✅ No more Ollama model management (`ollama pull`, etc.) — hailo-ollama kept installed for rollback only
- ✅ VLM runs via direct Python API in a subprocess (eliminates `pkill hailo-ollama` hack)
- ✅ Conversation history uses native message list format (API handles ChatML internally)

---

## Hailo-Apps API Reference

### Speech2Text (NPU)

```python
from hailo_platform import VDevice
from hailo_platform.genai import Speech2Text, Speech2TextTask

vdevice = VDevice()

speech2text = Speech2Text(vdevice, "/path/to/Whisper-Base.hef")
segments = speech2text.generate_all_segments(
    audio_data,                          # float32 numpy array, mono, 16kHz
    task=Speech2TextTask.TRANSCRIBE,
    language="en",
    timeout_ms=15000
)
text = " ".join(seg.text for seg in segments)
```

### LLM (NPU, Streaming)

```python
from hailo_platform import VDevice
from hailo_platform.genai import LLM

vdevice = VDevice()
llm = LLM(vdevice, "/path/to/Qwen2.5-1.5B-Instruct.hef")

prompt = [
    {"role": "system", "content": "You are BMO."},
    {"role": "user", "content": "Hello!"}
]

# Streaming (token by token)
with llm.generate(prompt=prompt, temperature=0.4, max_generated_tokens=180) as gen:
    for token in gen:
        print(token, end="", flush=True)

# Blocking
response = llm.generate_all(prompt=prompt, temperature=0.4, max_generated_tokens=180)

# Context caching (persist system prompt across turns)
context_data = llm.save_context()
llm.load_context(context_data)
```

### VLM (NPU)

```python
from hailo_platform import VDevice
from hailo_platform.genai import VLM

vdevice = VDevice()
vlm = VLM(vdevice, "/path/to/Qwen2-VL-2B-Instruct.hef")

prompt = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image"}
    ]}
]

with vlm.generate(prompt=prompt, frames=[numpy_image], max_generated_tokens=200) as gen:
    for token in gen:
        print(token, end="", flush=True)
```

### NPU Model Concurrency

The Hailo-10H has limited on-device memory. Smaller model pairs (e.g. Whisper + LLM) can share a VDevice via `group_id="SHARED"`:

```python
from hailo_platform import VDevice

params = VDevice.create_params()
params.group_id = "SHARED"
vdevice = VDevice(params)

# Both models share the same NPU device
speech2text = Speech2Text(vdevice, whisper_hef)
llm = LLM(vdevice, llm_hef)
```

Larger model pairs (e.g. LLM 1.5B + VLM 2B) **cannot coexist** — use sequential access with separate VDevices (BMO uses a child process for VLM).

---

## Current BMO Latency Profile

| Stage | Typical Latency | Notes |
|-------|----------------|-------|
| Wake word detection | Real-time | OWW runs continuously on CPU (suppressed during SPEAKING/JAMMING) |
| Audio recording | 1-30s | User-controlled; 1.5s grace period; chunk-based silence detection |
| ~~FFmpeg conversion~~ | ~~100-300ms~~ | **Eliminated** — NPU Speech2Text handles resampling in-process (Tier 3A) |
| STT transcription (NPU) | **~0.26s** | Down from ~1.9s with whisper.cpp CPU (Tier 3A, 7.3x faster) |
| Pre-LLM keyword matching | <1ms | Pure Python string ops |
| DuckDuckGo search (if triggered) | 1-3s | Network I/O, blocks before LLM |
| LLM first token (direct NPU) | **~0.37s** | Down from ~0.55s with hailo-ollama (Tier 2) |
| LLM streaming (per sentence) | ~0.5-2s | Depends on sentence length |
| Piper TTS synthesis | ~0.5-1s | CPU, per sentence (now via piper-tts library when available) |
| ~~aplay playback~~ | ~~Real-time~~ | **Eliminated** for TTS — persistent sounddevice OutputStream (Tier 1). Sound effects still use aplay. |
| VLM (analyze_image) | ~20-30s | Subprocess swap: release STT+LLM → fork → VLM init + inference → exit → reload LLM+STT |

### Startup Profile (measured on Pi 5)

| Component | Time | Notes |
|-----------|------|-------|
| Audio subsystem | ~0.03s | AudioPlayer + Piper init |
| LLM (NPU) | ~12.5s | VDevice + HEF load + system prompt cache |
| STT (NPU) | ~1.2s | Speech2Text on shared VDevice |
| Wake word (OWW) | ~0.5s | ONNX model load |
| **Total startup** | **~14.2s** | |
