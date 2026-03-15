#!/usr/bin/env python3
"""Benchmark STT: Hailo NPU Speech2Text vs whisper.cpp CPU subprocess.

Tests three scenarios:
  1. whisper.cpp on CPU (current approach)
  2. Speech2Text on NPU (standalone — no LLM loaded)
  3. Speech2Text on NPU + LLM coexisting via group_id="SHARED"

Usage:
    # Record a test clip first (or use an existing WAV)
    python benchmark_stt.py                      # uses default test recording
    python benchmark_stt.py --audio test.wav      # use a specific file
    python benchmark_stt.py --record              # record a 5s clip first
    python benchmark_stt.py --record --duration 8 # record 8s clip
"""
import argparse
import json
import os
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark_stt")

WHISPER_CMD = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "./models/ggml-base.en.bin"
TEST_AUDIO = "benchmark_stt_test.wav"


def record_test_audio(duration=5, output=TEST_AUDIO):
    """Record a test audio clip from the default mic."""
    import sounddevice as sd
    import scipy.io.wavfile as wav
    import numpy as np

    sample_rate = 16000
    print(f"\n  Recording {duration}s of audio at {sample_rate}Hz ...")
    print("  Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype='int16', blocking=True)
    wav.write(output, sample_rate, audio)
    print(f"  Saved to {output}\n")
    return output


def prepare_audio_for_whisper_cpp(audio_path):
    """Convert audio to 16kHz mono WAV for whisper.cpp (mimics core/stt.py)."""
    out_path = audio_path + "_16k.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        out_path
    ], capture_output=True, timeout=10)
    return out_path


def prepare_audio_for_npu(audio_path):
    """Load audio as float32 numpy array at 16kHz for Speech2Text."""
    import numpy as np
    import scipy.io.wavfile as wav

    sample_rate, audio = wav.read(audio_path)

    # Convert to float32 [-1.0, 1.0]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        from scipy.signal import resample
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = resample(audio, num_samples).astype(np.float32)

    return audio


def benchmark_whisper_cpp(audio_path, runs=3):
    """Benchmark whisper.cpp CPU subprocess (current approach)."""
    if not os.path.exists(WHISPER_CMD):
        print("  SKIP: whisper.cpp not found")
        return None

    prepped = prepare_audio_for_whisper_cpp(audio_path)
    results = []

    for i in range(runs):
        t0 = time.time()
        result = subprocess.run(
            [WHISPER_CMD, "-m", WHISPER_MODEL, "-f", prepped, "-nt"],
            capture_output=True, text=True, timeout=30
        )
        elapsed = time.time() - t0
        text = result.stdout.strip()
        results.append({"run": i + 1, "time_s": round(elapsed, 3), "text": text})
        print(f"    Run {i+1}: {elapsed:.2f}s — {text[:80]}")

    # Cleanup
    try:
        os.remove(prepped)
    except OSError:
        pass

    avg = sum(r["time_s"] for r in results) / len(results)
    return {"method": "whisper_cpp_cpu", "avg_s": round(avg, 3), "runs": results}


def benchmark_npu_standalone(audio_path, runs=3):
    """Benchmark Speech2Text on NPU (no LLM loaded)."""
    try:
        from hailo_platform import VDevice
        from hailo_platform.genai import Speech2Text, Speech2TextTask
    except ImportError:
        print("  SKIP: hailo_platform.genai.Speech2Text not available")
        return None

    # Check for Whisper HEF
    whisper_hefs = [
        "./models/Whisper-Base.hef",
        "./models/whisper-base.hef",
        "/usr/local/hailo/resources/models/Whisper-Base.hef",
    ]
    whisper_hef = None
    for path in whisper_hefs:
        if os.path.exists(path):
            whisper_hef = path
            break

    if whisper_hef is None:
        # Search hailo-ollama blob storage
        manifest_dirs = [
            os.path.expanduser("~/.local/share/hailo-ollama/models/manifests"),
        ]
        for mdir in manifest_dirs:
            for root, dirs, files in os.walk(mdir):
                if "whisper" in root.lower():
                    for f in files:
                        if f == "manifest.json":
                            try:
                                with open(os.path.join(root, f)) as mf:
                                    manifest = json.load(mf)
                                    hef_hash = manifest.get("hef_h10h", "")
                                    blob_path = os.path.join(
                                        os.path.expanduser("~/.local/share/hailo-ollama/models/blob"),
                                        f"sha256_{hef_hash}"
                                    )
                                    if os.path.exists(blob_path):
                                        whisper_hef = blob_path
                            except Exception:
                                pass

    if whisper_hef is None:
        print("  SKIP: No Whisper HEF found")
        print("  Searched: " + ", ".join(whisper_hefs))
        print("  Also searched hailo-ollama blob storage")
        print("  To download: hailo-ollama pull whisper-base (if available)")
        return None

    print(f"  Using Whisper HEF: {whisper_hef}")
    audio_data = prepare_audio_for_npu(audio_path)
    print(f"  Audio: {len(audio_data)} samples ({len(audio_data)/16000:.1f}s)")

    # Init
    t0 = time.time()
    vdevice = VDevice()
    s2t = Speech2Text(vdevice, whisper_hef)
    init_time = time.time() - t0
    print(f"  Init: {init_time:.2f}s")

    results = []
    for i in range(runs):
        t0 = time.time()
        try:
            segments = s2t.generate_all_segments(
                audio_data,
                task=Speech2TextTask.TRANSCRIBE,
                language="en",
                timeout_ms=30000,
            )
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            text = f"ERROR: {e}"
        elapsed = time.time() - t0
        results.append({"run": i + 1, "time_s": round(elapsed, 3), "text": text})
        print(f"    Run {i+1}: {elapsed:.2f}s — {text[:80]}")

    s2t.release()
    vdevice.release()

    avg = sum(r["time_s"] for r in results) / len(results)
    return {
        "method": "speech2text_npu_standalone",
        "whisper_hef": whisper_hef,
        "init_s": round(init_time, 3),
        "avg_s": round(avg, 3),
        "runs": results,
    }


def benchmark_npu_with_llm(audio_path, runs=3):
    """Benchmark Speech2Text on NPU with LLM co-resident (shared VDevice)."""
    try:
        from hailo_platform import VDevice
        from hailo_platform.genai import Speech2Text, Speech2TextTask, LLM
    except ImportError:
        print("  SKIP: hailo_platform.genai not available")
        return None

    from core.config import LLM_HEF_PATH

    # Find Whisper HEF (same search as standalone)
    whisper_hefs = [
        "./models/Whisper-Base.hef",
        "./models/whisper-base.hef",
        "/usr/local/hailo/resources/models/Whisper-Base.hef",
    ]
    whisper_hef = None
    for path in whisper_hefs:
        if os.path.exists(path):
            whisper_hef = path
            break

    if whisper_hef is None:
        manifest_dirs = [
            os.path.expanduser("~/.local/share/hailo-ollama/models/manifests"),
        ]
        for mdir in manifest_dirs:
            for root, dirs, files in os.walk(mdir):
                if "whisper" in root.lower():
                    for f in files:
                        if f == "manifest.json":
                            try:
                                with open(os.path.join(root, f)) as mf:
                                    manifest = json.load(mf)
                                    hef_hash = manifest.get("hef_h10h", "")
                                    blob_path = os.path.join(
                                        os.path.expanduser("~/.local/share/hailo-ollama/models/blob"),
                                        f"sha256_{hef_hash}"
                                    )
                                    if os.path.exists(blob_path):
                                        whisper_hef = blob_path
                            except Exception:
                                pass

    if whisper_hef is None:
        print("  SKIP: No Whisper HEF found")
        return None

    llm_hef = LLM_HEF_PATH
    if not os.path.isabs(llm_hef):
        llm_hef = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), llm_hef)
    if not os.path.exists(llm_hef):
        # Try resolving from project root
        llm_hef = os.path.join(os.getcwd(), LLM_HEF_PATH.lstrip("./"))
    if not os.path.exists(llm_hef):
        print(f"  SKIP: LLM HEF not found at {llm_hef}")
        return None

    print(f"  Whisper HEF: {whisper_hef}")
    print(f"  LLM HEF: {llm_hef}")
    audio_data = prepare_audio_for_npu(audio_path)

    # Init shared VDevice + both models
    t0 = time.time()
    try:
        params = VDevice.create_params()
        params.group_id = "SHARED"
        vdevice = VDevice(params)

        llm = LLM(vdevice, llm_hef)
        print(f"  LLM loaded: {time.time()-t0:.2f}s")

        s2t = Speech2Text(vdevice, whisper_hef)
        init_time = time.time() - t0
        print(f"  LLM + Speech2Text loaded: {init_time:.2f}s")
    except Exception as e:
        print(f"  FAILED to load both models: {e}")
        print("  (This confirms whether LLM + Whisper can coexist on the NPU)")
        return {"method": "speech2text_npu_with_llm", "error": str(e)}

    results = []
    for i in range(runs):
        t0 = time.time()
        try:
            segments = s2t.generate_all_segments(
                audio_data,
                task=Speech2TextTask.TRANSCRIBE,
                language="en",
                timeout_ms=30000,
            )
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            text = f"ERROR: {e}"
        elapsed = time.time() - t0
        results.append({"run": i + 1, "time_s": round(elapsed, 3), "text": text})
        print(f"    Run {i+1}: {elapsed:.2f}s — {text[:80]}")

    # Also do a quick LLM test to confirm it still works
    print("  Verifying LLM still works with Speech2Text loaded ...")
    t0 = time.time()
    try:
        resp = llm.generate_all(
            prompt=[{"role": "user", "content": "Say hello in one word."}],
            temperature=0.4, max_generated_tokens=10,
        )
        for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            resp = resp.replace(tok, "")
        print(f"    LLM response: {resp.strip()!r} ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"    LLM FAILED: {e}")

    s2t.release()
    llm.release()
    vdevice.release()

    avg = sum(r["time_s"] for r in results) / len(results)
    return {
        "method": "speech2text_npu_with_llm",
        "whisper_hef": whisper_hef,
        "init_s": round(init_time, 3),
        "avg_s": round(avg, 3),
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark STT methods")
    parser.add_argument("--audio", default=TEST_AUDIO, help="Path to test WAV file")
    parser.add_argument("--record", action="store_true", help="Record a test clip first")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per method")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip whisper.cpp CPU benchmark")
    parser.add_argument("--skip-npu", action="store_true", help="Skip NPU standalone benchmark")
    parser.add_argument("--skip-shared", action="store_true", help="Skip NPU+LLM shared benchmark")
    args = parser.parse_args()

    if args.record:
        args.audio = record_test_audio(args.duration, args.audio)

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        print("Use --record to record a test clip, or --audio to specify a file")
        sys.exit(1)

    all_results = {}

    # 1. whisper.cpp CPU
    if not args.skip_cpu:
        print("\n--- whisper.cpp on CPU (current approach) ---")
        result = benchmark_whisper_cpp(args.audio, runs=args.runs)
        if result:
            all_results["whisper_cpp_cpu"] = result

    # 2. Speech2Text NPU standalone
    if not args.skip_npu:
        print("\n--- Speech2Text on NPU (standalone) ---")
        result = benchmark_npu_standalone(args.audio, runs=args.runs)
        if result:
            all_results["speech2text_npu_standalone"] = result

    # 3. Speech2Text NPU + LLM shared
    if not args.skip_shared:
        print("\n--- Speech2Text on NPU + LLM (shared VDevice) ---")
        result = benchmark_npu_with_llm(args.audio, runs=args.runs)
        if result:
            all_results["speech2text_npu_with_llm"] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, data in all_results.items():
        if "error" in data:
            print(f"  {data['method']:40s}  FAILED: {data['error']}")
        elif "avg_s" in data:
            init = f"  init={data['init_s']:.2f}s" if "init_s" in data else ""
            print(f"  {data['method']:40s}  avg={data['avg_s']:.2f}s{init}")
    print()

    # Save
    outfile = "benchmark_stt_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {outfile}")


if __name__ == "__main__":
    main()
