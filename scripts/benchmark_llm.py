#!/usr/bin/env python3
"""Benchmark LLM inference: direct NPU API vs hailo-ollama HTTP.

Run with hailo-ollama stopped (direct API mode):
    python benchmark_llm.py

Run with hailo-ollama running (HTTP mode, for comparison):
    python benchmark_llm.py --http

Results are printed as a table and saved to benchmark_results.json.
"""
import argparse
import json
import time
import sys

PROMPTS = [
    # Short / simple
    {"label": "greeting", "text": "Hello BMO!"},
    {"label": "short_fact", "text": "What color is the sky?"},
    # Medium
    {"label": "question", "text": "What is the capital of France and why is it famous?"},
    {"label": "creative", "text": "Tell me a short joke about robots."},
    # Longer / complex
    {"label": "explain", "text": "Explain how a rainbow forms in two or three sentences."},
    {"label": "story", "text": "Write a very short story about a robot who finds a flower."},
]

SYSTEM_PROMPT = (
    "You are BMO, a sweet and cheerful little robot friend. "
    "Keep your answers short — two to four sentences."
)


def benchmark_direct(prompts, warmup=True):
    """Benchmark using hailo_platform.genai.LLM direct API."""
    from core.llm import init_llm, _get_llm

    print("Initialising direct NPU LLM ...")
    t0 = time.time()
    init_llm()
    init_time = time.time() - t0
    print(f"  LLM init: {init_time:.2f}s")

    llm = _get_llm()

    if warmup:
        # Warm up with a throwaway query
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Hi"}]
        llm.generate_all(prompt=messages, temperature=0.4, max_generated_tokens=20)
        print("  Warmup done\n")

    results = []
    for p in prompts:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p["text"]}]

        # Non-streaming
        t0 = time.time()
        content = llm.generate_all(prompt=messages, temperature=0.4, max_generated_tokens=180)
        total_time = time.time() - t0

        # Strip stop tokens
        for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            content = content.replace(tok, "")
        content = content.strip()

        tokens_approx = len(content.split())

        # Streaming (measure time-to-first-token)
        t0 = time.time()
        first_token_time = None
        stream_content = ""
        with llm.generate(prompt=messages, temperature=0.4, max_generated_tokens=180) as gen:
            for token in gen:
                if first_token_time is None and token.strip():
                    first_token_time = time.time() - t0
                stream_content += token
        stream_total = time.time() - t0

        result = {
            "label": p["label"],
            "prompt": p["text"],
            "mode": "direct_npu",
            "total_s": round(total_time, 3),
            "first_token_s": round(first_token_time, 3) if first_token_time else None,
            "stream_total_s": round(stream_total, 3),
            "response_words": tokens_approx,
            "response": content[:120],
        }
        results.append(result)
        print(f"  {p['label']:15s}  total={total_time:.2f}s  ttft={first_token_time:.2f}s  "
              f"stream={stream_total:.2f}s  words={tokens_approx:3d}  {content[:60]}")

    return {
        "mode": "direct_npu",
        "init_time_s": round(init_time, 2),
        "results": results,
    }


def benchmark_http(prompts, warmup=True):
    """Benchmark using hailo-ollama HTTP API (for comparison baseline)."""
    import requests

    LLM_URL = "http://127.0.0.1:8000/api/chat"
    MODEL = "qwen2.5-instruct:1.5b"

    # Check if hailo-ollama is running
    try:
        r = requests.get("http://127.0.0.1:8000/api/tags", timeout=5)
        if r.status_code != 200:
            print("ERROR: hailo-ollama is not responding. Start it first.")
            sys.exit(1)
    except requests.ConnectionError:
        print("ERROR: hailo-ollama is not running on port 8000.")
        print("Start it with: OLLAMA_HOST=0.0.0.0:8000 hailo-ollama serve")
        sys.exit(1)

    print("Using hailo-ollama HTTP API ...")

    if warmup:
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 20},
        }
        requests.post(LLM_URL, json=payload, timeout=60)
        print("  Warmup done\n")

    results = []
    for p in prompts:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p["text"]}]

        # Non-streaming
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 180},
        }
        t0 = time.time()
        resp = requests.post(LLM_URL, json=payload, timeout=180)
        total_time = time.time() - t0
        content = resp.json().get("message", {}).get("content", "").strip()
        tokens_approx = len(content.split())

        # Streaming (time-to-first-token)
        payload["stream"] = True
        t0 = time.time()
        first_token_time = None
        stream_content = ""
        with requests.post(LLM_URL, json=payload, stream=True, timeout=180) as resp:
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if first_token_time is None and chunk.strip():
                        first_token_time = time.time() - t0
                    stream_content += chunk
        stream_total = time.time() - t0

        result = {
            "label": p["label"],
            "prompt": p["text"],
            "mode": "http_ollama",
            "total_s": round(total_time, 3),
            "first_token_s": round(first_token_time, 3) if first_token_time else None,
            "stream_total_s": round(stream_total, 3),
            "response_words": tokens_approx,
            "response": content[:120],
        }
        results.append(result)
        print(f"  {p['label']:15s}  total={total_time:.2f}s  ttft={first_token_time:.2f}s  "
              f"stream={stream_total:.2f}s  words={tokens_approx:3d}  {content[:60]}")

    return {
        "mode": "http_ollama",
        "results": results,
    }


def print_comparison(direct, http):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON: Direct NPU API vs hailo-ollama HTTP")
    print("=" * 80)
    print(f"{'Prompt':<15s}  {'Total (direct)':>14s}  {'Total (HTTP)':>13s}  {'Speedup':>8s}  "
          f"{'TTFT (direct)':>14s}  {'TTFT (HTTP)':>12s}")
    print("-" * 80)

    for d, h in zip(direct["results"], http["results"]):
        speedup = h["total_s"] / d["total_s"] if d["total_s"] > 0 else 0
        d_ttft = f"{d['first_token_s']:.2f}s" if d["first_token_s"] else "N/A"
        h_ttft = f"{h['first_token_s']:.2f}s" if h["first_token_s"] else "N/A"
        print(f"  {d['label']:<15s}  {d['total_s']:>10.2f}s    {h['total_s']:>10.2f}s   "
              f"{speedup:>5.2f}x    {d_ttft:>10s}    {h_ttft:>10s}")

    # Averages
    avg_d = sum(r["total_s"] for r in direct["results"]) / len(direct["results"])
    avg_h = sum(r["total_s"] for r in http["results"]) / len(http["results"])
    avg_speedup = avg_h / avg_d if avg_d > 0 else 0
    print("-" * 80)
    print(f"  {'AVERAGE':<15s}  {avg_d:>10.2f}s    {avg_h:>10.2f}s   {avg_speedup:>5.2f}x")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference")
    parser.add_argument("--http", action="store_true",
                        help="Benchmark hailo-ollama HTTP API instead of direct NPU")
    parser.add_argument("--both", action="store_true",
                        help="Run both modes (requires hailo-ollama running for HTTP)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup query")
    args = parser.parse_args()

    warmup = not args.no_warmup
    all_results = {}

    if args.both:
        # Direct first
        print("\n--- Direct NPU API ---")
        direct = benchmark_direct(PROMPTS, warmup=warmup)
        all_results["direct_npu"] = direct

        # Need to release LLM before starting hailo-ollama
        print("\nNOTE: For HTTP benchmark, start hailo-ollama manually and re-run with --http")
        print("(Can't run both in same process — NPU conflict)")

    elif args.http:
        print("\n--- hailo-ollama HTTP API ---")
        http = benchmark_http(PROMPTS, warmup=warmup)
        all_results["http_ollama"] = http

    else:
        print("\n--- Direct NPU API ---")
        direct = benchmark_direct(PROMPTS, warmup=warmup)
        all_results["direct_npu"] = direct

    # Save results
    outfile = "benchmark_results.json"
    # Merge with existing results if present
    try:
        with open(outfile) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    existing.update(all_results)

    with open(outfile, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # If both modes have results, print comparison
    if "direct_npu" in existing and "http_ollama" in existing:
        print_comparison(existing["direct_npu"], existing["http_ollama"])


if __name__ == "__main__":
    main()
