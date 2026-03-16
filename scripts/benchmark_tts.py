#!/usr/bin/env python3
"""Benchmark: sentence-by-sentence TTS vs full-response TTS."""

import time
import sys
sys.path.insert(0, ".")

from core.tts import PiperSynthesizer

synth = PiperSynthesizer()

# Simulate a typical 2-sentence LLM response
chunks = [
    "I'm sorry, but I haven't seen any movie called Rod Cats.",
    "Could you please provide more information or a specific title so that I can assist you better?",
]
full_text = " ".join(chunks)

# Warm up (first call is slower due to process startup cache)
synth.synthesize("Hello.")

# Benchmark 1: Sentence-by-sentence synthesis
t0 = time.time()
pcm_parts = []
for c in chunks:
    pcm_parts.append(synth.synthesize(c))
t_sentence = time.time() - t0
total_sentence_bytes = sum(len(p) for p in pcm_parts)

# Benchmark 2: Full response synthesis
t0 = time.time()
pcm_full = synth.synthesize(full_text)
t_full = time.time() - t0

# Benchmark 3: 4-sentence response
chunks_long = [
    "BMO loves to play games with friends!",
    "Today BMO learned about a really cool new algorithm.",
    "It made BMO think about how robots process information.",
    "BMO hopes you are having a wonderful day too!",
]
full_long = " ".join(chunks_long)

t0 = time.time()
for c in chunks_long:
    synth.synthesize(c)
t_4sent = time.time() - t0

t0 = time.time()
synth.synthesize(full_long)
t_4full = time.time() - t0

print("=== 2-sentence response ===")
print(f"  Sentence-by-sentence: {t_sentence:.3f}s")
print(f"  Full response:        {t_full:.3f}s")
print(f"  Overhead:             {t_sentence - t_full:.3f}s ({(t_sentence/t_full - 1)*100:.0f}% slower)")
print(f"  Audio duration:       {total_sentence_bytes / 2 / 22050:.2f}s vs {len(pcm_full) / 2 / 22050:.2f}s")
print()
print("=== 4-sentence response ===")
print(f"  Sentence-by-sentence: {t_4sent:.3f}s")
print(f"  Full response:        {t_4full:.3f}s")
print(f"  Overhead:             {t_4sent - t_4full:.3f}s ({(t_4sent/t_4full - 1)*100:.0f}% slower)")
print()
print("=== Per-call subprocess overhead ===")
print(f"  ~{((t_sentence - t_full) / len(chunks)) * 1000:.0f}ms per extra Piper subprocess call")
