#!/usr/bin/env python3
"""
BMO Diagnostic Test Script — test_music_images.py
Run this on the Pi to verify each component of the music and image pipeline.

Usage:
    python3 test_music_images.py
"""
import os
import sys
import json
import subprocess
import urllib.parse
import urllib.request
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─── Test 1: Music Keyword Matching ───────────────────────────
def test_music_keywords():
    section("1. Music Keyword Matching")
    from core.llm import _MUSIC_KEYWORDS

    test_inputs = [
        ("play me a song", True),
        ("Play me a song.", True),
        ("BMO play music", True),
        ("can you sing a song", True),
        ("sing me a song please", True),
        ("play some music", True),
        ("sing for me", True),
        ("jam out", True),
        ("what time is it", False),
        ("how are you", False),
        ("tell me a joke", False),
    ]

    for text, expected in test_inputs:
        lower = text.lower()
        matched = any(kw in lower for kw in _MUSIC_KEYWORDS)
        status = PASS if matched == expected else FAIL
        print(f"  {status} '{text}' → matched={matched} (expected={expected})")

# ─── Test 2: Image Keyword Matching ──────────────────────────
def test_image_keywords():
    section("2. Image Keyword Matching")
    from core.llm import _DISPLAY_IMAGE_KEYWORDS

    test_inputs = [
        ("show me a picture of a cat", True),
        ("Show me a picture of a dog.", True),
        ("draw me a sunset", True),
        ("picture of a mountain", True),
        ("image of a robot", True),
        ("generate an image of space", True),
        ("what does a cat look like", False),
        ("how are you", False),
    ]

    for text, expected in test_inputs:
        lower = text.lower()
        matched = any(kw in lower for kw in _DISPLAY_IMAGE_KEYWORDS)
        status = PASS if matched == expected else FAIL
        print(f"  {status} '{text}' → matched={matched} (expected={expected})")

# ─── Test 3: Build Display Image Action ──────────────────────
def test_build_image_action():
    section("3. Build Display Image Action")
    from core.llm import _build_display_image_action

    test_inputs = [
        "show me a picture of a cat",
        "draw me a sunset over the ocean",
        "picture of BMO",
    ]

    for text in test_inputs:
        result = _build_display_image_action(text)
        try:
            data = json.loads(result)
            has_action = data.get("action") == "display_image"
            has_url = bool(data.get("image_url"))
            url = data.get("image_url", "")
            status = PASS if has_action and has_url else FAIL
            print(f"  {status} '{text}'")
            print(f"       → URL: {url}")
        except Exception as e:
            print(f"  {FAIL} '{text}' → JSON parse error: {e}")

# ─── Test 4: Brain.think() Action Detection ──────────────────
def test_brain_think():
    section("4. Brain.think() Action Detection")
    from core.llm import Brain

    brain = Brain()

    # Test music
    result = brain.think("play me a song")
    try:
        data = json.loads(result)
        status = PASS if data.get("action") == "play_music" else FAIL
        print(f"  {status} think('play me a song') → {result}")
    except:
        print(f"  {FAIL} think('play me a song') → '{result}' (not valid JSON!)")

    # Test image
    brain2 = Brain()
    result = brain2.think("show me a picture of a cat")
    try:
        data = json.loads(result)
        status = PASS if data.get("action") == "display_image" and data.get("image_url") else FAIL
        print(f"  {status} think('show me a picture of a cat') → action={data.get('action')}")
        if data.get("image_url"):
            print(f"       → URL: {data['image_url']}")
    except:
        print(f"  {FAIL} think('show me a picture of a cat') → '{result}' (not valid JSON!)")

# ─── Test 5: Brain.stream_think() Action Detection ───────────
def test_brain_stream_think():
    section("5. Brain.stream_think() Action Detection")
    from core.llm import Brain

    brain = Brain()
    chunks = list(brain.stream_think("sing me a song"))
    combined = "".join(chunks)
    print(f"  Chunks received: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"    Chunk {i}: '{c[:80]}'")

    try:
        data = json.loads(combined)
        status = PASS if data.get("action") == "play_music" else FAIL
        print(f"  {status} stream_think('sing me a song') → action={data.get('action')}")
    except:
        print(f"  {FAIL} stream_think('sing me a song') → '{combined[:100]}' (not valid JSON!)")

# ─── Test 6: Music Files ─────────────────────────────────────
def test_music_files():
    section("6. Music Files (sounds/music/)")
    music_dir = os.path.join("sounds", "music")

    if not os.path.exists(music_dir):
        print(f"  {FAIL} Directory not found: {music_dir}")
        return

    files = [f for f in os.listdir(music_dir) if f.lower().endswith('.wav')]
    if not files:
        print(f"  {FAIL} No .wav files found in {music_dir}")
        return

    print(f"  Found {len(files)} WAV files:")
    for f in sorted(files):
        path = os.path.join(music_dir, f)
        size = os.path.getsize(path)
        duration = size / (44100 * 2)  # rough estimate for 16-bit mono
        if size < 10000:
            status = WARN
            note = " (suspiciously small!)"
        else:
            status = PASS
            note = ""
        print(f"  {status} {f}: {size/1024:.0f}KB (~{duration:.1f}s){note}")

# ─── Test 7: aplay Playback ─────────────────────────────────
def test_aplay():
    section("7. aplay Playback Test")
    from core.config import ALSA_DEVICE

    music_dir = os.path.join("sounds", "music")
    files = [f for f in os.listdir(music_dir) if f.lower().endswith('.wav')] if os.path.exists(music_dir) else []

    if not files:
        print(f"  {FAIL} No music files to test")
        return

    # Test with the first file
    test_file = os.path.join(music_dir, files[0])
    print(f"  Testing: aplay -D {ALSA_DEVICE} -q {test_file}")
    try:
        result = subprocess.run(
            ['aplay', '-D', ALSA_DEVICE, '-q', test_file],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"  {PASS} aplay returned successfully")
        else:
            print(f"  {FAIL} aplay returned code {result.returncode}")
            print(f"       stderr: {result.stderr.strip()}")
    except FileNotFoundError:
        print(f"  {FAIL} 'aplay' command not found (not on a Pi?)")
    except subprocess.TimeoutExpired:
        print(f"  {WARN} aplay timed out (might be playing but took too long)")
    except Exception as e:
        print(f"  {FAIL} aplay error: {e}")

# ─── Test 8: Pollinations Image Download ─────────────────────
def test_pollinations():
    section("8. Pollinations Image Download Test")
    url = "https://image.pollinations.ai/prompt/cute%20robot?width=256&height=256&nologo=true"
    print(f"  Testing URL: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read()
            content_type = response.headers.get('Content-Type', 'unknown')
            print(f"  {PASS} Downloaded {len(data)} bytes (Content-Type: {content_type})")
            if len(data) < 1000:
                print(f"  {WARN} Response seems too small — may not be a real image")
    except urllib.error.URLError as e:
        print(f"  {FAIL} URL error: {e}")
    except Exception as e:
        print(f"  {FAIL} Download error: {e}")

    # Also test the gen.pollinations.ai endpoint
    url2 = "https://gen.pollinations.ai/image/cute%20robot?width=256&height=256&nologo=true"
    print(f"\n  Testing URL: {url2}")
    try:
        req = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read()
            content_type = response.headers.get('Content-Type', 'unknown')
            print(f"  {PASS} Downloaded {len(data)} bytes (Content-Type: {content_type})")
    except Exception as e:
        print(f"  {FAIL} Download error: {e}")

# ─── Test 9: JSON Regex Parsing ──────────────────────────────
def test_json_regex():
    section("9. JSON Regex Parsing (as used in agent_hailo.py)")

    test_chunks = [
        '{"action": "play_music"}',
        '{"action": "display_image", "image_url": "https://gen.pollinations.ai/image/cat"}',
        'Sure! Let me play a song! {"action": "play_music"} Here we go!',
        'No JSON here, just text.',
        '{"action": "set_expression", "value": "happy"}',
    ]

    for chunk in test_chunks:
        json_match = re.search(r'\{.*?\}', chunk, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                action = data.get("action", "none")
                print(f"  {PASS} '{chunk[:50]}...' → action={action}")
            except json.JSONDecodeError as e:
                print(f"  {FAIL} '{chunk[:50]}...' → regex matched but JSON invalid: {e}")
        else:
            expected_none = "No JSON" in chunk
            status = PASS if expected_none else FAIL
            print(f"  {status} '{chunk[:50]}...' → no regex match")


if __name__ == '__main__':
    print("=" * 60)
    print("  BMO Music & Image Diagnostic Test")
    print("=" * 60)

    test_music_keywords()
    test_image_keywords()
    test_build_image_action()
    test_brain_think()
    test_brain_stream_think()
    test_json_regex()
    test_music_files()

    # Only run hardware tests if on a Pi (aplay available)
    try:
        subprocess.run(['which', 'aplay'], capture_output=True, check=True)
        test_aplay()
    except (subprocess.CalledProcessError, FileNotFoundError):
        section("7. aplay Playback Test")
        print(f"  {WARN} Skipped — aplay not available (not on Pi?)")

    test_pollinations()

    print(f"\n{'='*60}")
    print("  Diagnostics complete!")
    print(f"{'='*60}\n")
