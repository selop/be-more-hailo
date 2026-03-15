# Be More Agent — Hailo-10H Edition

<p align="center">
  <img src="bmo_irl.jpg" height="300" alt="BMO On-Device" />
  <img src="bmo-web.png" height="300" alt="BMO Web Interface" />
</p>

A fork of [@moorew's be-more-hailo](https://github.com/moorew/be-more-hailo) (itself a fork of [@brenpoly's be-more-agent](https://github.com/brenpoly/be-more-agent)), built to run fully on-device on a **Raspberry Pi 5** with the **Raspberry Pi AI HAT 2+** (Hailo-10H). BMO listens for its wake word, understands what you say, thinks about it locally, and talks back — no cloud, no subscriptions, no data leaving your house.

This fork replaces the upstream's HTTP-based inference stack with **direct NPU Python APIs** and squeezes every bit of performance out of the Hailo-10H hardware.

---

## ELI5: What makes this fork different?

The [original project](https://github.com/moorew/be-more-hailo) talks to the Hailo chip through a web server (`hailo-ollama`) — your Pi sends an HTTP request to `localhost:8000`, waits for the full response, then speaks it. That's fine for a demo, but it adds latency at every step and means your Pi is running a separate server process just to forward data to the chip sitting right there on the board.

This fork **rips out the middleman**. Every AI model talks directly to the NPU through Hailo's Python API — no HTTP, no server, no serialization overhead. The result:

| What changed | Before (upstream) | After (this fork) | Improvement |
|---|---|---|---|
| **LLM inference** | HTTP request to hailo-ollama | Direct `hailo_platform.genai.LLM` API | **33% faster** first token (0.55s -> 0.37s) |
| **Speech-to-text** | CPU subprocess (whisper.cpp) | NPU `Speech2Text` API | **7.3x faster** (1.91s -> 0.26s) |
| **TTS audio** | Spawn `aplay` subprocess per sentence | Persistent `sounddevice` stream | **Zero process overhead** per sentence |
| **System prompt** | Re-processed every single turn | KV cache saved once at boot | **Processed once**, reused forever |
| **Response style** | Wait for full response, then speak | Stream token-by-token, speak per sentence | **User hears first sentence in <0.5s** |
| **Vision (VLM)** | `pkill hailo-ollama`, pray it restarts | Clean subprocess with dedicated VDevice | **No crashes**, clean NPU handoff |

The practical difference: you say "Hey BMO", ask a question, and BMO starts answering in **under a second**. The upstream approach takes several seconds before you hear anything.

### How the NPU sharing works

The Hailo-10H has 8GB of memory. The LLM (Qwen 2.5, 2.3GB) and the Whisper STT model (125MB) both fit and **run simultaneously** on a shared VDevice. They coexist because Whisper is small enough. The VLM (Qwen2-VL, 2.3GB) can't coexist with the LLM — not a memory issue, but a HailoRT runtime limitation that only allows one generative model at a time. So when BMO needs to see, it releases the LLM, forks a child process for VLM inference, and reloads the LLM when the child exits.

---

## What runs where

| Component | Runtime | Model | Notes |
|---|---|---|---|
| LLM | Hailo-10H NPU | Qwen2.5-1.5B-Instruct | Direct Python API, KV-cached system prompt |
| Vision (VLM) | Hailo-10H NPU | Qwen2-VL-2B-Instruct | Runs in subprocess (NPU holds one generative model at a time) |
| STT | Hailo-10H NPU | Whisper-Base | Coexists with LLM on shared VDevice; CPU whisper.cpp fallback |
| TTS | CPU | Piper en_GB-semaine-medium | Persistent audio stream, sentence-by-sentence |
| Wake word | CPU | OpenWakeWord (custom wakeword.onnx) | Suppressed during speech/music to prevent false triggers |

---

## Interfaces

### On-Device (`agent_hailo.py`)
BMO in its natural habitat. Plug in a screen, a USB mic, and a USB speaker and you get the full experience: 22+ animated face states, wake word detection, and the whole listen -> think -> speak loop running locally. When listening, BMO shows a microphone with an animated VU meter that reacts to your voice in real time. When taking a photo, BMO switches to a dedicated camera face with a shutter animation. After each response, BMO stays in "Still listening..." mode for 8 seconds so you can keep a conversation going without re-saying the wake word every time.

### Web (`web_app.py`)
A FastAPI server with a browser-based UI — useful if you want to talk to BMO from another room, or you'd rather not have a screen hanging off your Pi. Hold a button to record, and BMO responds with audio in your browser.

The web interface includes:
- **Debug panel** — conversation history and live server logs
- **Pronunciation override** — corrects how Piper pronounces specific words
- **LLM status indicator** — shows whether the NPU model is ready
- **Hands-free mode** — enables wake word detection so you don't need to hold the button
- **Pi Audio toggle** — routes audio to the Pi's physical speaker instead of browser playback

---

## Interactive Features

BMO includes several dynamic, interactive capabilities beyond basic conversation:

- **Timers & Alarms:** Ask BMO to *"Set a timer for 10 minutes"* or *"Remind me to check the oven"*. BMO will happily interrupt you later when the time is up!
- **Minigames:** BMO is a living game console. Say *"Let's play Trivia"* or *"Let's play a guessing game"* — BMO will act as the host, wait for your answers, and keep score.
- **Vision Analysis:** Hold an object up to the camera and say *"What am I holding?"* or *"Does this look good?"*. BMO will snap a photo, analyze it using the local VLM, and give you its opinion.
- **Musical Talent:** Ask BMO to *"Play some music"* or *"Sing a song"*, and BMO will cycle into a dancing `Jamming` face while playing chiptunes (add your own `.wav` files to `sounds/music/`).
- **Image Generation:** Ask BMO to *"Draw me a sunset"* and they'll generate an image via [Pollinations.ai](https://pollinations.ai/) and display it on-screen with a retro LCD border.
- **Idle Pet Animations:** When left alone in Screensaver mode, BMO will periodically show affection by flashing pixelated hearts, getting dizzy, or falling asleep to keep your desk feeling alive.

---

## Secure remote access

Modern browsers require HTTPS for microphone access, which makes things awkward when your Pi is just sitting on your local network. [Tailscale](https://tailscale.com/) solves this elegantly — install it on your Pi and your other devices, enable HTTPS certificates, and you get a proper `*.ts.net` address with a real cert, reachable from anywhere on your Tailnet. No port forwarding, no dynamic DNS nonsense.

> **Disclosure:** I work at Tailscale. That said, I genuinely use it for this project and it's the best solution I've found for exactly this problem.

1. Install Tailscale on the Pi and your client device
2. Enable [HTTPS certificates](https://tailscale.com/kb/1153/enabling-https/) in the Tailscale admin console
3. On the Pi, run:
   ```bash
   tailscale serve --bg --https=443 localhost:8080
   ```
4. Access the web UI at `https://<your-pi-hostname>.ts.net`

Your BMO is then reachable from your phone, laptop, or any device on your Tailnet — mic access works, and it's not exposed to the open internet.

---

## Hardware

- Raspberry Pi 5 (4GB or 8GB recommended)
- Raspberry Pi AI HAT 2+ (Hailo-10H, required for NPU features)
- USB microphone and speaker (for on-device mode)
- HDMI or DSI display (for on-device GUI)
- Raspberry Pi Camera Module (optional, for vision/photo features)

---

## Project structure

```
be-more-hailo/
├── agent_hailo.py          # On-device Tkinter GUI + animation + main loop
├── web_app.py              # FastAPI web server + browser UI
├── cli_chat.py             # Minimal CLI chat for quick testing
├── core/
│   ├── config.py           # All configuration (models, devices, paths, system prompt)
│   ├── npu.py              # NPU device lifecycle (VDevice, LLM, VLM subprocess)
│   ├── actions.py          # Keyword matching, response cleaning (pure functions, no HW)
│   ├── llm.py              # Brain class: conversation history + inference orchestration
│   ├── dispatch.py         # Stream chunk -> action dispatch (camera, music, timer, etc.)
│   ├── audio_input.py      # Wake word detection + mic recording
│   ├── screensaver.py      # Idle thought generation + animation loop
│   ├── tts.py              # Piper TTS with persistent AudioPlayer stream
│   ├── stt.py              # Whisper NPU (Speech2Text) + CPU fallback
│   ├── search.py           # DuckDuckGo web search wrapper
│   ├── log.py              # Colored logging
│   ├── bubble.py           # Thought bubble Tkinter widget
│   └── meter.py            # Mic level meter Tkinter widget
├── generate_faces.py       # Procedural face generator (4x supersampled)
├── faces/                  # 22+ expression directories with animation frames
├── sounds/                 # Categorized audio: boot, greeting, thinking, camera, music, etc.
├── templates/              # Jinja2 HTML templates for the web UI
├── static/                 # CSS, JS, favicon
├── models/                 # HEF files (LLM, VLM, Whisper) — gitignored
├── piper/                  # Piper TTS engine and voice model — gitignored
├── wakeword.onnx           # OpenWakeWord "Hey BMO" model
├── setup.sh                # Automated installation script
├── setup_services.sh       # Installs systemd background services
├── start_web.sh            # Starts the web server
├── start_agent.sh          # Starts the on-device GUI
└── requirements.txt        # Python dependencies
```

The `core/` modules follow SOLID principles — `actions.py` and `dispatch.py` are 100% testable without NPU hardware, `npu.py` owns all Hailo lifecycle, and `llm.py` only handles conversation logic.

---

## Installation

### Prerequisites

- Raspberry Pi OS (64-bit, current stable)
- `hailo-h10-all` installed — the setup script handles this, but if installing manually: `sudo apt install hailo-h10-all`

### Automated install

```bash
curl -sSL https://raw.githubusercontent.com/moorew/be-more-hailo/main/setup.sh | bash
cd be-more-agent
```

The script handles everything:
- Installs system packages including `libcamera-apps` for camera support
- Fixes the Hailo driver conflict (blacklists the legacy `hailo_pci` module)
- Downloads and extracts the Piper TTS engine
- Clones and compiles `whisper.cpp` (CPU fallback)
- Downloads model HEF files from Hailo's CDN
- Creates a Python virtual environment and installs dependencies
- Enables system site-packages in the venv so Python can use `hailo_platform`
- Checks camera availability and lets you know if anything's missing

### Manual install

```bash
git clone https://github.com/moorew/be-more-hailo.git be-more-agent
cd be-more-agent
chmod +x *.sh
./setup.sh
```

---

## Running

**Web Interface (Kiosk Mode):**
```bash
./setup_web.sh
```
This script installs all necessary Python and system audio dependencies, sets up the `bmo-web.service` to start on boot, and configures Chromium to automatically open in full-screen kiosk mode on desktop login.

To manually start/stop the web backend: `sudo systemctl start|stop|restart bmo-web`
To run manually without the service: `. venv/bin/activate && ./start_web.sh`

**On-device GUI (Tkinter):**
```bash
source venv/bin/activate
./start_agent.sh
```

**Auto-start services:**
```bash
./setup_services.sh
```
Then manage with `sudo systemctl start|stop|restart bmo-gui` or `bmo-web`.

---

## Configuration

All settings live in `core/config.py`. The most commonly changed values:

```python
# LLM model HEF (direct NPU inference — no hailo-ollama needed)
LLM_HEF_PATH = "./models/Qwen2.5-1.5B-Instruct.hef"

# Vision model HEF
VLM_HEF_PATH = "./models/Qwen2-VL-2B-Instruct.hef"

# STT model HEF (NPU Whisper — 7x faster than CPU)
WHISPER_HEF_PATH = "./models/Whisper-Base.hef"

# Audio device for local hardware playback (run `aplay -l` to find yours)
ALSA_DEVICE = "plughw:UACDemoV10,0"

# Microphone device index (run `python3 -c "import sounddevice as sd; print(sd.query_devices())"`)
MIC_DEVICE_INDEX = 1
MIC_SAMPLE_RATE  = 48000
```

Environment variables override any of these at runtime:
```bash
export ALSA_DEVICE="plughw:2,0"
export SILENCE_THRESHOLD=60000
```

---

## Camera and vision

If you have a Raspberry Pi Camera Module connected:

1. Enable the camera interface in `raspi-config`
2. Install camera tools if not already present:
   ```bash
   sudo apt install -y libcamera-apps
   ```
3. Say something like "Hey BMO, take a photo and tell me what you see" — the agent captures a frame with `rpicam-still` and sends it to the VLM running natively on the NPU

The VLM runs in a child process with its own VDevice. The main process releases the LLM before forking, and reloads it when the child exits. This avoids the old `pkill hailo-ollama` approach that was prone to crashes and race conditions.

---

## Customisation

BMO is pretty easy to make your own:

**Personality:** Edit `get_system_prompt()` in `core/config.py`. This is where BMO's voice, tone, and quirks are defined.

**Faces:** BMO's faces are procedurally generated by `generate_faces.py` using 4x supersampling for perfectly smooth, anti-aliased lines. Run `python generate_faces.py` to regenerate all frames across 22+ expression states.

**Expressions:** The LLM can trigger any expression by outputting `{"action": "set_expression", "value": "happy"}`. Available emotions:

| Expression | Description |
|---|---|
| `happy` | Upturned arc eyes with a bouncing smile |
| `sad` | Downturned slash eyes with a frown that droops |
| `angry` | Crossed slash eyes with a flat trembling mouth |
| `surprised` | Big round eyes with a pulsing O-shaped mouth |
| `sleepy` | Closed eyes with floating Z letters |
| `dizzy` | X-shaped eyes with a wavy squiggle mouth |
| `cheeky` | One open eye, one winking, wagging tongue |
| `heart` | Beating heart-shaped eyes (scales up and down) |
| `starry_eyed` | Spinning 4-point sparkle stars for eyes |
| `confused` | One oversized eye, one flat line, wiggly mouth |
| `jamming` | Closed eyes, big smile, bouncing musical notes |
| `football` | Football persona gag *(screensaver)* |
| `detective` | Detective persona gag *(screensaver)* |
| `sir_mano` | Sir Mano persona gag *(screensaver)* |
| `bee` | Bee persona gag *(screensaver)* |

**Sounds:** Put `.wav` files in `sounds/<category>/`. BMO picks one at random per event.

**Wake word:** Replace `wakeword.onnx` with any [OpenWakeWord](https://github.com/dscripka/openWakeWord)-compatible model.

**Image Generation:** When BMO discusses highly visual topics (especially during screensaver musings or when explicitly asked), they use the local LLM to generate a descriptive prompt. This prompt is then sent to [Pollinations.ai](https://pollinations.ai/), a free community API that generates the image in the cloud and returns it to the Pi. BMO then applies a custom retro LCD border before displaying it on-screen. This keeps the Pi fast and responsive without needing to run heavy Diffusion models locally!

---

## Screensaver personality

When BMO has been idle for 60 seconds, it enters screensaver mode and cycles through its expressions. Approximately every **30 minutes**, BMO will "think out loud" by:

1. Searching the web for a random topic (weather, news, fun facts, quotes, science, jokes)
2. Feeding the search result to the on-device LLM with a special prompt
3. Speaking the generated thought via Piper TTS

BMO stays quiet during:
- **Night hours** (10 PM - 8 AM)
- **Recent interaction** (within 60 seconds of your last conversation)

This all runs locally — search results go through DuckDuckGo and the LLM processes them on the Hailo NPU.

---

## Troubleshooting

**Hailo NPU not detected (`/dev/hailo0` missing)**

This is usually caused by a driver conflict. The system ships with both `hailo_pci` (Hailo-8) and `hailo1x_pci` (Hailo-10H) drivers. If the old one loads first, it blocks the new one from creating the device node. Fix it by blacklisting the old driver:
```bash
echo "blacklist hailo_pci" | sudo tee /etc/modprobe.d/blacklist-hailo-legacy.conf
sudo rmmod hailo1x_pci 2>/dev/null; sudo rmmod hailo_pci 2>/dev/null
sudo modprobe hailo1x_pci
ls /dev/hailo0  # should now exist
```
The setup script handles this automatically, but if you installed manually you may need to do it yourself.

**Inference fails with `HAILO_OUT_OF_PHYSICAL_DEVICES` (status 74)**

This means `/dev/hailo0` doesn't exist. Common causes:

1. **Driver conflict** — see the blacklist fix above.
2. **Kernel updated, driver not rebuilt** — after a kernel update the Hailo DKMS module may be missing for the new kernel. Verify with:
   ```bash
   uname -r                          # e.g. 6.12.62+rpt-rpi-2712
   ls /lib/modules/$(uname -r)/extra/hailo*  # should list .ko files
   ```
   If the module is missing, reinstall the driver package to rebuild it:
   ```bash
   sudo apt reinstall h10-hailort-pcie-driver
   sudo reboot
   ```
3. **Another process holding the device** — check with `lsof /dev/hailo0`.

**VLM fails with `HAILO_INVALID_OPERATION` / `HailoRTStatusException: 6`**

This usually means the VLM HEF file was compiled for a different HailoRT version. The HEF must match your installed runtime:
```bash
dpkg -l | grep hailort  # check your version (e.g. 5.1.1)
```
Re-download the matching HEF:
```bash
HAILORT_VER=$(dpkg-query -W -f='${Version}' h10-hailort)
wget -O models/Qwen2-VL-2B-Instruct.hef \
    "https://dev-public.hailo.ai/v${HAILORT_VER}/blob/Qwen2-VL-2B-Instruct.hef"
```

**Camera vision says "my eyes aren't working"**

If the VLM HEF is present but inference still fails, check that `hailo_platform` is importable:
```bash
source venv/bin/activate
python3 -c "from hailo_platform.genai import VLM; print('OK')"
```
If it fails, ensure system site-packages are enabled: `grep include-system venv/pyvenv.cfg` should say `true`.

---

## Credits

The original project is entirely the work of [@brenpoly](https://github.com/brenpoly/be-more-agent) — the concept, the character, and the original implementation. [@moorew](https://github.com/moorew/be-more-hailo) created the Hailo-10H fork that this project builds on, adding the web interface, the shared `core/` module layer, and initial Hailo NPU support via hailo-ollama. This fork takes that further with direct NPU inference APIs, modular architecture, and the performance optimizations described above.

**"BMO"** and **"Adventure Time"** are trademarks of Cartoon Network (Warner Bros. Discovery). This is a fan project for personal and educational use only, not affiliated with or endorsed by Cartoon Network.

---

## License

MIT — see [LICENSE](LICENSE).
