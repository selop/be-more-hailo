# Future Use Cases: Traditional NPU Models for BMO

The Hailo-10H can run traditional inference models (YOLO, pose estimation, depth) alongside the genai LLM. These use the standard HailoRT pipeline, not the genai API, so they may coexist on the shared VDevice — similar to how Whisper (125MB) coexists with the LLM. **This needs verification.**

Available pre-compiled HEFs: https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/MODELS.rst

---

## Pose Estimation (yolov8s_pose)

- **Exercise buddy** — BMO counts reps ("That's 12 squats! BMO is so proud!"), detects form issues
- **Simon Says** — BMO asks you to strike poses, uses pose estimation to verify ("Touch your toes! ...BMO sees you did it!")
- **Dance judge** — during JAMMING state, BMO rates your dance moves
- **Gesture control** — wave to wake BMO instead of voice wake word (accessibility, noisy environments)
- **Fall / slump detection** — detect if someone falls or is slumped for extended time (elderly care)

## Object Detection (yolov8s / yolov11m)

- **I Spy game** — "BMO sees something blue and round..." uses real-time detection from the camera. Perfect fit for BMO's Adventure Time personality
- **Pet watcher** — "Your cat just walked by! BMO says hello, kitty!" — detect pets and react
- **Room guardian** — alert when someone enters while you're away, log activity
- **Context-aware responses** — BMO notices you're holding a coffee mug: "Good morning! Coffee time?"
- **Lost item helper** — "BMO, have you seen my keys?" → check recent detection history (requires a simple rolling buffer of detected objects + timestamps)

## Depth Estimation (StereoNet)

- **Proximity-aware volume** — BMO speaks louder when you're far away, quieter when close
- **Approach detection** — BMO wakes up when someone walks toward them (no wake word needed)
- **Personal space personality** — different interaction style at different distances (intimate whisper vs room-scale projection)

## Face Detection (scrfd)

- **Who's there** — BMO recognizes household members, greets by name ("Good morning, friend!")
- **Attention detection** — BMO pauses speaking if you look away, resumes when you look back
- **Multi-person awareness** — "BMO sees two friends today! Hello everyone!"
- **Emotion mirroring** — detect smiles/frowns and mirror with BMO's expression states

---

## Implementation Considerations

### VDevice Coexistence
Traditional HailoRT inference models use `ConfiguredInferModel` / `InferModel`, not the genai `LLM`/`VLM` classes. They should be able to share a VDevice with the LLM via `group_id="SHARED"` since they use different NPU resources. **Needs benchmarking** — run a YOLO model alongside the LLM and check for contention.

### Camera Pipeline
BMO currently captures single frames via `rpicam-still` (subprocess). Continuous vision (object detection, pose) would need a persistent camera stream — either `picamera2` Python library or `rpicam-vid` piped to OpenCV. This is a significant change to the camera architecture.

### Priority
1. **Object detection** — highest bang-for-buck, enables I Spy game and context awareness
2. **Pose estimation** — enables physical games, most fun for the BMO character
3. **Face detection** — enables personalization
4. **Depth estimation** — nice-to-have, less immediately useful

### Model Sizes (from Hailo Model Zoo)
| Model | HEF Size | FPS (H10) | Use Case |
|-------|----------|-----------|----------|
| yolov8s | ~15MB | 30+ | Object detection |
| yolov8s_pose | ~20MB | 20+ | Pose estimation |
| scrfd_2.5g | ~5MB | 50+ | Face detection |
| StereoNet | ~10MB | 15+ | Depth estimation |

All are tiny compared to the LLM (2.3GB) — should coexist without memory pressure.
