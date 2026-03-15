# Polish / Future TODOs

## Audio UX

- **Analyzing sounds overlap with thinking sounds**: The `play_analyzing_sequence` plays one analyzing WAV, then falls back to looping thinking sounds as filler while VLM runs (~20-30s). Ideally it should loop analyzing sounds only (or play silence). The issue is that VLM inference takes long enough that one analyzing clip isn't sufficient, but thinking sounds are thematically wrong for image analysis.
  - Options: generate more/longer analyzing clips, loop the analyzing category instead of falling back to thinking, or play a gentle ambient loop during analysis.

## VLM

- **VLM swap time**: ~20-30s total (release LLM+STT → fork subprocess → VLM init → inference → exit → reload LLM+STT + re-cache system prompt). Most of this is model load time. No obvious optimization without NPU memory increase.

## Post-Rewrite Cleanup

After Tiers 1-3 the codebase has accumulated dead references and stale documentation. Sweep through these areas:

- **Dead imports & unused variables**: audit `agent_hailo.py`, `core/llm.py`, `core/tts.py`, `web_app.py` for stale imports (e.g. `requests`, `signal`, `subprocess` where no longer used)
- **Config cleanup**: `core/config.py` still has comments referencing "offload to your Linux server" and hailo-ollama. `bmo/config.py` is a legacy duplicate with old constants (`LLM_URL`, etc.)
- **CLAUDE.md**: still describes the hailo-ollama architecture ("hailo-ollama (LLM server)", `LLM_URL`, `FAST_LLM_MODEL`). Needs to reflect direct NPU API, shared VDevice, VLM subprocess, NPU STT
- **Obsolete files**: `ensure_model.py` (hailo-ollama model puller), `scripts/fix_readme.py` (references old config constants), `bmo/` directory (legacy pre-Hailo code?)
- **start_web.sh**: check for hailo-ollama references like `start_agent.sh` had
- **Stale comments**: references to "hailo-ollama", "HTTP POST", "pkill" scattered in code comments
- **Benchmark files**: `benchmark_llm.py` still imports `_format_prompt` which was removed — update or remove
