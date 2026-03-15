# Polish / Future TODOs

## Audio UX

- ~~Analyzing sounds overlap with thinking sounds~~ ✅ FIXED — `play_analyzing_sequence` now loops analyzing sounds only, no thinking sounds during image analysis.

## VLM

- **VLM swap time**: ~20-30s total (release LLM+STT → fork subprocess → VLM init → inference → exit → reload LLM+STT + re-cache system prompt). Most of this is model load time. No obvious optimization without NPU memory increase.

## Post-Rewrite Cleanup ✅ DONE

Completed: deleted `ensure_model.py`, `bmo/` directory, `scripts/fix_readme.py`, stale test files.
Updated: `CLAUDE.md`, `setup.sh`, `setup_services.sh`, `benchmark_llm.py`, `web_app.py` comment.
