# Polish / Future TODOs

## Audio UX

- **Analyzing sounds overlap with thinking sounds**: The `play_analyzing_sequence` plays one analyzing WAV, then falls back to looping thinking sounds as filler while VLM runs (~20-30s). Ideally it should loop analyzing sounds only (or play silence). The issue is that VLM inference takes long enough that one analyzing clip isn't sufficient, but thinking sounds are thematically wrong for image analysis.
  - Options: generate more/longer analyzing clips, loop the analyzing category instead of falling back to thinking, or play a gentle ambient loop during analysis.

## VLM

- **VLM swap time**: ~20-30s total (release LLM+STT → fork subprocess → VLM init → inference → exit → reload LLM+STT + re-cache system prompt). Most of this is model load time. No obvious optimization without NPU memory increase.

## Post-Rewrite Cleanup ✅ DONE

Completed: deleted `ensure_model.py`, `bmo/` directory, `scripts/fix_readme.py`, stale test files.
Updated: `CLAUDE.md`, `setup.sh`, `setup_services.sh`, `benchmark_llm.py`, `web_app.py` comment.
