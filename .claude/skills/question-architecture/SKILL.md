---
name: question-architecture
description: Interrogate BMO's overall architecture to find its weak links, and keep the project cut into problem domains that dedicated agents can own. Use this skill whenever the user asks what could be improved, where the weak points or bottlenecks are, how the project should be split into modules, which agent or domain should own a piece of work, or whether a planned feature fits the current design. Also use it before starting any change that touches more than one domain, and after finishing a larger feature, to re-check that the domain boundaries still hold.
user-invocable: true
argument-hint: "[domain name | planned change | empty for a full sweep]"
---

# Questioning BMO's Architecture

This skill exists because BMO's real problems have not been where the docs said they were. The first sweep found that the docs described sentence-by-sentence speech while the code buffered the whole reply, and that the "cached system prompt" optimisation had frozen BMO's clock at boot time. Neither showed up as a bug report; both were found by asking a pointed question and reading the code for the answer.

So the job here is to ask questions and answer them from code. Docs, `CLAUDE.md` and comments tell you what was intended. Only the code tells you what happens.

Two files hold the state this skill maintains:

- `docs/domain-map.md` — the problem domains, the files each owns, and the seams between them.
- `docs/weak-links.md` — the ledger of known weak links, with status.

Read both before you start so you build on earlier sweeps instead of rediscovering them.

## Pick the mode from the argument

- **No argument: full sweep.** Trace one voice turn end to end, apply every lens below, re-rank the ledger.
- **A domain name** (e.g. `hearing`): apply the lenses inside that domain and at its seams only.
- **A planned change** (e.g. "add barge-in"): say which domains it touches, which seams it crosses, which open weak links it depends on or would make worse, and whether it should be split into per-domain tasks.

## How to look

Start from the path a single utterance takes: wake word, recording, transcription, routing, LLM, dispatch, speech, back to idle. Read the code along that path in order. Most weak links sit on it, and reading in execution order is what exposes a step that waits when it could stream or a value computed once that should be fresh.

Then apply the lenses. Each is a question to answer with a file and line, not a checklist item to tick.

1. **Claim versus code.** Pick the claims the docs are proudest of (streaming, caching, coexistence, fallback). Does the code do that today? Refactors quietly undo these.
2. **What is the user waiting for?** At every step of a turn, what blocks before BMO makes a sound or changes its face? What is the worst case, given token limits and tokens per second?
3. **Frozen and stale state.** What is computed once at startup or import time and then reused: prompts, caches, config read from env, module-level singletons? Would any of it be wrong an hour later, or after a model reload?
4. **Budgets.** What is bounded by count when the real limit is something else? Context is measured in tokens, recordings in seconds, RAM in bytes. Look for the mismatch.
5. **What a restart forgets.** Which user-visible things live only in memory (history, timers, learned preferences)? Would the user expect them to survive?
6. **Seams.** Where does one domain reach into another's internals: private imports, shared globals, GUI widgets touched from worker threads, two owners of one device (mic, speaker, NPU)? A seam crossed by private names is a place two agents will collide.
7. **Fragile decisions.** Where does behaviour hang on substring matching, a magic threshold, or a small model emitting exact JSON? What ordinary sentence would trigger it wrongly, and what does a misfire cost?
8. **Twins that drift.** Where is the same logic written twice (streaming and non-streaming, GUI and CLI)? Have the copies already diverged?
9. **How would we know?** If this broke tomorrow, what would catch it? Pure-function modules with no tests are the cheapest gap to close.
10. **Abandoned attempts.** The docs and git history record things that were tried and reverted. Has the reason for dropping them changed, for example because another weak link got fixed?
11. **The error path.** When a step fails halfway (no camera, mic gone, model load error), what is left behind? Look for cleanup or reload that only runs on success, and for fallbacks that keep BMO working but slower with nothing telling the user.
12. **Who else is running?** Several threads share one LLM, one speaker and one phase. For each shared thing, what stops two of them using it at once, and what does the user see when they collide?

Where a suspicion concerns a pure module (`core/actions.py`, `core/dispatch.py`), run it locally with a few inputs instead of reasoning about it. A misfire you reproduced is worth more than one you predicted.

Stop when further reading stops changing the ranking. A sweep that lists twenty findings has not done the ranking work; aim for the handful that matter.

## Evidence standard

Every weak link needs a `file:line` that a reader can open and see for themselves. Say plainly whether each claim is read from code, measured, or estimated, because a Pi-side latency number guessed from tokens per second is not the same kind of fact as a line of code. If the Pi is reachable, prefer measuring to estimating. If it is not, say so once.

Measuring is cheap: `scripts/bmo-remote.sh say "..." --quiet` runs a real turn on the device and prints its trace (transcript, route, model output, actions, timings), `stats` summarises recent turns, and `screenshot` shows what is on the display. See `docs/observability.md`. A suspicion about behaviour on the device should be reproduced this way before it goes into the ledger, and a fix should be confirmed the same way before the entry is marked fixed.

If a suspicion does not survive reading the code, drop it. Do not report it as a "possible" issue.

## Report format

Lead with the single most important finding in one sentence. Then:

```
## <n>. <weak link, stated as what goes wrong for the user>
Domain: <owner> (seam with <other domain> if it crosses one)
Evidence: <file:line>, <read from code | measured | estimated>
Cost: <what the user experiences, with a number where possible>
Smallest fix: <one or two sentences>
```

Rank by user impact divided by effort. Close with:

- **Where to start:** the two or three items to do first, and why in that order.
- **Questions for you:** up to three things the code cannot answer and that would change the ranking (how the device is actually used, what annoys them most, what they are willing to break).

For a planned-change review, replace the ranked list with: what in the current code stands in the way, any earlier attempt and whether its reasons for failing still hold, domains touched, seams that have to change, and a split into per-domain tasks with an owner, files and a way to verify each.

For a domain review, follow the ranked list with a briefing the owning agent can start from: the files it owns, the interfaces it must not change alone (signature and call sites), things outside its files that shape its work, where the docs are wrong about this domain, and what can be verified off-device versus only on the Pi.

## Keep the ledger and the map current

After a sweep, update `docs/weak-links.md`: add new entries, mark fixed ones as fixed with the commit, and re-rank. Each entry has a stable ID (domain prefix and number, such as `HEAR-2`) that never changes; rank is only the order of the table. Refer to entries by ID in reports and task briefs so references survive re-ranking. A domain or planned-change review adds or refines entries for what it examined and leaves the rest alone.

An entry that has been open across several sweeps without moving deserves a question to the user about whether it still matters.

Update `docs/domain-map.md` when a file moves, a new module appears, or a seam changes. The map is what lets a dedicated agent be told "you own hearing" and know exactly which files that means and which interfaces it may not change alone.

## Handing work to domain agents

When a finding becomes a task, write the brief from the map:

- the domain and the files it owns,
- the seam contracts it has to keep stable (function signatures other domains call),
- the weak-link entry it is fixing, with its evidence,
- how to verify the fix without the Pi where possible (the pure modules) and what has to be checked on the device.

A task that needs to change a seam is not a single-domain task. Name both owners and settle the new contract before splitting the work.
