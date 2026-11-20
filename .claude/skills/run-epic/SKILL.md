---
name: run-epic
description: Carry a piece of work that needs many steps from plan to done, across sessions, without losing track. Use this skill when the user asks for something that will take several commits or touches several domains (a new language, a model swap, a hardware change, a migration), when they say "epic", "plan and build", "work through this list" or "continue the epic", and when a TODO list has just been agreed and should now be executed.
user-invocable: true
argument-hint: "[epic name, or a goal for a new epic]"
---

# Running an epic

An epic is work too large for one sitting: several steps, some of them blocked on a decision or a measurement. What goes wrong with such work is not the coding. Steps get skipped, a decision made in conversation is forgotten, "done" is claimed for something nobody checked on the device. So the state lives in a file, and a step is only done when its check has been run.

## The epic file

One file per epic: `docs/epics/<slug>.md`. It is the single source of truth and what lets the next session continue. Keep this shape:

```markdown
# <Goal in one sentence, as what the user will be able to do>

Branch: epic/<slug>     Status: planning | in progress | blocked | done

## Decisions
| # | Question | Answer | Decided |
(open ones say "open". Only the user answers these.)

## Steps
| # | Step | Depends on | Done when | Status |
(status: todo, doing, done <commit>, blocked <why>, dropped <why>)

## Log
(dated lines: what was measured or found, with numbers. Newest last.)
```

"Done when" is a check someone can run: a test name, a `make say` phrase and what the trace must show, a measurement with a threshold, or "the user confirmed by ear". A step without one is not ready to start.

## Procedure

**Starting a new epic**

1. Read the code the goal touches before writing steps. Test the current behaviour with real inputs where that is cheap; a step list built from assumptions is wrong in the places that matter.
2. Write the epic file. Order the steps so the riskiest unknown comes first: if one measurement could sink the plan, it is step 1.
3. Put every choice that belongs to the user (how it looks, sounds or costs; what may be broken; what leaves the device) under Decisions as open. Ask them in one batch. Do not start a step that depends on an open decision; do start the ones that do not.
4. Create the branch `epic/<slug>` from `main` (or from the branch the work depends on; say which).

**Working a step**

1. Take the first step whose dependencies are done. Mark it `doing`.
2. Make the smallest change that completes it. If it crosses domains, check `docs/domain-map.md` for the seams (the `question-architecture` skill).
3. Run its "Done when" check, and `make check`. Anything that changes behaviour on the device is also checked there: `make ship`, then `make say` or a screenshot (see `docs/observability.md`).
4. One commit per step. Then mark the step `done <commit>` and add a Log line with what the check showed. Commit the epic file with the step.
5. If the step uncovered more work, add it as new steps. If it showed the plan is wrong, stop and say so before continuing.

**Between steps, stop and report when**

- a step needs a decision that is still open,
- a check fails twice for reasons you cannot explain,
- the next step is hard to undo or reaches outside the repository and the device (a purchase, a push, deleting data),
- the user's design judgement is the check ("does this sound right").

Otherwise keep going to the next step without asking.

**Finishing**

1. Every step is `done` or `dropped` with a reason. Nothing is left `doing`.
2. Review the whole branch once (`/ponytail:ponytail-review` if available) and fix what must be fixed.
3. Update `CLAUDE.md` and the docs the change made untrue. Move lasting findings from the Log into `docs/weak-links.md`.
4. Set Status to `done` and report: what the user can now do, what was verified and how, what was dropped, what is still open. Merging and pushing are the user's call.

## Continuing an epic in a new session

Read the epic file first, then `git log` on its branch. Trust the file over memory, and re-run the last step's check before building on it: a step marked done in an earlier session is a claim until you have seen it pass.

## Do not

- Mark a step done on the strength of a test alone when it changes what the device does.
- Batch several steps into one commit, or start a second step while one is `doing`.
- Let the file drift: if a decision changes in conversation, write it down before acting on it.
- Grow the epic silently. New scope is a new step the user can see, or a new epic.
