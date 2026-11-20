---
name: solid-python
description: Hold Python code in this project to the SOLID rules, read the way Python is written (functions, callables and small classes, not Java-style hierarchies). Use this skill when reviewing or refactoring a class or module, before adding a responsibility to an existing class, when a class or function has grown large, when a function takes a long list of callbacks, and whenever the user mentions SOLID, god classes, coupling, or "clean up this class".
user-invocable: true
argument-hint: "[file, class, or nothing for the whole engine]"
---

# SOLID for this Python codebase

The rules exist to keep a change in one place. A finding is only worth fixing when you can name the change that the current shape makes expensive or risky. "It breaks the rule" is not a reason; "muting cannot be tested without a display" is.

## Procedure

1. **Point.** Run the pointer script. It lists places to look at, largest first. It is advisory: every line is a question.

   ```bash
   python .claude/skills/solid-python/solid_check.py            # the engine
   python .claude/skills/solid-python/solid_check.py core/turn.py
   ```

2. **Read.** Open each place and answer its question from the code. Drop the ones that are fine as they are and say why in one line (see "Not a violation").
3. **Rank** what is left by what it costs today: an untestable behaviour, a change that has to be made in two places, a caller forced to supply things it does not care about.
4. **Fix the top of the list, smallest change first.** Move code, do not rewrite it. Moved code keeps its error handling. Every behaviour you move behind a new seam gets one hardware-free test in `tests/`.
5. **Verify.** `make check`, then on the device (`make ship`, a test turn, a screenshot). Run the pointer script again and report before and after.
6. **Report** in this form:

   ```
   <file:line> [rule] what was wrong, as the change it made hard
   Fix: what moved where
   Check: the test or measurement that covers it
   ```

   End with what you left alone and why.

## The five rules, as they apply here

**S: one reason to change.** A class or module answers to one domain in `docs/domain-map.md`. Signs: a display class that also decides behaviour, a function that both decides and performs, a class whose methods fall into groups that never touch the same attributes. Fix by moving the group to the domain that owns it. Size alone is a hint, not a finding.

**O: add a case without editing the dispatcher.** A chain of `if kind == ...` that grows with every feature should be a dict from name to function, or data in the character pack. Two or three branches that will not grow are fine.

**L: a subclass must work wherever its base does.** No overrides that raise `NotImplementedError`, narrow the accepted arguments or return a different kind of thing. `TurnUI` is the model: every method is a harmless no-op, so any subclass and the base itself can be passed. Prefer passing a callable to subclassing.

**I: ask only for what you use.** A function that takes ten callbacks, most of them `lambda: store.something`, should take the object those lambdas read. A caller should never have to build arguments it does not care about. In the other direction, do not pass a whole object when one value or one callable is all that is used.

**D: depend on what is passed in, not on what you can reach.** Behaviour that needs hardware (NPU, microphone, speaker, camera, display) receives it as a parameter or calls it through a module function a test can replace. `core/turn.py` is the model: `Conversation` gets `store`, `sounds`, `ui` and `brain` from its caller, and the tests pass fakes.

## Not a violation

- **The composition root.** One place has to build the objects and hand them to each other. Here that is `Agent.__init__` in `agent_hailo.py`. It may construct many collaborators; it should hold little else.
- **Pure-function modules** (`core/actions.py`, `core/timers.py`). A module of functions is already the right shape.
- **A long function that does one thing in order**, such as a boot sequence, if splitting it would only scatter the steps.

## Do not

- Add an abstract base class, `Protocol` or factory for something with one implementation. Pass the object or a callable.
- Split a class into several that always change together. That moves the problem and adds wiring.
- Change behaviour in the same step as structure. A refactor commit moves code; the tests that passed before pass after.
- Break the engine and character split: no character names or lines in engine code (`tests/test_character.py` checks this).
