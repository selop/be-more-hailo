"""Tests for Brain.stream_think with a fake LLM (no NPU needed).

Run with ``pytest tests/test_brain.py`` or ``python tests/test_brain.py``.
"""

import os
import sys
import tempfile

os.environ["TRACE_DIR"] = tempfile.mkdtemp(prefix="bmo-test-trace-")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import llm  # noqa: E402


class FakeLLM:
    """Streams a fixed reply token by token, like hailo_platform.genai.LLM."""

    def __init__(self, reply):
        # Like the real tokenizer: the space belongs to the start of a token
        words = reply.split(" ")
        self.tokens = [words[0]] + [" " + w for w in words[1:]]

    def generate(self, **kwargs):
        return self

    def __enter__(self):
        return iter(self.tokens)

    def __exit__(self, *exc):
        return False


def stream(reply, user_text="tell me a story"):
    saved = (llm._get_llm, llm._prepare_prompt)
    llm._get_llm = lambda: FakeLLM(reply)
    llm._prepare_prompt = lambda history: history
    try:
        brain = llm.Brain()
        return brain, list(brain.stream_think(user_text))
    finally:
        llm._get_llm, llm._prepare_prompt = saved


def test_sentences_are_yielded_one_by_one():
    _, chunks = stream("Hello there. How are you?")
    assert [c.strip() for c in chunks] == ["Hello there.", "How are you?"]


def test_reply_is_cut_at_four_sentences_without_repeating_the_last():
    _, chunks = stream("One. Two. Three. Four. Five. Six.")
    assert [c.strip() for c in chunks] == ["One.", "Two.", "Three.", "Four."]


def test_unfinished_last_sentence_is_still_yielded_once():
    _, chunks = stream("First one. And then it trails off")
    assert [c.strip() for c in chunks] == ["First one.", "And then it trails off"]


def test_history_keeps_the_user_turn_and_the_reply():
    brain, _ = stream("Sure thing.", user_text="can you help")
    assert [m["role"] for m in brain.history] == ["system", "user", "assistant"]
    assert brain.history[1]["content"] == "can you help"


def test_misspelled_name_in_reply_is_corrected():
    from core.config import CHARACTER
    _, chunks = stream(f"{CHARACTER.heard_as[0]} is happy.")
    assert chunks[0].strip() == f"{CHARACTER.name} is happy."


def test_think_is_the_streamed_reply_joined():
    saved = (llm._get_llm, llm._prepare_prompt)
    llm._get_llm = lambda: FakeLLM("Hello there. How are you?")
    llm._prepare_prompt = lambda history: history
    try:
        assert llm.Brain().think("hi") == "Hello there. How are you?"
    finally:
        llm._get_llm, llm._prepare_prompt = saved


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
