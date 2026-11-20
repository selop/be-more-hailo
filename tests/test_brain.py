"""Tests for Brain.stream_think with a fake LLM (no NPU needed).

Run with ``pytest tests/test_brain.py`` or ``python tests/test_brain.py``.
"""

import os
import sys
import tempfile

os.environ["TRACE_DIR"] = tempfile.mkdtemp(prefix="bmo-test-trace-")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["LLM_BACKEND"] = "local"      # a device's .env may select the hosted model; tests never call it

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


def test_pronunciation_tag_is_learned_and_never_spoken():
    learned = []
    saved = (llm._get_llm, llm._prepare_prompt, llm.add_pronunciation)
    llm._prepare_prompt = lambda history: history
    llm.add_pronunciation = lambda word, phonetic: learned.append((word, phonetic))
    try:
        for tokens in (["Got", " it.\n!PRONOUNCE:", " tomato=tuh-may-toe"],
                       ["Got it.", "\n", "!", "PRONOUNCE:", " tomato=tuh-may-toe"]):     # tag split after the '!'
            fake = FakeLLM("")
            fake.tokens = tokens
            llm._get_llm = lambda fake=fake: fake
            chunks = list(llm.Brain().stream_think("it is pronounced tuh-may-toe"))
            assert [c.strip() for c in chunks] == ["Got it."], chunks
        assert learned == [("tomato", "tuh-may-toe")] * 2

        # Dots in the phonetic spelling cut the tag into several sentences: none of it is spoken
        fake = FakeLLM("")
        fake.tokens = ["Got it.", "\n", "!PRONOUNCE:", " tomato=tuh.", "may.", "toe"]
        llm._get_llm = lambda: fake
        del learned[:]
        chunks = list(llm.Brain().stream_think("you pronounce it tuh may toe"))
        assert [c.strip() for c in chunks] == ["Got it."] and learned == [("tomato", "tuh.may.toe")], (chunks, learned)

        # The model parrots the tag although nobody corrected anything: spoken text is clean, nothing is saved
        fake = FakeLLM("")
        fake.tokens = ["Once upon a time.", "\n!PRONOUNCE:", " word=phonetic"]
        llm._get_llm = lambda: fake
        del learned[:]
        chunks = list(llm.Brain().stream_think("tell me a story"))
        assert [c.strip() for c in chunks] == ["Once upon a time."] and learned == []
    finally:
        llm._get_llm, llm._prepare_prompt, llm.add_pronunciation = saved


def test_cloud_backend_answers_and_falls_back_to_local_when_it_fails():
    saved = (llm._get_llm, llm._prepare_prompt, llm.LLM_BACKEND, llm.cloud_llm.stream)
    llm._get_llm = lambda: FakeLLM("Local answer.")
    llm._prepare_prompt = lambda history: history
    llm.LLM_BACKEND = "cloud"
    try:
        seen = []

        def cloud(history):
            seen.append([m["role"] for m in history])
            yield from llm.cloud_llm._words(["Cloud ", "answer. Sec", "ond one."])    # cut anywhere, as the API streams
        llm.cloud_llm.stream = cloud
        brain = llm.Brain()
        assert [c.strip() for c in brain.stream_think("hi")] == ["Cloud answer.", "Second one."]
        assert seen == [["system", "user"]] and brain.history[-1]["content"] == "Cloud answer. Second one."

        def offline(history):
            raise ConnectionError("no network")
            yield
        llm.cloud_llm.stream = offline
        assert [c.strip() for c in llm.Brain().stream_think("hi")] == ["Local answer."]

        llm.cloud_llm.stream = lambda history: iter(())         # a refusal: nothing said
        assert [c.strip() for c in llm.Brain().stream_think("hi")] == ["Local answer."]
    finally:
        llm._get_llm, llm._prepare_prompt, llm.LLM_BACKEND, llm.cloud_llm.stream = saved


def test_action_json_after_a_blank_line_is_not_cut_in_two():
    saved = (llm._get_llm, llm._prepare_prompt)
    llm._prepare_prompt = lambda history: history
    fake = FakeLLM("")
    # As the hosted model wrote it on the device: a blank line, then the action
    fake.tokens = ["Timer", " kommt!", "\n\n{\"action\":", " \"set_timer\",", " \"minutes\":", " 0.0833,",
                   " \"message\":", " \"Timer", " is", " up!\"}"]
    llm._get_llm = lambda: fake
    try:
        chunks = [c.strip() for c in llm.Brain().stream_think("erzähl was")]
        assert chunks == ["Timer kommt!", '{"action": "set_timer", "minutes": 0.0833, "message": "Timer is up!"}'], chunks
    finally:
        llm._get_llm, llm._prepare_prompt = saved


def test_blank_line_ends_the_chunk_before_it():
    saved = (llm._get_llm, llm._prepare_prompt)
    llm._prepare_prompt = lambda history: history
    fake = FakeLLM("")
    fake.tokens = ['{"action":', ' "set_expression",', ' "value":', ' "happy"}', "\n\nYay,", " congratulations!", " More."]
    llm._get_llm = lambda: fake
    try:
        chunks = [c.strip() for c in llm.Brain().stream_think("i won")]
        assert chunks == ['{"action": "set_expression", "value": "happy"}', "Yay, congratulations!", "More."], chunks
    finally:
        llm._get_llm, llm._prepare_prompt = saved


def test_german_refusal_resets_the_history():
    brain, chunks = stream("Als KI kann ich dabei leider nicht helfen.", user_text="wer bist du")
    assert chunks and len(brain.history) == 1          # only the system prompt is left
    brain, _ = stream("Mir geht es gut, danke!", user_text="wie geht es dir")
    assert len(brain.history) == 3


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
