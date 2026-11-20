"""Replies from a hosted model (Claude) instead of the small local one.

Off unless ``LLM_BACKEND=cloud`` is set. When it is on, what the user said
and the conversation history leave the device. Needs ``ANTHROPIC_API_KEY``
(in ``.env`` or the service environment) and a network connection; when
either is missing the caller falls back to the local model.
"""

import re

from .config import CLOUD_LLM_MODEL, get_system_prompt

_client = None


def _words(pieces):
    """Re-cut streamed text into words that carry their leading space, the
    way the local tokenizer delivers them. The sentence splitter in
    ``Brain.stream_think`` relies on a sentence ending at the end of a token."""
    pending = ""
    for piece in pieces:
        *words, pending = re.split(r"(?<=\S)(?=\s)", pending + piece)
        yield from words
    if pending:
        yield pending


def stream(history):
    """Yield the reply to *history* word by word. Raises on any failure."""
    global _client
    import anthropic                # only needed when the cloud backend is on
    if _client is None:
        # Short timeout and no retries: with the network down, the local model should answer soon
        _client = anthropic.Anthropic(timeout=20.0, max_retries=0)

    messages = [{"role": m["role"], "content": m["content"]} for m in history
                if m["role"] != "system" and m["content"]]
    while messages and messages[0]["role"] != "user":       # trimmed history can start with a reply
        messages.pop(0)

    with _client.messages.stream(
        model=CLOUD_LLM_MODEL,
        max_tokens=4096,
        system=get_system_prompt(),             # built per turn, so the time and date in it are current
        thinking={"type": "disabled"},          # a spoken reply: first word fast, fewest tokens
        output_config={"effort": "low"},
        messages=messages,
    ) as response:
        yield from _words(response.text_stream)
