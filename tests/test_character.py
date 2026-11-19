"""Tests for the character pack and the engine/character separation.

Run with ``pytest tests/test_character.py`` or ``python tests/test_character.py``.
No hardware needed.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import character as character_module  # noqa: E402
from core.character import CharacterError, load_character  # noqa: E402

# Engine code: must not contain any character's name or lines.
ENGINE_FILES = sorted(
    [os.path.join(ROOT, "core", f) for f in os.listdir(os.path.join(ROOT, "core")) if f.endswith(".py")]
    + [os.path.join(ROOT, "agent_hailo.py"), os.path.join(ROOT, "cli_chat.py")]
)

# String literals that contain a pack id or project identifier, not character text.
ALLOWED_LITERALS = {
    "bmo",            # default pack id in core/character.py
    "BMO_LANGUAGE",   # environment variable name
}


def _pack_ids():
    d = os.path.join(ROOT, "characters")
    return sorted(p for p in os.listdir(d) if os.path.exists(os.path.join(d, p, "character.toml")))


def _string_literals(path):
    """Yield (lineno, text) for every string literal that is not a docstring."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            for stmt in body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    docstrings.add(id(stmt.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def test_every_pack_loads():
    assert "bmo" in _pack_ids()
    for pack_id in _pack_ids():
        c = load_character(pack_id)
        assert c.name
        # Every placeholder the engine fills must be accepted
        assert c.prompt("system", time="10:00 AM", date="Monday", language_rule="RULE ")
        assert c.prompt("live_data", user_text="hi")
        assert c.prompt("idle_thought", info="fact")
        for key in ("vision", "search_summary", "idle_system"):
            assert c.prompt(key)


def test_pack_faces_exist():
    """Every phase, expression and persona needs a folder of frames."""
    from core.state import Phase
    for pack_id in _pack_ids():
        c = load_character(pack_id)
        faces = c.asset("faces")
        # display_image shows a picture and screensaver cycles other faces,
        # so neither has frames of its own
        phases = {p.value for p in Phase} - {"display_image", "screensaver"}
        needed = phases | c.expressions | c.personas
        missing = sorted(n for n in needed if not os.path.isdir(os.path.join(faces, n)))
        assert not missing, f"{pack_id}: no faces folder for {missing}"
        unknown = sorted(set(c.idle_expressions) - c.expressions)
        assert not unknown, f"{pack_id}: idle_expressions not in expressions: {unknown}"


def test_missing_pack_and_incomplete_pack_fail_clearly():
    try:
        load_character("no-such-character")
        assert False, "expected CharacterError"
    except CharacterError as e:
        assert "no-such-character" in str(e)
    try:
        character_module.Character("broken", {"identity": {"name": "X"}})
        assert False, "expected CharacterError"
    except CharacterError as e:
        assert "prompts.system" in str(e)


def test_name_normalisation_and_lines():
    c = load_character("bmo")
    assert c.normalize_name("Hey beemo, Bemo and BEEMO") == f"Hey {c.name}, {c.name} and {c.name}"
    assert c.normalize_name("beemoth") == "beemoth"          # whole words only
    assert isinstance(c.line("camera_intros", "en"), list)
    assert c.say("camera_intros", "de") in c.line("camera_intros", "de")
    assert c.line("vision_error", "de") == c.line("vision_error", "en")   # falls back to English
    assert c.line("not_a_key", "en") == "not_a_key"


def test_engine_contains_no_character_text():
    """No string literal in the engine may contain a character's name."""
    words = set()
    for pack_id in _pack_ids():
        c = load_character(pack_id)
        words.update([c.name] + c.heard_as)
    pattern = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(w) for w in sorted(words)) + r")(?![A-Za-z])", re.IGNORECASE)
    offenders = []
    for path in ENGINE_FILES:
        for lineno, text in _string_literals(path):
            if text in ALLOWED_LITERALS:
                continue
            if pattern.search(text):
                offenders.append(f"{os.path.relpath(path, ROOT)}:{lineno}: {text[:70]!r}")
    assert not offenders, "character text in engine code:\n" + "\n".join(offenders)


def test_swapping_the_character_changes_engine_text():
    """A pack with another name must reach the prompt and the fallback lines
    without any engine change."""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "robo"))
        src = open(os.path.join(ROOT, "characters", "bmo", "character.toml"), encoding="utf-8").read()
        robo = re.sub(r"BMO", "Robo", src).replace('heard_as = ["Beemo", "Bemo"]', 'heard_as = ["Rowbo"]')
        open(os.path.join(tmp, "robo", "character.toml"), "w", encoding="utf-8").write(robo)
        code = (
            "import os, sys\n"
            f"sys.path.insert(0, {ROOT!r})\n"
            "import core.character as c\n"
            f"c.CHARACTERS_DIR = {tmp!r}\n"
            "os.environ['CHARACTER'] = 'robo'\n"
            "from core.config import get_system_prompt\n"
            "from core.actions import clean_llm_response, inject_search_context\n"
            "p = get_system_prompt()\n"
            "assert 'Your name is Robo' in p and 'BMO' not in p, p[:200]\n"
            "assert 'Robo' in clean_llm_response('   ')\n"
            "assert clean_llm_response('Rowbo is here') == 'Robo is here'\n"
            "assert 'as Robo' in inject_search_context('q', 'r')\n"
            "print('swap ok')\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0 and "swap ok" in out.stdout, out.stderr[-800:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
