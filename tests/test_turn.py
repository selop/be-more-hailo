"""Headless tests for the turn loop (core/turn.py).

Run with ``pytest tests/test_turn.py`` or ``python tests/test_turn.py``.
Hardware is replaced by fakes: no mic, speaker, NPU or display needed.
"""

import os
import sys
import tempfile
import threading
import time

# Traces from these tests go to a temporary folder, never into the project
os.environ["TRACE_DIR"] = tempfile.mkdtemp(prefix="bmo-test-trace-")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["LLM_BACKEND"] = "local"      # a device's .env may select the hosted model; tests never call it

from core import trace, tts, turn  # noqa: E402
from core.control import ControlServer  # noqa: E402
from core.config import say, t  # noqa: E402
from core.llm import Brain  # noqa: E402
from core.state import Phase, StateStore  # noqa: E402


class NoSleepTime:
    """Stands in for the ``time`` module inside core.turn: real clock, no waiting."""
    time = staticmethod(time.time)

    @staticmethod
    def sleep(seconds):
        pass


class FakeSynth:
    def synthesize(self, text):
        return b"\x00\x00" * 100


class FakePlayer:
    def __init__(self):
        self.played = 0

    def resume(self):
        pass

    def play(self, pcm):
        self.played += 1


class FakeSounds:
    def __init__(self):
        self.played = []

    def play(self, category):
        self.played.append(category)
        return None

    def get_voice_files(self):
        return []


class FakeUI(turn.TurnUI):
    def __init__(self):
        self.calls = []

    def set_status(self, text):
        self.calls.append(("status", text))

    def show_user_prompt(self, text):
        self.calls.append(("prompt", text))

    def take_photo(self, user_text):
        self.calls.append(("photo", user_text))

    def show_image(self, url):
        self.calls.append(("image", url))

    def boot_failed(self, error):
        self.calls.append(("boot_failed", str(error)))


class ScriptedBrain:
    """Yields fixed chunks the way ``Brain.stream_think`` does."""

    def __init__(self, chunks):
        self.chunks = chunks

    def stream_think(self, user_text):
        yield from self.chunks


class Harness:
    def __init__(self, brain=None):
        self.store = StateStore()
        self.sounds = FakeSounds()
        self.ui = FakeUI()
        self.player = FakePlayer()
        self.spoken = []
        self.phases = []
        self.store.on_phase_change(lambda old, new, msg: self.phases.append(new))
        self.conv = turn.Conversation(
            store=self.store, sounds=self.sounds, ui=self.ui,
            stop_event=threading.Event(), brain=brain if brain is not None else Brain(),
        )
        # Record what is spoken while still running the real speak()
        real_speak = self.conv.speak

        def speak(text, msg="Speaking..."):
            self.spoken.append(text)
            real_speak(text, msg)
        self.conv.speak = speak

    def __enter__(self):
        self._saved = (turn.get_synthesizer, turn.get_player, turn.time, tts.save_pronunciations)
        turn.get_synthesizer = lambda: FakeSynth()
        turn.get_player = lambda: self.player
        turn.time = NoSleepTime                # only the turn module skips its waits
        tts.save_pronunciations = lambda pronunciations: None   # never write to the project
        return self

    def __exit__(self, *exc):
        turn.get_synthesizer, turn.get_player, turn.time, tts.save_pronunciations = self._saved


def test_reply_is_spoken_sentence_by_sentence_and_ends_idle():
    brain = ScriptedBrain(["Hello there.", "How are you?"])
    with Harness(brain) as h:
        h.conv.respond("hi")
        assert h.spoken == ["Hello there.", "How are you?"]
        assert h.player.played == 2
        assert Phase.SPEAKING in h.phases
        assert h.store.phase == Phase.IDLE
        assert h.store.last_spoke_at > 0


def test_muted_reply_is_not_played():
    with Harness(ScriptedBrain(["Hello there."])) as h:
        h.store.set_muted(True)
        h.conv.respond("hi")
        assert h.player.played == 0


def test_timer_music_and_expression_actions_are_routed():
    brain = ScriptedBrain([
        'Okay! {"action": "set_timer", "minutes": 5, "message": "Tea is ready"}',
        '{"action": "play_music"}',
        '{"action": "set_expression", "value": "happy"}',
    ])
    with Harness(brain) as h:
        timers, music = [], []
        h.conv.start_timer = lambda minutes, message: timers.append((minutes, message))
        h.conv.start_music = lambda: music.append(True)
        result = h.conv.respond("set a timer")
        assert timers == [(5.0, "Tea is ready")]
        assert music == [True] and result.music_triggered
        assert ("status", "Feeling happy...") in h.ui.calls
        assert h.spoken == ["Okay!"]            # JSON is never spoken


def test_spoken_timer_request_starts_a_timer_and_confirms_it():
    with Harness() as h:                        # real Brain: recognised before the LLM
        timers = []
        h.conv.start_timer = lambda minutes, message: timers.append((minutes, message))
        injection = h.conv.inject(text="Set a timer for 90 seconds.")
        h.conv.run_turn(oww=None)
        assert timers == [(1.5, say("timer_done"))]
        assert len(h.spoken) == 1 and "1 minute and 30 seconds" in h.spoken[0]
        assert injection.result["route"] == "keyword"
        assert injection.result["actions"][0]["action"] == "set_timer"


def test_reminder_message_is_what_gets_spoken_when_it_is_due():
    with Harness() as h:
        timers = []
        h.conv.start_timer = lambda minutes, message: timers.append((minutes, message))
        h.conv.respond("remind me in ten minutes to call my mom")
        assert timers[0][0] == 10 and "call your mom" in timers[0][1]


def test_timer_action_with_spoken_reply_is_not_confirmed_twice():
    brain = ScriptedBrain(['Okay, timer started! {"action": "set_timer", "minutes": 2}'])
    with Harness(brain) as h:
        h.conv.start_timer = lambda minutes, message: None
        h.conv.respond("timer please")
        assert h.spoken == ["Okay, timer started!"]


def test_running_timers_are_listed_soonest_first_until_they_fire():
    with Harness() as h:
        long_timer = h.conv._add_timer(10, "ten minutes")
        short_timer = h.conv._add_timer(0.5, "thirty seconds")
        timers = h.conv.active_timers()
        assert [t["message"] for t in timers] == ["thirty seconds", "ten minutes"]
        assert 29 <= timers[0]["remaining_s"] <= 30 and timers[0]["total_s"] == 30
        h.conv._remove_timer(short_timer)
        assert [t["message"] for t in h.conv.active_timers()] == ["ten minutes"]
        h.conv._remove_timer(long_timer)
        assert h.conv.active_timers() == []


def test_photo_request_reaches_the_ui():
    with Harness() as h:                        # real Brain: keyword route, no LLM
        h.conv.respond("take a photo")
        assert ("photo", "take a photo") in h.ui.calls
        assert h.spoken == []



def test_image_request_shows_the_image():
    with Harness() as h:
        h.conv.respond("draw me a cat")
        images = [c for c in h.ui.calls if c[0] == "image"]
        assert len(images) == 1 and "pollinations" in images[0][1] and "cat" in images[0][1]
        assert h.spoken == [t("draw_image")]
        assert h.store.phase == Phase.DISPLAY_IMAGE


def test_full_turn_from_wake_word_to_idle():
    with Harness(ScriptedBrain(["It is sunny."])) as h:
        saved = (turn.wait_for_wakeword, turn.record_until_silence, turn.transcribe_audio)
        turn.wait_for_wakeword = lambda *a, **k: True
        turn.record_until_silence = lambda *a, **k: "input.wav"
        turn.transcribe_audio = lambda path: "how is the weather in my head"
        try:
            h.conv.run_turn(oww=None)
        finally:
            turn.wait_for_wakeword, turn.record_until_silence, turn.transcribe_audio = saved
        assert h.phases[:2] == [Phase.LISTENING, Phase.THINKING]
        assert ("prompt", "how is the weather in my head") in h.ui.calls
        assert h.spoken == ["It is sunny."]
        assert h.store.phase == Phase.IDLE
        assert "ack_sounds" in h.sounds.played


def test_turn_with_nothing_heard_returns_to_idle_silently():
    with Harness(ScriptedBrain(["should not be said"])) as h:
        saved = (turn.wait_for_wakeword, turn.record_until_silence)
        turn.wait_for_wakeword = lambda *a, **k: True
        turn.record_until_silence = lambda *a, **k: None
        try:
            h.conv.run_turn(oww=None)
        finally:
            turn.wait_for_wakeword, turn.record_until_silence = saved
        assert h.spoken == [] and h.store.phase == Phase.IDLE


def test_injected_text_turn_is_traced():
    with Harness(ScriptedBrain(["It is sunny.", "Enjoy!"])) as h:
        injection = h.conv.inject(text="how is the weather")
        h.conv.run_turn(oww=None)               # no wake word needed
        assert injection.done.is_set()
        turn_trace = injection.result
        assert turn_trace["source"] == "text" and turn_trace["status"] == "ok"
        assert turn_trace["transcript"] == "how is the weather"
        assert [s["text"] for s in turn_trace["spoken"]] == ["It is sunny.", "Enjoy!"]
        assert "first_audio" in turn_trace["marks"] and "response_ms" in turn_trace
        assert [p[1] for p in turn_trace["phases"]][:2] == ["thinking", "speaking"]
        assert trace.read("turns.jsonl", last=1)[0]["id"] == turn_trace["id"]
        assert h.store.phase == Phase.IDLE


def test_quiet_injection_plays_nothing_and_restores_mute():
    with Harness(ScriptedBrain(["Hello."])) as h:
        injection = h.conv.inject(text="hi", quiet=True)
        h.conv.run_turn(oww=None)
        assert h.player.played == 0
        assert h.store.muted is False
        assert injection.result["spoken"][0]["text"] == "Hello."


def test_keyword_route_and_actions_appear_in_the_trace():
    with Harness() as h:                        # real Brain, keyword route
        injection = h.conv.inject(text="take a photo")
        h.conv.run_turn(oww=None)
        assert injection.result["route"] == "keyword"
        assert {"action": "take_photo"} in injection.result["actions"]


def test_a_failing_turn_is_traced_and_the_loop_survives():
    class BrokenBrain:
        def stream_think(self, user_text):
            raise RuntimeError("npu went away")
            yield  # pragma: no cover
    with Harness(BrokenBrain()) as h:
        injection = h.conv.inject(text="hello")
        h.conv.run_turn(oww=None)               # must not raise
        assert injection.result["status"] == "error"
        assert "npu went away" in injection.result["error"]
        assert h.store.phase == Phase.IDLE


def test_control_channel_runs_a_turn_and_reports_status():
    import json
    import socket
    sock_path = os.path.join(tempfile.mkdtemp(prefix="bmo-ctl-"), "c.sock")
    with Harness(ScriptedBrain(["Pong."])) as h:
        ControlServer(h.conv, sock_path).start()

        def call(payload):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect(sock_path)
                s.sendall((json.dumps(payload) + "\n").encode())
                return json.loads(s.makefile().readline())

        # The turn loop normally picks the injection up; run it here by hand
        runner = threading.Thread(target=lambda: (_wait_for_injection(h.conv), h.conv.run_turn(oww=None)))
        runner.start()
        reply = call({"cmd": "say", "text": "ping", "quiet": True, "timeout": 10})
        runner.join(timeout=10)
        assert reply["ok"] and reply["turn"]["spoken"][0]["text"] == "Pong."

        status = call({"cmd": "status"})
        assert status["ok"] and status["status"]["phase"] == "idle"
        assert status["status"]["last_turn"]["transcript"] == "ping"
        assert call({"cmd": "say", "text": "  "})["ok"] is False
        assert call({"cmd": "nonsense"})["ok"] is False
        assert (os.stat(sock_path).st_mode & 0o777) == 0o600


def _wait_for_injection(conv):
    for _ in range(500):
        if not conv._injections.empty():
            return
        time.sleep(0.01)


def test_boot_failure_is_reported_instead_of_hanging():
    with Harness(ScriptedBrain([])) as h:
        def failing_boot():
            raise RuntimeError("model load timed out")
        h.conv.boot = failing_boot
        h.conv.run()                             # must return, not raise
        assert ("boot_failed", "model load timed out") in h.ui.calls
        assert h.store.phase == Phase.ERROR


def test_recording_ends_after_speech_even_in_a_loud_room():
    from core.audio_input import Endpointer

    def seconds_recorded(levels, floor):
        e = Endpointer(floor)
        for n, level in enumerate(levels, 1):
            if e.feed(level, 0.03):
                return round(n * 0.03, 1), e.ended, e.spoke
        return None

    room = 240                                              # the fan on its loud step
    speech = [room] * 20 + [1200] * 50 + [room + 60] * 400   # 0.6 s wait, 1.5 s speech, then the fan
    assert seconds_recorded(speech, room) == (3.0, "silence", True)
    assert seconds_recorded([room + 60] * 400, room) == (4.0, "no_speech", False)
    assert seconds_recorded([room] * 20 + [5000] + [room] * 400, room) == (4.0, "no_speech", False)   # a click
    assert seconds_recorded([1200] * 1000, room) == (15.0, "max", True)


def test_mute_stops_playback_and_unmute_returns_to_idle():
    class NoShell:
        calls = []

        @staticmethod
        def run(cmd, **kwargs):
            NoShell.calls.append(cmd)

    with Harness() as h:
        saved, turn.subprocess = turn.subprocess, NoShell
        try:
            h.player.stopped = 0
            h.player.stop_playback = lambda: setattr(h.player, "stopped", h.player.stopped + 1)
            h.conv.toggle_mute()
            assert h.store.muted and h.store.phase == Phase.SHHH
            assert h.player.stopped == 1 and NoShell.calls == [["killall", "-9", "aplay"]]
            h.conv.toggle_mute()
            assert not h.store.muted and h.store.phase == Phase.IDLE and h.store.expression == "happy"
        finally:
            turn.subprocess = saved


def screensaver_tick(rolls):
    """Run one Screensaver.tick() at noon after a long idle, with the given
    dice rolls. Returns what it did, in order, and the screensaver."""
    import datetime as real_datetime
    from core import screensaver as module

    class Clock:
        @staticmethod
        def time():
            return time.time() + 3600        # long idle

        @staticmethod
        def sleep(seconds):
            pass

    class Noon:
        class datetime:
            @staticmethod
            def now():
                return real_datetime.datetime(2026, 1, 1, 12, 0)

    class Dice:
        @staticmethod
        def random():
            return rolls.pop(0)

        @staticmethod
        def choice(items):
            return sorted(items)[0]

    store, did = StateStore(), []

    def set_state(name, msg=""):
        did.append(name)
        try:
            store.set_phase(Phase(name), msg=msg)
        except ValueError:
            store.set_expression(name)

    saver = module.Screensaver(store=store, set_state=set_state, speak=lambda text, msg="": did.append(("spoke", text)),
                               show_image=None, get_llm=lambda: None, stop_event=threading.Event())
    saved = (module.time, module.datetime, module.random)
    module.time, module.datetime, module.random = Clock, Noon, Dice
    try:
        saver.tick()                          # not in the screensaver phase: nothing
        assert did == []
        store.set_phase(Phase.SCREENSAVER)
        saver.tick()
    finally:
        module.time, module.datetime, module.random = saved
    return did, saver


def test_screensaver_shows_an_idle_face_then_returns():
    from core.config import CHARACTER
    did, _ = screensaver_tick([0.05, 0.5])                 # an idle face, then no spoken thought
    assert did == [sorted(CHARACTER.idle_expressions)[0], "screensaver"]


def test_screensaver_speaks_one_stock_line_when_the_model_is_not_loaded():
    from core.config import CHARACTER
    before = time.time()
    did, saver = screensaver_tick([0.5, 0.5, 0.01])        # no face, no gag, a spoken thought
    assert did == [("spoke", sorted(CHARACTER.fallback_phrases)[0]), "screensaver"]
    assert saver.last_audio_time > before + 3000           # stamped, so the next thought waits


def test_transcript_with_leaked_decoder_tokens_counts_as_nothing_heard():
    from core.stt import _clean_transcription
    assert _clean_transcription("Wir<|de|>ga<|de|>") == ""
    assert _clean_transcription("<|de|>ga") == ""
    assert _clean_transcription("Wie spät ist es gerade?") == "Wie spät ist es gerade?"


def test_german_requests_take_the_same_shortcuts_as_english():
    from core.actions import detect_pre_llm_action, needs_web_search, vlm_question
    assert detect_pre_llm_action("Mach ein Foto von mir.") == {"action": "take_photo"}
    assert detect_pre_llm_action("Was siehst du?") == {"action": "take_photo"}
    assert detect_pre_llm_action("Spiel ein Lied für mich.") == {"action": "play_music"}
    picture = detect_pre_llm_action("Mal mir ein Bild von einer Katze.")
    assert picture["action"] == "display_image" and "einer%20Katze" in picture["image_url"], picture
    assert vlm_question("Mach ein Foto von mir.").startswith("What do you see")
    assert needs_web_search("Wie wird das Wetter morgen in Berlin?")
    # Ordinary German sentences take no shortcut
    for sentence in ("Guten Morgen, wie geht es dir heute?", "Wie spät ist es gerade?",
                     "Erzähl mir einen Witz über Roboter.", "Was ist die Hauptstadt von Frankreich?"):
        assert detect_pre_llm_action(sentence) is None and not needs_web_search(sentence), sentence


def test_following_sentences_are_synthesised_while_the_first_one_plays():
    events = []

    class SlowSynth:
        def synthesize(self, text):
            events.append(("synth", text))
            return b"\x00\x00" * 100

    class Player(FakePlayer):
        def play(self, pcm):
            real_time.sleep(0.05)                 # "playing": time for the worker to get ahead
            events.append(("play", sum(1 for kind, _ in events if kind == "synth")))
            super().play(pcm)

    import time as real_time
    with Harness(ScriptedBrain(["One.", "Two.", "Three."])) as h:
        h.player = Player()
        turn.get_player = lambda: h.player
        synth = SlowSynth()
        turn.get_synthesizer = lambda: synth
        h.conv.respond("go")
        assert h.spoken == ["One.", "Two.", "Three."]
        assert [text for kind, text in events if kind == "synth"] == ["One.", "Two.", "Three."]     # each once, in order
        plays = [count for kind, count in events if kind == "play"]
        assert plays[0] == 3, events             # by the end of the first sentence, all three were ready


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                print(f"FAIL {name}: {e!r}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
