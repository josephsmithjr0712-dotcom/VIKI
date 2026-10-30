"""Empty-answer guard (v0.3.39).

A model turn that emits only reasoning — stripped to "" by _ThinkFilter —
or nothing at all used to return "" outright: the web UI showed
"(no answer)" and the clock spoke nothing and went back to Listening
(the 2026-10-29 evening failure, seen on both web and clock). These
tests pin the guard: one retry with a direct-answer nudge, then an
honest fallback sentence instead of silence. A normal answer is
untouched and costs no extra turn.
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import ollama
except ImportError:
    import types

    ollama = types.ModuleType("ollama")

    class ResponseError(Exception):
        pass

    ollama.ResponseError = ResponseError
    ollama.Client = None
    sys.modules["ollama"] = ollama

try:
    from dotenv import load_dotenv  # noqa: F401
except ImportError:
    import types

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = dotenv

import agent as agent_mod
from memory import SessionStore
from tools import Toolbox


class FakeMsg:
    def __init__(self, content=""):
        self.content = content
        self.tool_calls = []
        self.thinking = None


class FakeChunk:
    def __init__(self, message):
        self.message = message


class SequenceClient:
    """Answers each chat() call with the next scripted message list."""

    def __init__(self, turns):
        self.turns = [list(t) for t in turns]
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        msgs = self.turns[min(self.calls - 1, len(self.turns) - 1)]
        return [FakeChunk(m) for m in msgs]


def make_viki(client, tmp):
    store = SessionStore(folder=tmp / "session")
    with patch.object(agent_mod.ollama, "Client", return_value=client):
        v = agent_mod.VIKI(Toolbox(store))
    return v


def run(client, tmp):
    v = make_viki(client, tmp)
    return v.run("hello", [], [], lambda k, p: None,
                 lambda cid, n, a: True, think=False,
                 cancel=threading.Event())


class TestEmptyAnswer(unittest.TestCase):
    def test_empty_then_answer(self):
        client = SequenceClient([[FakeMsg(content="")],
                                 [FakeMsg(content="Got it.")]])
        out = run(client, Path(tempfile.mkdtemp()))
        self.assertEqual(out["content"], "Got it.")
        self.assertEqual(client.calls, 2)

    def test_reasoning_only_then_answer(self):
        # v0.3.38 strips the <think> block, leaving ""; the guard must
        # retry instead of returning that empty string.
        client = SequenceClient([[FakeMsg(content="<think>deliberation</think>")],
                                 [FakeMsg(content="Hey boss!")]])
        out = run(client, Path(tempfile.mkdtemp()))
        self.assertEqual(out["content"], "Hey boss!")
        self.assertEqual(client.calls, 2)

    def test_always_empty_gives_fallback_not_silence(self):
        client = SequenceClient([[FakeMsg(content="")]])
        out = run(client, Path(tempfile.mkdtemp()))
        self.assertEqual(out["content"], agent_mod._EMPTY_ANSWER_FALLBACK)
        self.assertNotEqual(out["content"], "")
        self.assertEqual(client.calls, 2)  # initial + exactly one retry

    def test_normal_answer_costs_no_extra_turn(self):
        client = SequenceClient([[FakeMsg(content="Hey boss!")]])
        out = run(client, Path(tempfile.mkdtemp()))
        self.assertEqual(out["content"], "Hey boss!")
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
