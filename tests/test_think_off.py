"""Thinking must be sent explicitly to Ollama (v0.3.37).

Omitting ``think`` let the model default win: qwen3 defaults thinking ON,
so turns with reasoning off still paid for a long reasoning trace before
answering (a ~2 min black-hole summary on a GTX 1060). These tests pin
that _stream passes think=False through to client.chat when the turn
asks for no reasoning, and think=True when it does.
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


class RecordingClient:
    """Records every chat() kwargs dict, answers with one short chunk."""

    def __init__(self):
        self.kwargs = []

    def chat(self, **kw):
        self.kwargs.append(kw)
        return [FakeChunk(FakeMsg(content="ok"))]


def make_viki(client, tmp):
    store = SessionStore(folder=tmp / "session")
    with patch.object(agent_mod.ollama, "Client", return_value=client):
        v = agent_mod.VIKI(Toolbox(store))
    return v


class TestThinkOff(unittest.TestCase):
    def test_think_false_sent_explicitly(self):
        client = RecordingClient()
        v = make_viki(client, Path(tempfile.mkdtemp()))
        out = v.run("hello", [], [], lambda k, p: None,
                    lambda cid, n, a: True, think=False,
                    cancel=threading.Event())
        self.assertEqual(out["type"], "text")
        self.assertTrue(client.kwargs)
        self.assertIn("think", client.kwargs[0])
        self.assertIs(client.kwargs[0]["think"], False)

    def test_think_true_still_sent(self):
        client = RecordingClient()
        v = make_viki(client, Path(tempfile.mkdtemp()))
        out = v.run("hello", [], [], lambda k, p: None,
                    lambda cid, n, a: True, think=True,
                    cancel=threading.Event())
        self.assertEqual(out["type"], "text")
        self.assertTrue(client.kwargs)
        self.assertIs(client.kwargs[0]["think"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
