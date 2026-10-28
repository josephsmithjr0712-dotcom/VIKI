"""Tests for the agent's tool-result history cap (v0.3.20). Ollama is mocked.

A huge tool result (a long web page, a big file read) used to land
verbatim in the model history of every later turn, bloating context and
slowing the small local model. History now keeps the first
_MAX_TOOL_RESULT_CHARS plus a marker; short results pass through untouched.
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


class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeCall:
    def __init__(self, name, arguments):
        self.function = FakeFunc(name, arguments)


class FakeMsg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChunk:
    def __init__(self, message):
        self.message = message


class FakeClient:
    """Scripted chat responses; records the messages kwarg of every call."""

    def __init__(self, script):
        self.script = list(script)
        self.seen_messages = []

    def chat(self, **kw):
        # Snapshot: run() keeps appending to the same messages list, so a
        # bare reference would show the final state for every turn.
        self.seen_messages.append(list(kw.get("messages") or []))
        kind, payload = self.script.pop(0)
        if kind == "raise":
            raise payload
        return [FakeChunk(m) for m in payload]


def make_viki(script, tmp):
    store = SessionStore(folder=tmp / "session")
    client = FakeClient(script)
    with patch.object(agent_mod.ollama, "Client", return_value=client):
        v = agent_mod.VIKI(Toolbox(store))
    v.client = client
    return v


def run_loop(viki, prompt):
    events = []

    def approve(cid, name, args):
        return True

    out = viki.run(
        prompt, [], [], lambda k, p: events.append((k, p)),
        approve, think=False, cancel=threading.Event(),
    )
    return out, events


def tool_then_text_script(name, args):
    return [("chunks", [FakeMsg(tool_calls=[FakeCall(name, args)])]),
            ("chunks", [FakeMsg(content="done")])]


class TestToolResultTruncation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _tool_message_in_next_turn(self, result_text):
        v = make_viki(tool_then_text_script("calculate", {"expression": "1+1"}), self.tmp)
        v.toolbox.run = lambda n, a: result_text
        out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        # Second turn's messages carry the tool result in history.
        second_turn = v.client.seen_messages[1]
        tool_msgs = [m for m in second_turn if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        return tool_msgs[0]["content"]

    def test_long_tool_result_truncated_in_history(self):
        content = self._tool_message_in_next_turn("x" * 10000)
        self.assertIn("[tool result truncated:", content)
        self.assertIn("showing first 4000 of 10000 characters", content)
        self.assertTrue(content.startswith("x" * 4000))
        self.assertLess(len(content), 10000)

    def test_short_tool_result_untouched(self):
        content = self._tool_message_in_next_turn("4")
        self.assertEqual(content, "4")

    def test_truncation_boundary(self):
        limit = agent_mod._MAX_TOOL_RESULT_CHARS
        self.assertEqual(agent_mod._truncate_tool_result("y" * limit), "y" * limit)
        over = agent_mod._truncate_tool_result("y" * (limit + 1))
        self.assertIn("[tool result truncated:", over)
        self.assertTrue(over.startswith("y" * limit))


if __name__ == "__main__":
    unittest.main(verbosity=2)
