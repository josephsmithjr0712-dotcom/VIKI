"""Tests for the agent's repeat-call guard (v0.2.18). Ollama is mocked.

If the model emits the same tool call with identical arguments over and
over, the agent nudges it after the 2nd identical call and stops
re-executing from the 3rd identical call on, serving the cached result.
"""
import os
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
        self.calls = 0
        self.seen_messages = []

    def chat(self, **kw):
        self.calls += 1
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


def run_loop(viki, prompt, approve_ok=True):
    events = []
    approvals = []

    def approve(cid, name, args):
        approvals.append((name, args))
        return approve_ok

    out = viki.run(
        prompt, [], [], lambda k, p: events.append((k, p)),
        approve, think=False, cancel=threading.Event(),
    )
    return out, events, approvals


def call_chunks(name, args, n_turns):
    return [("chunks", [FakeMsg(tool_calls=[FakeCall(name, args)])])
           for _ in range(n_turns)]


class TestRepeatCallGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _viki_with_counted_run(self, script):
        v = make_viki(script, self.tmp)
        runs = []
        orig_run = v.toolbox.run
        v.toolbox.run = lambda n, a: (runs.append((n, a)), orig_run(n, a))[1]
        return v, runs

    def test_second_identical_call_executes_and_nudges(self):
        script = call_chunks("calculate", {"expression": "1+1"}, 2) + [
            ("chunks", [FakeMsg(content="done")]),
        ]
        v, runs = self._viki_with_counted_run(script)
        out, events, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        # Both executions happened...
        self.assertEqual(len(runs), 2)
        # ...and the model saw a nudge before its next turn.
        nudge_turns = [
            m for msgs in v.client.seen_messages for m in (msgs or [])
            if m.get("role") == "user"
            and m.get("content", "").startswith("System note:")
            and "twice in a row" in m.get("content", "")
        ]
        self.assertEqual(len(nudge_turns), 1)
        self.assertIn("calculate", nudge_turns[0]["content"])

    def test_third_identical_call_serves_cached_result(self):
        script = call_chunks("calculate", {"expression": "2+2"}, 3) + [
            ("chunks", [FakeMsg(content="done")]),
        ]
        v, runs = self._viki_with_counted_run(script)
        out, events, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        # Only two real executions; the 3rd was served from cache.
        self.assertEqual(len(runs), 2)
        third_result = [p for k, p in events
                        if k == "tool_result"][2]["result"]
        self.assertIn("identical arguments", third_result)
        self.assertIn("Do not call it again", third_result)
        skipped = [p for k, p in events
                   if k == "status" and "Skipped calculate" in str(p)]
        self.assertEqual(len(skipped), 1)

    def test_different_calls_reset_the_counter(self):
        script = (call_chunks("calculate", {"expression": "1+1"}, 1)
                  + call_chunks("calculate", {"expression": "2+2"}, 1)
                  + call_chunks("calculate", {"expression": "1+1"}, 1)
                  + [("chunks", [FakeMsg(content="done")])])
        v, runs = self._viki_with_counted_run(script)
        out, events, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        self.assertEqual(len(runs), 3)  # no skips: nothing repeated consecutively
        nudges = [p for k, p in events if k == "status" and "Skipped" in str(p)]
        self.assertEqual(nudges, [])

    def test_identical_calls_inside_one_batch_count(self):
        script = [("chunks", [FakeMsg(tool_calls=[
            FakeCall("calculate", {"expression": "3+3"}),
            FakeCall("calculate", {"expression": "3+3"}),
            FakeCall("calculate", {"expression": "3+3"}),
        ])])] + [("chunks", [FakeMsg(content="done")])]
        v, runs = self._viki_with_counted_run(script)
        out, events, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        self.assertEqual(len(runs), 2)  # 3rd in the same batch was cached

    def test_denied_repeat_does_not_reprompt_approval(self):
        script = call_chunks("write_file", {"path": "x.txt", "content": "y"}, 3) + [
            ("chunks", [FakeMsg(content="done")]),
        ]
        v = make_viki(script, self.tmp)
        out, events, approvals = run_loop(v, "hi", approve_ok=False)
        self.assertEqual(out["type"], "text")
        # Approval asked twice (1st + 2nd call), not a third time: the 3rd
        # identical denied call is served from cache.
        self.assertEqual(len(approvals), 2)
        third_result = [p for k, p in events
                        if k == "tool_result"][2]["result"]
        self.assertIn("identical arguments", third_result)

    def test_args_key_is_order_insensitive(self):
        self.assertEqual(
            agent_mod._args_key({"b": 1, "a": 2}),
            agent_mod._args_key({"a": 2, "b": 1}),
        )
        self.assertNotEqual(
            agent_mod._args_key({"a": 2}),
            agent_mod._args_key({"a": 3}),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
