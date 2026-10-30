"""Tests for Ollama connection retry in agent._stream (v0.2.15). Ollama is mocked."""
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChunk:
    def __init__(self, message):
        self.message = message


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        kind, payload = self.script.pop(0)
        if kind == "raise":
            raise payload
        return [FakeChunk(m) for m in payload]


class FakeRequestError(Exception):
    """Stand-in for ollama.RequestError (absent in the sandbox stub)."""


def make_viki(script, tmp, **env_overrides):
    store = SessionStore(folder=tmp / "session")
    for k, v in env_overrides.items():
        os.environ[k] = v
    try:
        with patch.object(agent_mod.ollama, "Client", return_value=FakeClient(script)):
            v = agent_mod.VIKI(Toolbox(store))
    finally:
        for k in env_overrides:
            del os.environ[k]
    return v


def run_loop(viki, prompt):
    events = []
    out = viki.run(
        prompt, [], [], lambda k, p: events.append((k, p)),
        lambda cid, n, a: True, think=False, cancel=threading.Event(),
    )
    return out, events


class TestRetryDefaults(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_defaults(self):
        v = make_viki([], self.tmp)
        self.assertEqual(v.retries, 2)
        self.assertEqual(v.retry_delay, 1.0)

    def test_env_config(self):
        v = make_viki([], self.tmp, VIKI_RETRIES="4", VIKI_RETRY_DELAY="0.25")
        self.assertEqual(v.retries, 4)
        self.assertEqual(v.retry_delay, 0.25)

    def test_bad_env_falls_back_to_defaults(self):
        v = make_viki([], self.tmp, VIKI_RETRIES="abc", VIKI_RETRY_DELAY="-1")
        self.assertEqual(v.retries, 2)
        self.assertEqual(v.retry_delay, 0.0)  # negative clamped to 0, parse OK


class TestRetryBehaviour(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_transient_failure_then_success(self):
        v = make_viki([
            ("raise", ConnectionError("reset by peer")),
            ("raise", ConnectionError("reset by peer")),
            ("chunks", [FakeMsg(content="recovered")]),
        ], self.tmp)
        # Retry sleeps go through cancel.wait (sliced, Stop-aware), not
        # time.sleep: a mock cancel records the backoff without waiting.
        cancel = MagicMock()
        cancel.wait.return_value = False
        cancel.is_set.return_value = False
        events = []
        out = v.run("hi", [], [], lambda k, p: events.append((k, p)),
                    lambda cid, n, a: True, think=False, cancel=cancel)
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["content"], "recovered")
        self.assertEqual(v.client.calls, 3)
        retry_statuses = [p for k, p in events if k == "status" and "retrying" in str(p)]
        self.assertEqual(len(retry_statuses), 2)
        # exponential backoff: 1.0, then 2.0
        self.assertEqual([c.args[0] for c in cancel.wait.call_args_list], [1.0, 2.0])

    def test_retries_exhausted_surfaces_error(self):
        v = make_viki([
            ("raise", ConnectionError("down")),
            ("raise", ConnectionError("down")),
            ("raise", ConnectionError("down")),
        ], self.tmp, VIKI_RETRIES="2", VIKI_RETRY_DELAY="0")
        with patch.object(agent_mod, "time"):
            out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "error")
        self.assertIn("Could not reach the model", out["content"])
        self.assertEqual(v.client.calls, 3)  # 1 initial + 2 retries

    def test_zero_retries_raises_immediately(self):
        v = make_viki([("raise", ConnectionError("down"))], self.tmp, VIKI_RETRIES="0", VIKI_RETRY_DELAY="0")
        with patch.object(agent_mod, "time") as mock_time:
            out, events = run_loop(v, "hi")
        self.assertEqual(out["type"], "error")
        self.assertEqual(v.client.calls, 1)
        self.assertFalse(mock_time.sleep.called)

    def test_non_transient_error_not_retried(self):
        v = make_viki([("raise", RuntimeError("boom"))], self.tmp, VIKI_RETRY_DELAY="0")
        with patch.object(agent_mod, "time"):
            out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "error")
        self.assertEqual(v.client.calls, 1)

    def test_request_error_class_is_retried(self):
        # Documents the getattr wiring: when the real ollama.RequestError
        # exists (patched in here), it lands in the transient set.
        v = make_viki([
            ("raise", FakeRequestError("http fail")),
            ("chunks", [FakeMsg(content="ok")]),
        ], self.tmp, VIKI_RETRY_DELAY="0")
        with patch.object(agent_mod, "_TRANSIENT_ERRORS", (FakeRequestError, OSError)):
            with patch.object(agent_mod, "time"):
                out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["content"], "ok")
        self.assertEqual(v.client.calls, 2)

    def test_success_first_try_no_status_noise(self):
        v = make_viki([("chunks", [FakeMsg(content="hi boss")])], self.tmp)
        out, events = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        self.assertFalse([p for k, p in events if k == "status" and "retrying" in str(p)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
