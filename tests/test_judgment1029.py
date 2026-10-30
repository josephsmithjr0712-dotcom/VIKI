"""Tests for the NEEDS-DECISION findings resolved in the fix-1029 pass
with conservative defaults (each flagged for Joey's veto in the report).

Covered here: 7 (cloud circuit breaker), 8 (reset_answer before a
retry/fallback re-streams), 14 (run_one last_run semantics — the
scheduler assertions live in test_tasks_run_now.py), 30b (Spotify
volume device — connector assertions live in test_spotify.py),
33 (risky-tool timeout message), 47 (long-chat char budget),
54 (HA URL fallbacks). Findings 34 (DST wall-clock storage) and 35
(history/thread split) are documented decisions with no behavior
change; 53(a) (whisper defaults) is likewise documented in code.
"""

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as agent_mod
import chats as chats_mod
import connectors.lamp as lamp_mod
import connectors.roku as roku_mod
import tools as tools_mod
from memory import SessionStore
from tools import Toolbox

KEYS = ("VIKI_CLOUD_MODEL", "VIKI_CLOUD_BASE_URL", "VIKI_CLOUD_API_KEY",
        "VIKI_CLOUD_TIMEOUT", "VIKI_CLOUD_BREAKER_FAILURES",
        "VIKI_CLOUD_BREAKER_COOLDOWN", "VIKI_TOOL_TIMEOUT",
        "VIKI_HISTORY_CHAR_BUDGET", "HA_URL", "HA_TOKEN",
        "HA_URL_FALLBACKS")


class EnvGuard(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in KEYS}
        for k in KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class FakeChunk:
    def __init__(self, content="", thinking=None, tool_calls=None):
        self.message = MagicMock()
        self.message.content = content
        self.message.thinking = thinking
        self.message.tool_calls = tool_calls or []


class FakeClient:
    """chat() yields one scripted chunk list per call, or raises."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return [FakeChunk(c) for c in action]


def make_viki(script, tmp):
    store = SessionStore(folder=str(tmp / "session"))
    client = FakeClient(script)
    with patch.object(agent_mod.ollama, "Client", return_value=client):
        v = agent_mod.VIKI(Toolbox(store))
    v.client = client
    return v


class TestCloudBreaker(EnvGuard):
    def test_breaker_opens_after_threshold_and_skips_cloud(self):
        os.environ["VIKI_CLOUD_MODEL"] = "gemini-2.5-flash"
        os.environ["VIKI_CLOUD_API_KEY"] = "k"
        os.environ["VIKI_CLOUD_BREAKER_FAILURES"] = "2"
        os.environ["VIKI_CLOUD_BREAKER_COOLDOWN"] = "300"
        tmp = Path(tempfile.mkdtemp())
        v = make_viki([["local answer"]], tmp)
        cloud = MagicMock(side_effect=RuntimeError("cloud down"))
        v._stream_cloud = cloud
        events = []
        emit = lambda k, p: events.append((k, p))
        # Two failures trip the breaker on the second one.
        out1, _, _ = v._stream([{"role": "user", "content": "hi"}], None, False, emit, threading.Event())
        self.assertEqual(cloud.call_count, 1)
        self.assertIn("local answer", out1)
        v.client = FakeClient([["local two"]])
        out2, _, _ = v._stream([{"role": "user", "content": "hi"}], None, False, emit, threading.Event())
        self.assertEqual(cloud.call_count, 2)
        self.assertIn("local two", out2)
        # Third turn: breaker open, cloud not tried at all.
        v.client = FakeClient([["local again"]])
        out3, _, _ = v._stream([{"role": "user", "content": "hi"}], None, False, emit, threading.Event())
        self.assertEqual(cloud.call_count, 2)  # breaker open: not tried again
        self.assertIn("local again", out3)
        self.assertTrue(any("paused after 2" in p for k, p in events if k == "status"))

    def test_cloud_success_resets_failure_count(self):
        os.environ["VIKI_CLOUD_MODEL"] = "gemini-2.5-flash"
        os.environ["VIKI_CLOUD_API_KEY"] = "k"
        tmp = Path(tempfile.mkdtemp())
        v = make_viki([["local"]], tmp)
        v._stream_cloud = MagicMock(side_effect=RuntimeError("boom"))
        emit = lambda k, p: None
        v._stream([{"role": "user", "content": "hi"}], None, False, emit, threading.Event())
        self.assertEqual(v._cloud_failures, 1)
        v._stream_cloud = MagicMock(return_value=("cloud answer", "", []))
        out, _, _ = v._stream([{"role": "user", "content": "hi"}], None, False, emit, threading.Event())
        self.assertEqual(out, "cloud answer")
        self.assertEqual(v._cloud_failures, 0)


class TestResetOnRetry(EnvGuard):
    def test_cloud_fallback_clears_partial_tokens(self):
        os.environ["VIKI_CLOUD_MODEL"] = "gemini-2.5-flash"
        os.environ["VIKI_CLOUD_API_KEY"] = "k"
        tmp = Path(tempfile.mkdtemp())
        v = make_viki([["local full"]], tmp)

        def flaky_cloud(cfg, messages, tools, think, emit, cancel):
            emit("token", "cloud part ")
            emit("token", "cloud more ")
            raise RuntimeError("cloud died mid-stream")

        v._stream_cloud = flaky_cloud
        events = []
        out, _, _ = v._stream([{"role": "user", "content": "hi"}], None, False,
                              lambda k, p: events.append((k, p)), threading.Event())
        kinds = [k for k, _ in events]
        self.assertIn("reset_answer", kinds)
        # reset comes after the cloud tokens and before the local ones.
        reset_at = kinds.index("reset_answer")
        self.assertEqual([p for k, p in events[:reset_at] if k == "token"],
                         ["cloud part ", "cloud more "])
        self.assertEqual([p for k, p in events[reset_at + 1:] if k == "token"],
                         ["local full"])
        self.assertIn("local full", out)

    def test_local_retry_clears_partial_tokens(self):
        tmp = Path(tempfile.mkdtemp())
        store = SessionStore(folder=str(tmp / "session"))
        with patch.object(agent_mod.ollama, "Client", return_value=MagicMock()):
            v = agent_mod.VIKI(Toolbox(store))

        calls = {"n": 0}

        def chat(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                def gen():
                    yield FakeChunk("part ")
                    raise ConnectionError("dropped")
                return gen()
            return [FakeChunk("full answer")]

        v.client = MagicMock()
        v.client.chat = chat
        events = []
        out, _, _ = v._stream([{"role": "user", "content": "hi"}], None, False,
                              lambda k, p: events.append((k, p)), threading.Event())
        self.assertEqual(out, "full answer")
        kinds = [k for k, _ in events]
        self.assertIn("reset_answer", kinds)
        reset_at = kinds.index("reset_answer")
        self.assertEqual([p for k, p in events[:reset_at] if k == "token"], ["part "])
        self.assertEqual([p for k, p in events[reset_at + 1:] if k == "token"],
                         ["full answer"])


class TestHistoryBudget(EnvGuard):
    def test_budget_drops_oldest_kept_messages(self):
        msgs = [{"role": "user", "content": "x" * 100},
                {"role": "assistant", "content": "y" * 100},
                {"role": "user", "content": "z" * 100}]
        out = chats_mod.context_history(msgs, budget=150)
        self.assertEqual(out, [msgs[2]])

    def test_budget_disabled_keeps_everything(self):
        msgs = [{"role": "user", "content": "x" * 100},
                {"role": "user", "content": "y" * 100}]
        self.assertEqual(chats_mod.context_history(msgs, budget=0), msgs)

    def test_budget_never_drops_summary_or_newest(self):
        msgs = [{"role": "user", "content": f"topic {i} " + "w" * 500}
                for i in range(25)]
        out = chats_mod.context_history(msgs, keep=20, budget=100)
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("Summary", out[0]["content"])
        self.assertEqual(out[-1], msgs[-1])


class TestRiskyTimeout(EnvGuard):
    def _toolbox_with_slow(self, name):
        tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        if name not in tb.enabled:
            tb.enabled.append(name)
        tb.funcs[name] = lambda **kw: __import__("time").sleep(5)
        return tb

    def test_risky_tool_timeout_warns_not_to_retry(self):
        os.environ["VIKI_TOOL_TIMEOUT"] = "0.05"
        tb = self._toolbox_with_slow("viki_apply_update")
        out = tb.run("viki_apply_update", {})
        self.assertIn("may still have gone through", out)
        self.assertIn("do NOT retry", out)

    def test_safe_tool_timeout_suggests_retry(self):
        os.environ["VIKI_TOOL_TIMEOUT"] = "0.05"
        tb = self._toolbox_with_slow("spotify_pause")
        out = tb.run("spotify_pause", {})
        self.assertIn("Try again", out)


class Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class TestHaFallback(EnvGuard):
    def test_tools_falls_through_to_fallback_on_connection_error(self):
        os.environ["HA_URL"] = "http://primary:8123"
        os.environ["HA_TOKEN"] = "tok"
        os.environ["HA_URL_FALLBACKS"] = "http://backup:8123"

        def fake_urlopen(req, timeout=None):
            url = req.full_url
            if "primary" in url:
                raise urllib.error.URLError("primary down")
            return Resp([{"entity_id": "light.x", "state": "on"}])

        with patch.object(tools_mod.urllib.request, "urlopen", side_effect=fake_urlopen):
            out = tools_mod._ha_request("/api/states")
        self.assertEqual(out, [{"entity_id": "light.x", "state": "on"}])

    def test_tools_http_error_does_not_fall_through(self):
        os.environ["HA_URL"] = "http://primary:8123"
        os.environ["HA_TOKEN"] = "tok"
        os.environ["HA_URL_FALLBACKS"] = "http://backup:8123"

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

        with patch.object(tools_mod.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(urllib.error.HTTPError):
                tools_mod._ha_request("/api/states")

    def test_roku_falls_through_to_fallback(self):
        os.environ["HA_URL"] = "http://primary:8123"
        os.environ["HA_TOKEN"] = "tok"
        os.environ["HA_URL_FALLBACKS"] = "http://backup:8123"

        def fake_urlopen(req, timeout=None):
            if "primary" in req.full_url:
                raise urllib.error.URLError("primary down")
            return Resp({"state": "playing"})

        with patch.object(roku_mod.urllib.request, "urlopen", side_effect=fake_urlopen):
            out = roku_mod._state("media_player.roku_tv")
        self.assertEqual(out, {"state": "playing"})

    def test_lamp_falls_through_to_fallback(self):
        os.environ["HA_URL"] = "http://primary:8123"
        os.environ["HA_TOKEN"] = "tok"
        os.environ["HA_URL_FALLBACKS"] = "http://backup:8123"

        def fake_urlopen(req, timeout=None):
            if "primary" in req.full_url:
                raise urllib.error.URLError("primary down")
            return Resp([{"entity_id": "light.desk", "state": "on"}])

        with patch.object(lamp_mod.urllib.request, "urlopen", side_effect=fake_urlopen):
            out = lamp_mod._get_states()
        self.assertEqual(out, [{"entity_id": "light.desk", "state": "on"}])


if __name__ == "__main__":
    unittest.main()
