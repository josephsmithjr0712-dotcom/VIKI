"""Tests for the v0.3.40 audit fix batch (fix-1029).

Covers the SAFE-TO-FIX findings: _ThinkFilter depth/stray/case handling
and flush tail, cloud config (finite timeout, bare gemini), cloud message
leftover calls, cloud tool-call indexing, cloud think status, max-steps
clamp, tools_ok mirroring, malformed tool calls, deferred repeat-guard
note, wrap-up guard, generic error text, cancel-aware retries, scheduler
validation (bool/float every_minutes, action validation, fired-once
reschedule/duplicate, tz-aware last_run, save-after-claim), HA cache TTL,
home_assistant string data, executor unification, disabled-tool hints,
per-turn enablement refresh, Roku remote caching, lamp restore no-op,
Plex launch note, Home Mini candidates + port guard, GitHub token and
Spotify redirect at call time, and VIKI_WORKSPACE routing.
No network: HTTP transports are mocked.
"""
import json
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


# --- fakes ---------------------------------------------------------------
class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeCall:
    def __init__(self, name, arguments):
        self.function = FakeFunc(name, arguments)


class FakeMsg:
    def __init__(self, content="", tool_calls=None, thinking=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.thinking = thinking


class FakeChunk:
    def __init__(self, message):
        self.message = message


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen_messages = []
        self.seen_kwargs = []

    def chat(self, **kw):
        self.calls += 1
        self.seen_messages.append(list(kw.get("messages") or []))
        self.seen_kwargs.append(kw)
        kind, payload = self.script.pop(0)
        if kind == "raise":
            raise payload
        return [FakeChunk(m) for m in payload]


def make_viki(script, tmp):
    store = SessionStore(folder=str(tmp / "session"))
    client = FakeClient(script)
    with patch.object(agent_mod.ollama, "Client", return_value=client):
        v = agent_mod.VIKI(Toolbox(store))
    v.client = client
    return v


def run_loop(viki, prompt, cancel=None):
    events = []
    out = viki.run(prompt, [], [],
                   lambda k, p: events.append((k, p)),
                   lambda cid, n, a: True, think=False,
                   cancel=cancel or threading.Event())
    return out, events


class EnvGuard(unittest.TestCase):
    """Save/restore the env vars tests touch."""
    KEYS = ("VIKI_CLOUD_MODEL", "VIKI_CLOUD_BASE_URL", "VIKI_CLOUD_API_KEY",
            "VIKI_CLOUD_TIMEOUT", "VIKI_GEMINI_MODEL", "VIKI_GEMINI_API_KEY",
            "VIKI_MAX_STEPS", "VIKI_WORKSPACE", "VIKI_ANNOUNCE_PORT",
            "SPOTIFY_REDIRECT_URI", "GITHUB_TOKEN")

    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- _ThinkFilter ----------------------------------------------------------
class TestThinkFilterFixes(unittest.TestCase):
    def test_nested_think_blocks(self):
        f = agent_mod._ThinkFilter()
        self.assertEqual(
            f.feed("a<think>x<think>y</think>z</think>b"), "ab")

    def test_stray_close_tag_stripped(self):
        f = agent_mod._ThinkFilter()
        self.assertEqual(f.feed("a</think>b"), "ab")

    def test_case_insensitive_and_whitespace_tags(self):
        f = agent_mod._ThinkFilter()
        self.assertEqual(f.feed("a< THINK >secret</ Think >b"), "ab")

    def test_flush_returns_held_tail(self):
        f = agent_mod._ThinkFilter()
        self.assertEqual(f.feed("answer <"), "answer ")
        self.assertEqual(f.flush(), "<")
        self.assertEqual(f.feed(" done"), " done")

    def test_flush_drops_unclosed_block(self):
        f = agent_mod._ThinkFilter()
        self.assertEqual(f.feed("Answer <think>never closed"), "Answer ")
        self.assertEqual(f.flush(), "")


# --- cloud config / messages / stream --------------------------------------
class TestCloudFixes(EnvGuard):
    def test_non_finite_timeout_falls_back(self):
        with patch.dict(os.environ, {"VIKI_CLOUD_MODEL": "m",
                                    "VIKI_CLOUD_BASE_URL": "https://x/v1",
                                    "VIKI_CLOUD_API_KEY": "k",
                                    "VIKI_CLOUD_TIMEOUT": "inf"}):
            self.assertEqual(agent_mod._cloud_config()["timeout"], 60.0)

    def test_bare_gemini_gets_google_base(self):
        with patch.dict(os.environ, {"VIKI_GEMINI_API_KEY": "AIza-x",
                                    "VIKI_GEMINI_MODEL": "gemini"}):
            cfg = agent_mod._cloud_config()
        self.assertEqual(cfg["base_url"], agent_mod._GEMINI_OPENAI_BASE_URL)

    def test_leftover_tool_calls_raise(self):
        with self.assertRaises(ValueError):
            agent_mod._cloud_messages([
                {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "a", "arguments": {}}}]},
            ])

    def test_cloud_think_status_emitted(self):
        tmp = Path(tempfile.mkdtemp())
        v = make_viki([], tmp)
        events = []

        class Resp:
            def __init__(self, lines):
                self.lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter(self.lines)

        lines = [b'data: {"choices": [{"delta": {"content": "hi"}}]}',
                 b"data: [DONE]"]
        with patch.dict(os.environ, {"VIKI_GEMINI_API_KEY": "AIza-x",
                                    "VIKI_GEMINI_MODEL": "gemini-2.5-flash"}):
            with patch.object(agent_mod.urllib.request, "urlopen",
                              return_value=Resp(lines)):
                content, thinking, calls = v._stream_cloud(
                    agent_mod._cloud_config(), [{"role": "user", "content": "hi"}],
                    None, True, lambda k, p: events.append((k, p)),
                    threading.Event())
        self.assertEqual(content, "hi")
        self.assertTrue(any(k == "status" and "reasoning" in p
                            for k, p in events))

    def test_indexless_calls_get_sequential_indexes(self):
        tmp = Path(tempfile.mkdtemp())
        v = make_viki([], tmp)

        class Resp:
            def __init__(self, lines):
                self.lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter(self.lines)

        def chunk(tc):
            return ("data: " + json.dumps({"choices": [{"delta": {"tool_calls": [tc]}}]})).encode()
        lines = [
            chunk({"function": {"name": "a", "arguments": "{}"}}),
            chunk({"function": {"name": "b", "arguments": "{}"}}),
            b"data: [DONE]",
        ]
        with patch.dict(os.environ, {"VIKI_GEMINI_API_KEY": "AIza-x",
                                    "VIKI_GEMINI_MODEL": "gemini-2.5-flash"}):
            with patch.object(agent_mod.urllib.request, "urlopen",
                              return_value=Resp(lines)):
                _, _, calls = v._stream_cloud(
                    agent_mod._cloud_config(), [{"role": "user", "content": "hi"}],
                    [{"type": "function", "function": {"name": "a"}}],
                    False, lambda k, p: None, threading.Event())
        self.assertEqual([c.function.name for c in calls], ["a", "b"])

    def test_indexless_continuation_appends_to_current_call(self):
        tmp = Path(tempfile.mkdtemp())
        v = make_viki([], tmp)

        class Resp:
            def __init__(self, lines):
                self.lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter(self.lines)

        def chunk(tc):
            return ("data: " + json.dumps({"choices": [{"delta": {"tool_calls": [tc]}}]})).encode()
        lines = [
            chunk({"function": {"name": "calculate", "arguments": '{"'}}),
            chunk({"function": {"arguments": 'expression": "1+1"'}}),
            chunk({"function": {"arguments": "}"}}),
            b"data: [DONE]",
        ]
        with patch.dict(os.environ, {"VIKI_GEMINI_API_KEY": "AIza-x",
                                    "VIKI_GEMINI_MODEL": "gemini-2.5-flash"}):
            with patch.object(agent_mod.urllib.request, "urlopen",
                              return_value=Resp(lines)):
                _, _, calls = v._stream_cloud(
                    agent_mod._cloud_config(), [{"role": "user", "content": "hi"}],
                    [{"type": "function", "function": {"name": "calculate"}}],
                    False, lambda k, p: None, threading.Event())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.arguments, {"expression": "1+1"})

    def test_repeated_index_with_new_name_is_protocol_error(self):
        tmp = Path(tempfile.mkdtemp())
        v = make_viki([], tmp)

        class Resp:
            def __init__(self, lines):
                self.lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter(self.lines)

        def chunk(tc):
            return ("data: " + json.dumps({"choices": [{"delta": {"tool_calls": [tc]}}]})).encode()
        lines = [
            chunk({"index": 0, "function": {"name": "a", "arguments": "{}"}}),
            chunk({"index": 0, "function": {"name": "b", "arguments": "{}"}}),
            b"data: [DONE]",
        ]
        with patch.dict(os.environ, {"VIKI_GEMINI_API_KEY": "AIza-x",
                                    "VIKI_GEMINI_MODEL": "gemini-2.5-flash"}):
            with patch.object(agent_mod.urllib.request, "urlopen",
                              return_value=Resp(lines)):
                with self.assertRaises(ValueError):
                    v._stream_cloud(
                        agent_mod._cloud_config(),
                        [{"role": "user", "content": "hi"}],
                        [{"type": "function", "function": {"name": "a"}}],
                        False, lambda k, p: None, threading.Event())


# --- agent run fixes ---------------------------------------------------------
class TestAgentRunFixes(EnvGuard):
    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())

    def test_max_steps_clamped(self):
        with patch.dict(os.environ, {"VIKI_MAX_STEPS": "0"}):
            v = make_viki([], self.tmp)
        self.assertEqual(v.max_steps, 1)
        with patch.dict(os.environ, {"VIKI_MAX_STEPS": "garbage"}):
            v = make_viki([], self.tmp)
        self.assertEqual(v.max_steps, 10)

    def test_malformed_tool_call_continues_loop(self):
        v = make_viki([
            ("chunks", [FakeMsg(tool_calls=[FakeCall(None, {})])]),
            ("chunks", [FakeMsg(content="recovered")]),
        ], self.tmp)
        out, events = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["content"], "recovered")
        tool_msgs = [m for m in v.client.seen_messages[-1]
                     if m.get("role") == "tool"]
        self.assertTrue(any("Malformed" in m["content"] for m in tool_msgs))

    def test_string_arguments_parsed(self):
        v = make_viki([
            ("chunks", [FakeMsg(tool_calls=[FakeCall(
                "calculate", '{"expression": "1+1"}')])]),
            ("chunks", [FakeMsg(content="two")]),
        ], self.tmp)
        out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["content"], "two")

    def test_repeat_note_deferred_after_all_tool_messages(self):
        v = make_viki([
            ("chunks", [FakeMsg(tool_calls=[
                FakeCall("calculate", {"expression": "1+1"}),
                FakeCall("calculate", {"expression": "1+1"})])]),
            ("chunks", [FakeMsg(content="done")]),
        ], self.tmp)
        out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        msgs = v.client.seen_messages[-1]
        note_idx = next(i for i, m in enumerate(msgs)
                        if m.get("role") == "user" and "System note" in m.get("content", ""))
        tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
        self.assertEqual(len(tool_idx), 2)
        self.assertTrue(all(i < note_idx for i in tool_idx))

    def test_wrap_up_failure_falls_back_with_tools(self):
        with patch.dict(os.environ, {"VIKI_MAX_STEPS": "1"}):
            v = make_viki([
                ("chunks", [FakeMsg(tool_calls=[
                    FakeCall("current_time", {})])]),
            ], self.tmp)
        out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["content"], agent_mod._EMPTY_ANSWER_FALLBACK)
        self.assertTrue(out["tools"])

    def test_error_return_has_tools_and_no_raw_exception(self):
        v = make_viki([("raise", RuntimeError("secret-boom"))], self.tmp)
        out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "error")
        self.assertIn("tools", out)
        self.assertNotIn("secret-boom", out["content"])

    def test_tools_ok_mirroring(self):
        v = make_viki([
            ("raise", ollama.ResponseError("model 'x' does not support tools")),
            ("chunks", [FakeMsg(content="plain")]),
            ("chunks", [FakeMsg(content="second")]),
        ], self.tmp)
        out, _ = run_loop(v, "hi")
        self.assertEqual(out["type"], "text")
        self.assertFalse(v.tools_ok)
        out2, _ = run_loop(v, "again")
        self.assertEqual(out2["type"], "text")
        self.assertNotIn("tools", v.client.seen_kwargs[-1])

    def test_cancel_during_retry_sleep(self):
        v = make_viki([
            ("raise", ConnectionError("down")),
            ("chunks", [FakeMsg(content="never")]),
        ], self.tmp)
        cancel = MagicMock()
        cancel.wait.return_value = True
        cancel.is_set.return_value = False
        out, _ = run_loop(v, "hi", cancel=cancel)
        self.assertEqual(out["type"], "cancelled")
        cancel.wait.assert_called()


# --- scheduler fixes ---------------------------------------------------------
class TestSchedulerFixes(unittest.TestCase):
    def setUp(self):
        from scheduler import Scheduler
        self.tmp = Path(tempfile.mkdtemp())
        self.s = Scheduler(folder=str(self.tmp))

    def test_bool_and_fractional_every_minutes_rejected(self):
        from scheduler import Scheduler
        with self.assertRaises(ValueError):
            self.s.add("x", "interval", every_minutes=True)
        with self.assertRaises(ValueError):
            self.s.add("x", "interval", every_minutes=2.5)
        job = self.s.add("x", "interval", every_minutes=2.0)
        self.assertEqual(job["every_minutes"], 2)

    def test_unknown_action_rejected_blank_allowed(self):
        with self.assertRaises(ValueError):
            self.s.add("x", "interval", action="bogus-check", every_minutes=5)
        job = self.s.add("x", "interval", action="", every_minutes=5)
        self.assertEqual(job["action"], "")

    def test_reschedule_fired_once_revives(self):
        job = self.s.add("Once", "once", at="2030-01-01 10:00")
        self.s.run_one(job["id"], lambda j: "done")
        fired = [t for t in self.s.list_tasks() if t["id"] == job["id"]][0]
        self.assertFalse(fired["enabled"])
        self.assertTrue(self.s.reschedule(job["id"], "once", at="2030-01-02 10:00"))
        revived = [t for t in self.s.list_tasks() if t["id"] == job["id"]][0]
        self.assertTrue(revived["enabled"])
        self.assertIsNone(revived["last_run"])

    def test_duplicate_of_fired_once_stays_disabled(self):
        job = self.s.add("Once", "once", at="2030-01-01 10:00")
        self.s.run_one(job["id"], lambda j: "done")
        copy = self.s.duplicate(job["id"])
        self.assertFalse(copy["enabled"])
        self.assertIsNone(copy["last_run"])

    def test_tz_aware_last_run_normalised(self):
        from datetime import datetime, timedelta
        job = self.s.add("x", "interval", every_minutes=60)
        job["last_run"] = "2026-10-29T10:00:00+02:00"
        now = datetime(2026, 10, 29, 12, 0)
        # 10:00+02:00 == 08:00 UTC; naive-local comparison must not raise.
        self.assertIsInstance(self.s.is_due(job, now), bool)
        bad = dict(job, last_run="not-a-date")
        self.assertIsNone(self.s._parse_last_run(bad))

    def test_run_due_saves_after_claim(self):
        job = self.s.add("x", "interval", every_minutes=60)
        seen = {}

        def executor(j):
            import json as js
            on_disk = js.loads(self.s.tasks_path.read_text(encoding="utf-8"))
            row = [t for t in on_disk["tasks"] if t["id"] == job["id"]][0]
            seen["claimed_on_disk"] = row["last_run"] is not None
            return "ok"

        self.s.run_due(executor)
        self.assertTrue(seen["claimed_on_disk"])


# --- tools fixes ---------------------------------------------------------------
class TestToolsFixes(unittest.TestCase):
    def setUp(self):
        import tools as tools_mod
        self.mod = tools_mod
        self.orig = {k: v for k, v in tools_mod.HA_ENTITIES_CACHE.items()}
        self.orig["entities"] = list(self.orig["entities"])
        self.orig["full"] = list(self.orig["full"])

    def tearDown(self):
        self.mod.HA_ENTITIES_CACHE.clear()
        self.mod.HA_ENTITIES_CACHE.update(self.orig)

    def test_ha_stale_cache_refreshes(self):
        from datetime import datetime, timedelta
        self.mod.HA_ENTITIES_CACHE.update({
            "updated": (datetime.now() - timedelta(seconds=3600)).isoformat(),
            "entities": [{"entity_id": "light.x", "name": "x", "state": "on"}],
            "full": []})
        with patch.object(self.mod, "_ha_request",
                          return_value=[]) as req:
            self.mod.home_assistant("list")
        req.assert_called()

    def test_ha_fresh_cache_no_network(self):
        from datetime import datetime
        self.mod.HA_ENTITIES_CACHE.update({
            "updated": datetime.now().isoformat(),
            "entities": [{"entity_id": "light.x", "name": "x", "state": "on"}],
            "full": []})
        with patch.object(self.mod, "_ha_request") as req:
            out = self.mod.home_assistant("list")
        req.assert_not_called()
        self.assertIn("light.x", out)

    def test_home_assistant_string_data_parsed(self):
        with patch.object(self.mod, "_ha_request",
                          return_value={"ok": True}) as req:
            out = self.mod.home_assistant(
                "call", entity_id="light.x", domain="light",
                service="turn_on", data='{"brightness_pct": 50}')
        self.assertNotIn("must be a JSON object", out)
        sent = req.call_args[0][1]
        self.assertEqual(sent["brightness_pct"], 50)

    def test_home_assistant_bad_string_data_rejected(self):
        out = self.mod.home_assistant(
            "call", entity_id="light.x", domain="light",
            service="turn_on", data='not json')
        self.assertIn("must be a JSON object", out)

    def test_job_action_blank_and_unknown(self):
        with patch.object(self.mod, "connection_check",
                          return_value="brief") as cc:
            self.assertEqual(self.mod._run_job_action({"action": ""}), "brief")
        self.assertIn("No built-in action",
                      self.mod._run_job_action({"action": "bogus"}))

    def test_disabled_spotify_hint(self):
        store = SessionStore(folder=str(Path(tempfile.mkdtemp()) / "s"))
        tb = Toolbox(store)
        tb.enabled = [n for n in tb.enabled if n != "spotify_play"]
        out = tb.run("spotify_play", {"query": "x"})
        self.assertIn("Spotify is not connected", out)
        self.assertNotIn("Unknown tool", out)

    def test_refresh_enabled_preserves_non_specs_names(self):
        store = SessionStore(folder=str(Path(tempfile.mkdtemp()) / "s"))
        tb = Toolbox(store)
        tb.enabled.append("fake-tool")
        tb.refresh_enabled()
        self.assertIn("fake-tool", tb.enabled)


# --- connector fixes -----------------------------------------------------------
class TestConnectorFixes(EnvGuard):
    def test_roku_none_remote_not_cached(self):
        import connectors.roku as roku_mod
        states = [{"entity_id": "media_player.roku_tv", "state": "idle"}]

        class Resp:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(self.payload).encode()

        with patch.object(roku_mod, "_ha", return_value=("http://ha", "tok")), \
                patch.dict(os.environ, {"HA_URL": "http://ha", "HA_TOKEN": "tok"}):
            with patch.object(roku_mod.urllib.request, "urlopen",
                              return_value=Resp(states)):
                roku_mod._entities_cache = None
                ent, remote = roku_mod._entities()
        self.assertEqual(ent, "media_player.roku_tv")
        self.assertIsNone(remote)
        self.assertIsNone(roku_mod._entities_cache)

    def test_roku_404_drops_cache(self):
        import urllib.error
        import connectors.roku as roku_mod
        with patch.object(roku_mod, "_ha", return_value=("http://ha", "tok")), \
                patch.dict(os.environ, {"HA_URL": "http://ha", "HA_TOKEN": "tok"}):
            with patch.object(roku_mod.urllib.request, "urlopen",
                              side_effect=urllib.error.HTTPError(
                                  "http://ha/api/states/x", 404, "Not Found", {}, None)):
                roku_mod._entities_cache = ("media_player.roku_tv", "remote.roku")
                with self.assertRaises(urllib.error.HTTPError):
                    roku_mod._state("media_player.roku_tv")
        self.assertIsNone(roku_mod._entities_cache)

    def test_lamp_restore_without_snapshot_is_noop(self):
        import connectors.lamp as lamp_mod
        ind = lamp_mod.LampIndicator.__new__(lamp_mod.LampIndicator)
        ind._snapshot = None
        ind.entity = "light.x"
        with patch.object(lamp_mod, "_service") as svc:
            ind.restore()
        svc.assert_not_called()

    def test_plex_launch_failure_surfaced(self):
        import connectors.plex as plex_mod
        with patch("connectors.roku.ha_connected", return_value=True), \
                patch("connectors.roku.roku_launch",
                      return_value="Roku action failed: boom"):
            note = plex_mod._maybe_launch_plex_on_roku("Roku TV")
        self.assertIsNotNone(note)
        self.assertIn("Roku", note)

    def test_home_mini_candidates_and_port_guard(self):
        import connectors.home_mini as mini_mod
        with patch.object(mini_mod, "_ha_states",
                          return_value=[{"entity_id": "media_player.kitchen",
                                         "attributes": {}}]):
            self.assertEqual(mini_mod._mini_candidates(),
                             ["media_player.kitchen"])
        with patch.dict(os.environ, {"VIKI_ANNOUNCE_PORT": "not-a-port"}):
            self.assertEqual(mini_mod._announce_port(), 7871)

    def test_github_token_read_at_call_time(self):
        import connectors.github as gh_mod
        with patch.dict(os.environ, {"GITHUB_TOKEN": "tok-123"}):
            self.assertTrue(gh_mod.token_exists())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_TOKEN", None)
            self.assertFalse(gh_mod.token_exists())

    def test_spotify_redirect_read_at_call_time(self):
        import connectors.spotify as sp_mod
        with patch.dict(os.environ,
                        {"SPOTIFY_REDIRECT_URI": "http://x.test/cb"}):
            self.assertEqual(sp_mod._redirect(), "http://x.test/cb")

    def test_workspace_routing(self):
        import importlib
        import connectors.browser as browser_mod
        import connectors.maps as maps_mod
        tmp = str(Path(tempfile.mkdtemp()))
        with patch.dict(os.environ, {"VIKI_WORKSPACE": tmp}):
            importlib.reload(browser_mod)
            importlib.reload(maps_mod)
            try:
                self.assertEqual(str(browser_mod.WORK_DIR), tmp + "/browser")
                self.assertEqual(str(maps_mod._CACHE_FILE),
                                 tmp + "/geocode_cache.json")
            finally:
                os.environ.pop("VIKI_WORKSPACE", None)
                importlib.reload(browser_mod)
                importlib.reload(maps_mod)


if __name__ == "__main__":
    unittest.main(verbosity=2)
