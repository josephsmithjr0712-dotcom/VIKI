"""Tests for the cloud-model option (v0.3.17).

When VIKI_CLOUD_MODEL, VIKI_CLOUD_BASE_URL and VIKI_CLOUD_API_KEY are all
set, model turns go to an OpenAI-compatible chat-completions endpoint;
any cloud failure falls back to local Ollama and never kills the turn.
The API key travels only in the Authorization header — never in URLs,
errors, or emitted output. No network: the HTTP transport is mocked.
"""

import json
import os
import sys
import threading
import unittest
import urllib.error
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

CLOUD_ENV = {
    "VIKI_CLOUD_MODEL": "cloud-model-x",
    "VIKI_CLOUD_BASE_URL": "https://cloud.example.com/v1",
    "VIKI_CLOUD_API_KEY": "sk-test-secret-12345",
}
CLOUD_KEYS = ("VIKI_CLOUD_MODEL", "VIKI_CLOUD_BASE_URL",
              "VIKI_CLOUD_API_KEY", "VIKI_CLOUD_TIMEOUT")


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


class FakeHTTPResponse:
    """urlopen stand-in: a context manager yielding SSE byte lines."""

    def __init__(self, lines):
        self.lines = [l if isinstance(l, bytes) else l.encode("utf-8")
                      for l in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


def sse(*chunks):
    lines = ["data: " + json.dumps(c) for c in chunks]
    lines.append("data: [DONE]")
    return FakeHTTPResponse(lines)


def text_chunk(content):
    return {"choices": [{"delta": {"content": content}}]}


def tool_chunk(index, name=None, args=None):
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    return {"choices": [{"delta": {"tool_calls": [{"index": index,
                                                  "function": fn}]}}]}


class CloudTestBase(unittest.TestCase):
    def setUp(self):
        self.saved_env = {k: os.environ.get(k) for k in CLOUD_KEYS}
        for k in CLOUD_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def make_viki(self, script):
        with patch.object(agent_mod.ollama, "Client",
                          return_value=FakeClient(script)):
            return agent_mod.VIKI(toolbox=object())

    def stream(self, viki, messages, tools=None):
        events = []
        out = viki._stream(messages, tools, False,
                           lambda k, p: events.append((k, p)),
                           threading.Event())
        return out, events


class TestCloudConfig(unittest.TestCase):
    def test_absent_env_is_none(self):
        for k in CLOUD_KEYS:
            os.environ.pop(k, None)
        self.assertIsNone(agent_mod._cloud_config())

    def test_partial_env_is_none(self):
        with patch.dict(os.environ, {"VIKI_CLOUD_MODEL": "m",
                                    "VIKI_CLOUD_BASE_URL": "https://x/v1"}):
            self.assertIsNone(agent_mod._cloud_config())

    def test_full_env_parsed(self):
        with patch.dict(os.environ, CLOUD_ENV):
            cfg = agent_mod._cloud_config()
        self.assertEqual(cfg["model"], "cloud-model-x")
        self.assertEqual(cfg["base_url"], "https://cloud.example.com/v1")
        self.assertEqual(cfg["timeout"], 60.0)

    def test_trailing_slash_and_bad_timeout(self):
        env = dict(CLOUD_ENV, VIKI_CLOUD_BASE_URL="https://x/v1/",
                   VIKI_CLOUD_TIMEOUT="nope")
        with patch.dict(os.environ, env):
            cfg = agent_mod._cloud_config()
        self.assertEqual(cfg["base_url"], "https://x/v1")
        self.assertEqual(cfg["timeout"], 60.0)


class TestCloudMessages(unittest.TestCase):
    def test_tool_calls_paired_in_order(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "a", "arguments": {"x": 1}}},
                {"function": {"name": "b", "arguments": {}}}]},
            {"role": "tool", "tool_name": "a", "content": "ra"},
            {"role": "tool", "tool_name": "b", "content": "rb"},
        ]
        out = agent_mod._cloud_messages(messages)
        self.assertEqual(out[1]["tool_calls"][0]["id"], "call_0")
        self.assertEqual(out[1]["tool_calls"][1]["id"], "call_1")
        self.assertEqual(out[2], {"role": "tool", "tool_call_id": "call_0",
                                  "content": "ra"})
        self.assertEqual(out[3]["tool_call_id"], "call_1")
        self.assertNotIn("tool_name", json.dumps(out))

    def test_unpaired_tool_message_raises(self):
        with self.assertRaises(ValueError):
            agent_mod._cloud_messages(
                [{"role": "tool", "tool_name": "a", "content": "x"}])


class TestCloudStream(CloudTestBase):
    def run_cloud(self, viki, response, messages=None, tools=None):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["request"] = req
            seen["timeout"] = timeout
            if isinstance(response, Exception):
                raise response
            return response

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out, events = self.stream(
                viki, messages or [{"role": "user", "content": "hi"}], tools)
        return out, events, seen

    def test_cloud_used_when_env_set(self):
        viki = self.make_viki([])
        with patch.dict(os.environ, CLOUD_ENV):
            out, events, seen = self.run_cloud(
                viki, sse(text_chunk("hel"), text_chunk("lo")))
        self.assertEqual(out, ("hello", "", []))
        req = seen["request"]
        self.assertEqual(req.full_url,
                         "https://cloud.example.com/v1/chat/completions")
        self.assertEqual(req.get_header("Authorization"),
                         "Bearer sk-test-secret-12345")
        self.assertEqual(seen["timeout"], 60.0)
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "cloud-model-x")
        self.assertTrue(payload["stream"])
        self.assertNotIn("tools", payload)
        self.assertEqual([p for k, p in events if k == "token"],
                         ["hel", "lo"])
        self.assertEqual(viki.client.calls, 0)

    def test_cloud_tools_passed_and_assembled(self):
        viki = self.make_viki([])
        tools = [{"type": "function",
                  "function": {"name": "remember", "parameters": {}}}]
        resp = sse(tool_chunk(0, name="remember", args='{"fact":'),
                   tool_chunk(0, args=' "likes tea"}'))
        with patch.dict(os.environ, CLOUD_ENV):
            out, _, seen = self.run_cloud(viki, resp, tools=tools)
        payload = json.loads(seen["request"].data.decode("utf-8"))
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(len(out[2]), 1)
        self.assertEqual(out[2][0].function.name, "remember")
        self.assertEqual(out[2][0].function.arguments, {"fact": "likes tea"})

    def test_local_used_when_env_absent(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        with patch("urllib.request.urlopen") as urlopen:
            out, _ = self.stream(viki, [{"role": "user", "content": "hi"}])
        urlopen.assert_not_called()
        self.assertEqual(out, ("local reply", "", []))
        self.assertEqual(viki.client.calls, 1)

    def test_partial_env_stays_local(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        with patch.dict(os.environ, {"VIKI_CLOUD_MODEL": "m",
                                    "VIKI_CLOUD_BASE_URL": "https://x/v1"}), \
                patch("urllib.request.urlopen") as urlopen:
            out, _ = self.stream(viki, [{"role": "user", "content": "hi"}])
        urlopen.assert_not_called()
        self.assertEqual(out[0], "local reply")

    def test_images_turn_skips_cloud(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        messages = [{"role": "user", "content": "see this", "images": [b"img"]}]
        with patch.dict(os.environ, CLOUD_ENV), \
                patch("urllib.request.urlopen") as urlopen:
            out, _ = self.stream(viki, messages)
        urlopen.assert_not_called()
        self.assertEqual(out[0], "local reply")

    def test_fallback_on_network_error(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        with patch.dict(os.environ, CLOUD_ENV):
            out, events, _ = self.run_cloud(
                viki, urllib.error.URLError("boom"))
        self.assertEqual(out, ("local reply", "", []))
        self.assertIn("falling back", " ".join(p for k, p in events
                                               if k == "status"))
        self.assertEqual(viki.client.calls, 1)

    def test_fallback_on_non_2xx(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        err = urllib.error.HTTPError("https://cloud.example.com/v1", 500,
                                     "Server Error", {}, None)
        with patch.dict(os.environ, CLOUD_ENV):
            out, events, _ = self.run_cloud(viki, err)
        self.assertEqual(out[0], "local reply")
        self.assertTrue(any("falling back" in p for k, p in events
                            if k == "status"))

    def test_fallback_on_bad_payload(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        with patch.dict(os.environ, CLOUD_ENV):
            out, _, _ = self.run_cloud(
                viki, FakeHTTPResponse(["data: not json\n", "data: [DONE]\n"]))
        self.assertEqual(out[0], "local reply")

    def test_fallback_on_unparsable_tool_args(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        resp = sse(tool_chunk(0, name="remember", args="{bad json"))
        with patch.dict(os.environ, CLOUD_ENV):
            out, _, _ = self.run_cloud(viki, resp)
        self.assertEqual(out[0], "local reply")

    def test_key_never_in_output(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        key = CLOUD_ENV["VIKI_CLOUD_API_KEY"]
        with patch.dict(os.environ, CLOUD_ENV):
            _, events, seen = self.run_cloud(
                viki, urllib.error.URLError("boom"))
        req = seen["request"]
        self.assertNotIn(key, req.full_url)
        self.assertNotIn(key, " ".join(str(p) for _, p in events))
        self.assertNotIn(key, " ".join(str(v) for v in seen.values()
                                        if isinstance(v, str)))

    def test_key_never_in_success_output(self):
        viki = self.make_viki([])
        key = CLOUD_ENV["VIKI_CLOUD_API_KEY"]
        with patch.dict(os.environ, CLOUD_ENV):
            out, events, seen = self.run_cloud(
                viki, sse(text_chunk("hello")))
        self.assertNotIn(key, " ".join(str(p) for _, p in events))
        self.assertNotIn(key, out[0])
        payload = json.loads(seen["request"].data.decode("utf-8"))
        self.assertNotIn(key, json.dumps(payload))

    def test_cancel_propagates_not_fallback(self):
        viki = self.make_viki([("ok", [FakeMsg("local reply")])])
        cancel = threading.Event()
        cancel.set()
        events = []
        with patch.dict(os.environ, CLOUD_ENV), \
                patch("urllib.request.urlopen",
                      return_value=sse(text_chunk("hello"))):
            with self.assertRaises(agent_mod.Cancelled):
                viki._stream([{"role": "user", "content": "hi"}], None, False,
                             lambda k, p: events.append((k, p)), cancel)
        self.assertEqual(viki.client.calls, 0)


if __name__ == "__main__":
    unittest.main()
