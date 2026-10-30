"""Gemini cloud shortcut (v0.3.39).

A gemini-* model routes to Google's OpenAI-compatible endpoint without
an explicit base URL, keyed by VIKI_GEMINI_API_KEY (or VIKI_CLOUD_API_KEY).
An explicit base URL always wins, and a Gemini model with no key stays
local — the cloud path must never fire without credentials. No network:
the HTTP transport is mocked. The key is asserted only via the
Authorization header, never in the URL.
"""
import json
import os
import sys
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

GEMINI_KEYS = ("VIKI_CLOUD_MODEL", "VIKI_CLOUD_BASE_URL",
               "VIKI_CLOUD_API_KEY", "VIKI_CLOUD_TIMEOUT",
               "VIKI_GEMINI_MODEL", "VIKI_GEMINI_API_KEY")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


class GeminiTestBase(unittest.TestCase):
    def setUp(self):
        self.saved_env = {k: os.environ.get(k) for k in GEMINI_KEYS}
        for k in GEMINI_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class FakeMsg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChunk:
    def __init__(self, message):
        self.message = message


class FakeClient:
    def chat(self, **kw):
        return [FakeChunk(FakeMsg(content="local"))]


class FakeHTTPResponse:
    def __init__(self, lines):
        self.lines = [l if isinstance(l, bytes) else l.encode("utf-8")
                      for l in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


def sse_text(content):
    return FakeHTTPResponse([
        "data: " + json.dumps({"choices": [{"delta": {"content": content}}]}),
        "data: [DONE]",
    ])


class TestGeminiConfig(GeminiTestBase):
    def test_gemini_key_only_routes_to_google(self):
        with patch.dict(os.environ, {
                "VIKI_GEMINI_API_KEY": "AIza-test-key",
                "VIKI_GEMINI_MODEL": "gemini-2.5-flash"}):
            cfg = agent_mod._cloud_config()
        self.assertEqual(cfg["model"], "gemini-2.5-flash")
        self.assertEqual(cfg["base_url"], GEMINI_BASE)
        self.assertEqual(cfg["api_key"], "AIza-test-key")
        self.assertEqual(cfg["timeout"], 60.0)

    def test_cloud_model_name_with_gemini_key(self):
        with patch.dict(os.environ, {
                "VIKI_GEMINI_API_KEY": "AIza-test-key",
                "VIKI_CLOUD_MODEL": "gemini-2.5-pro"}):
            cfg = agent_mod._cloud_config()
        self.assertEqual(cfg["model"], "gemini-2.5-pro")
        self.assertEqual(cfg["base_url"], GEMINI_BASE)

    def test_cloud_key_also_works_for_gemini(self):
        with patch.dict(os.environ, {
                "VIKI_CLOUD_API_KEY": "sk-test-key",
                "VIKI_CLOUD_MODEL": "gemini-2.5-flash"}):
            cfg = agent_mod._cloud_config()
        self.assertEqual(cfg["base_url"], GEMINI_BASE)
        self.assertEqual(cfg["api_key"], "sk-test-key")

    def test_explicit_base_url_wins(self):
        with patch.dict(os.environ, {
                "VIKI_GEMINI_API_KEY": "AIza-test-key",
                "VIKI_GEMINI_MODEL": "gemini-2.5-flash",
                "VIKI_CLOUD_BASE_URL": "https://proxy.example.com/v1"}):
            cfg = agent_mod._cloud_config()
        self.assertEqual(cfg["base_url"], "https://proxy.example.com/v1")

    def test_gemini_model_without_key_stays_local(self):
        with patch.dict(os.environ, {
                "VIKI_GEMINI_MODEL": "gemini-2.5-flash"}):
            self.assertIsNone(agent_mod._cloud_config())

    def test_non_gemini_model_without_base_stays_local(self):
        with patch.dict(os.environ, {
                "VIKI_CLOUD_API_KEY": "sk-test-key",
                "VIKI_CLOUD_MODEL": "some-other-model"}):
            self.assertIsNone(agent_mod._cloud_config())


class TestGeminiStream(GeminiTestBase):
    def test_posts_to_google_chat_completions(self):
        with patch.dict(os.environ, {
                "VIKI_GEMINI_API_KEY": "AIza-test-key",
                "VIKI_GEMINI_MODEL": "gemini-2.5-flash"}):
            with patch.object(agent_mod.ollama, "Client",
                              return_value=FakeClient()):
                viki = agent_mod.VIKI(toolbox=object())
            seen = {}

            def fake_urlopen(req, timeout=None):
                seen["url"] = req.full_url
                seen["auth"] = req.get_header("Authorization")
                return sse_text("cloud hello")

            with patch.object(agent_mod.urllib.request, "urlopen",
                              side_effect=fake_urlopen):
                content, thinking, calls = viki._stream(
                    [{"role": "user", "content": "hi"}], None, False,
                    lambda k, p: None, threading.Event())
        self.assertEqual(seen["url"], GEMINI_BASE + "/chat/completions")
        self.assertEqual(seen["auth"], "Bearer AIza-test-key")
        self.assertNotIn("AIza-test-key", seen["url"])
        self.assertEqual(content, "cloud hello")


if __name__ == "__main__":
    unittest.main(verbosity=2)
