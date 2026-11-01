"""Thinking-mode repair (2026-10-31): reasoning must never reach speech.

v0.3.38 only stripped <think>. Models and cloud providers also emit
<thinking>, <reasoning>, <thought> and <internal> blocks, and a model
can leak reasoning into Ollama's separate thinking channel even with
think=False. With think=False, none of that may be emitted as tokens,
shown as thinking, or returned as final content — the clock speaks
tokens and the Done content, so both paths are pinned here. The online
(Gemini) path must also ask for zero thinking budget and must ignore
reasoning_content deltas.
"""
import json
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


class FakeMsg:
    def __init__(self, content="", thinking=None):
        self.content = content
        self.tool_calls = []
        self.thinking = thinking


class FakeChunk:
    def __init__(self, message):
        self.message = message


class ChunkClient:
    def __init__(self, messages):
        self.messages = list(messages)
        self.kwargs = []

    def chat(self, **kw):
        self.kwargs.append(kw)
        return [FakeChunk(m) for m in self.messages]


def make_viki(client, tmp):
    store = SessionStore(folder=tmp / "session")
    with patch.object(agent_mod.ollama, "Client", return_value=client):
        return agent_mod.VIKI(Toolbox(store))


class TestReasoningTagVariants(unittest.TestCase):
    def test_all_tag_spellings_stripped(self):
        for tag in ("think", "thinking", "reasoning", "thought", "internal"):
            filt = agent_mod._ThinkFilter()
            out = filt.feed(f"Hi <{tag}>secret deliberation</{tag}> boss") + filt.flush()
            self.assertEqual(out, "Hi  boss", tag)
            self.assertNotIn("secret", out)

    def test_mixed_case_and_split_tags(self):
        filt = agent_mod._ThinkFilter()
        out = filt.feed("Hi <Rea") + filt.feed("soning>secret</rea") + filt.feed("soning> boss")
        self.assertEqual(out, "Hi  boss")
        self.assertNotIn("secret", out)

    def test_unclosed_reasoning_block_dropped(self):
        filt = agent_mod._ThinkFilter()
        out = filt.feed("Answer <reasoning>never closed") + filt.flush()
        self.assertEqual(out, "Answer ")

    def test_final_scrub_covers_done_path(self):
        self.assertEqual(
            agent_mod._strip_reasoning_blocks("<thinking>plan</thinking>Hey boss!"),
            "Hey boss!")


class TestThinkChannelSuppressed(unittest.TestCase):
    def test_think_false_never_emits_thinking_event(self):
        client = ChunkClient([FakeMsg(content="Hey boss!", thinking="leaked reasoning")])
        v = make_viki(client, Path(tempfile.mkdtemp()))
        events = []
        out = v.run("hello", [], [], lambda k, p: events.append((k, p)),
                    lambda cid, n, a: True, think=False,
                    cancel=threading.Event())
        self.assertEqual(out["content"], "Hey boss!")
        self.assertEqual([p for k, p in events if k == "thinking"], [])
        self.assertNotIn("leaked", "".join(str(p) for _, p in events))

    def test_think_true_still_surfaces_thinking(self):
        client = ChunkClient([FakeMsg(content="Hey boss!", thinking="shown reasoning")])
        v = make_viki(client, Path(tempfile.mkdtemp()))
        events = []
        v.run("hello", [], [], lambda k, p: events.append((k, p)),
              lambda cid, n, a: True, think=True,
              cancel=threading.Event())
        self.assertIn("shown reasoning", [p for k, p in events if k == "thinking"])

    def test_reasoning_in_content_never_reaches_tokens_or_final(self):
        client = ChunkClient([FakeMsg(
            content="<reasoning>debate the greeting</reasoning>Hey boss!")])
        v = make_viki(client, Path(tempfile.mkdtemp()))
        events = []
        out = v.run("hello", [], [], lambda k, p: events.append((k, p)),
                    lambda cid, n, a: True, think=False,
                    cancel=threading.Event())
        self.assertEqual(out["content"], "Hey boss!")
        tokens = "".join(p for k, p in events if k == "token")
        self.assertEqual(tokens, "Hey boss!")
        self.assertNotIn("debate", tokens)


class FakeHTTPResponse:
    def __init__(self, lines):
        self.lines = [l if isinstance(l, bytes) else l.encode("utf-8") for l in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


class TestOnlineGeminiNoThinking(unittest.TestCase):
    def test_gemini_think_false_sends_zero_budget_and_ignores_reasoning(self):
        cfg = {"model": "gemini-2.5-flash",
               "base_url": agent_mod._GEMINI_OPENAI_BASE_URL,
               "api_key": "AIza-test-key", "timeout": 60.0}
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            lines = [
                "data: " + json.dumps({"choices": [{"delta": {
                    "reasoning_content": "secret online deliberation",
                    "content": "<reasoning>also secret</reasoning>Hey boss!"}}]}),
                "data: [DONE]",
            ]
            return FakeHTTPResponse(lines)

        store = SessionStore(folder=Path(tempfile.mkdtemp()) / "session")
        with patch.object(agent_mod.ollama, "Client", return_value=ChunkClient([])):
            v = agent_mod.VIKI(Toolbox(store))
        events = []
        with patch.object(agent_mod.urllib.request, "urlopen",
                          side_effect=fake_urlopen):
            content, thinking, calls = v._stream_cloud(
                cfg, [{"role": "user", "content": "hello"}], None, False,
                lambda k, p: events.append((k, p)), threading.Event())
        self.assertEqual(captured["payload"]["thinking_config"],
                         {"thinking_budget": 0})
        self.assertEqual(content, "Hey boss!")
        self.assertEqual(thinking, "")
        self.assertNotIn("secret", "".join(str(p) for _, p in events))


if __name__ == "__main__":
    unittest.main()
