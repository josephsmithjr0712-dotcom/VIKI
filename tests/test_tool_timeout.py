"""Tool execution timeout tests (v0.2.19).

A hung network/browser tool must not stall the agent loop forever:
Toolbox.run caps each call at VIKI_TOOL_TIMEOUT seconds (default 90) and
returns a graceful, actionable message instead of hanging.
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import ollama  # noqa: F401
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

from memory import SessionStore
from tools import Toolbox


def _toolbox():
    return Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))


class TestToolTimeout(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("VIKI_TOOL_TIMEOUT")
        self.tb = _toolbox()
        self.tb.funcs["fast"] = lambda: "quick result"
        self.tb.funcs["slow"] = lambda: (time.sleep(30), "never")[1]
        self.tb.funcs["boom"] = lambda: 1 / 0
        self.tb.enabled += ["fast", "slow", "boom"]

    def tearDown(self):
        if self._old is None:
            os.environ.pop("VIKI_TOOL_TIMEOUT", None)
        else:
            os.environ["VIKI_TOOL_TIMEOUT"] = self._old

    def test_fast_tool_returns_result(self):
        self.assertEqual(self.tb.run("fast", {}), "quick result")

    def test_exception_still_reported(self):
        self.assertIn("failed", self.tb.run("boom", {}))

    def test_bad_args_still_reported(self):
        self.assertIn("Bad arguments", self.tb.run("fast", {"nope": 1}))

    def test_unknown_tool(self):
        self.assertEqual(self.tb.run("nope", {}), "Unknown tool: nope")

    def test_disabled_tool(self):
        self.tb.enabled = [n for n in self.tb.enabled if n != "fast"]
        self.assertEqual(self.tb.run("fast", {}), "Unknown tool: fast")

    def test_slow_tool_times_out_gracefully(self):
        os.environ["VIKI_TOOL_TIMEOUT"] = "0.3"
        start = time.monotonic()
        out = self.tb.run("slow", {})
        elapsed = time.monotonic() - start
        self.assertIn("timed out", out)
        self.assertIn("slow", out)
        self.assertLess(elapsed, 10, "run() must not block on a hung tool")

    def test_bad_timeout_env_falls_back_to_default(self):
        os.environ["VIKI_TOOL_TIMEOUT"] = "not-a-number"
        self.assertEqual(self.tb.run("fast", {}), "quick result")


if __name__ == "__main__":
    unittest.main()
