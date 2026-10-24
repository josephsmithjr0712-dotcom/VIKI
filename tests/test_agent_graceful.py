"""Tests for agent.py graceful degradation when optional deps are missing.

agent.py must stay importable without the `ollama` / `python-dotenv`
packages (CI sandboxes, minimal installs); constructing VIKI then raises a
clear error instead of crashing at import time.
"""
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_BLOCKED = {"ollama", "dotenv"}


class _ImportBlocker:
    """Meta-path finder that makes chosen top-level modules unimportable."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            raise ImportError("blocked for test: " + name)
        return None


class GracefulImportTests(unittest.TestCase):
    def setUp(self):
        # Save and clear any existing entries so agent.py is imported fresh
        # with ollama/dotenv genuinely missing; restore everything after so
        # the rest of the suite (which stubs ollama) is unaffected.
        self.saved = {m: sys.modules.get(m) for m in ("ollama", "dotenv", "agent")}
        for m in self.saved:
            sys.modules.pop(m, None)
        self.blocker = _ImportBlocker()
        sys.meta_path.insert(0, self.blocker)

    def tearDown(self):
        sys.meta_path.remove(self.blocker)
        for m, mod in self.saved.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod

    def test_import_agent_without_optional_deps(self):
        agent = importlib.import_module("agent")
        self.assertIsNone(agent.ollama)
        self.assertTrue(callable(agent.load_dotenv))
        # load_dotenv fallback is a harmless no-op
        self.assertFalse(agent.load_dotenv())
        # transient-error tuple still builds (getattr default covers None)
        self.assertIsInstance(agent._TRANSIENT_ERRORS, tuple)

    def test_viki_construction_raises_clear_error(self):
        agent = importlib.import_module("agent")
        with self.assertRaisesRegex(RuntimeError, "ollama"):
            agent.VIKI(toolbox=object())


if __name__ == "__main__":
    unittest.main()
