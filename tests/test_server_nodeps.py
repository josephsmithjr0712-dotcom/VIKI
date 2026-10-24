"""Regression test (v0.2.22): server.py must import without python-dotenv.

v0.2.21 added graceful degradation for missing ollama/dotenv in agent.py,
but server.py kept an unguarded top-level ``from dotenv import ...``, so the
whole web server crashed at import time in a minimal environment. This test
spawns a subprocess with every dotenv import blocked and asserts server.py
loads and builds its Flask app.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class ServerNoDotenvTest(unittest.TestCase):
    def test_server_imports_without_dotenv(self):
        tmp = tempfile.mkdtemp(prefix="viki-nodeps-test-")
        child = (
            "import sys\n"
            "from types import ModuleType\n"
            "class _Blocker:\n"
            "    def find_module(self, name, path=None):\n"
            "        return self if name == 'dotenv' or name.startswith('dotenv.') else None\n"
            "    def load_module(self, name):\n"
            "        raise ImportError('dotenv blocked for test')\n"
            "sys.meta_path.insert(0, _Blocker())\n"
            "assert 'dotenv' not in sys.modules\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "import agent\n"
            "class FakeVIKI:\n"
            "    def __init__(self, toolbox): self.toolbox = toolbox; self.model = 'test-model'\n"
            "    def warmup(self): pass\n"
            "agent.VIKI = FakeVIKI\n"
            "import server\n"
            "assert server.app is not None\n"
            "assert server.SECRET and server.WEB_PASSWORD\n"
            "print('SERVER_IMPORT_OK')\n"
        )
        env = dict(os.environ)
        env.update({"VIKI_SESSION_DIR": tmp,
                    "VIKI_SECRET_KEY": "test-secret-key",
                    "VIKI_WEB_PASSWORD": "test-password",
                    "VIKI_AUTO_MEMORY": "off",
                    "VIKI_RETRIES": "0"})
        r = subprocess.run([sys.executable, "-c", child],
                           capture_output=True, text=True, env=env, timeout=60)
        self.assertIn("SERVER_IMPORT_OK", r.stdout,
                      f"server.py failed to import without dotenv:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
