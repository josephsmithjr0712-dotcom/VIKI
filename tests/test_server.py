"""Web server endpoint contract tests (v0.2.17).

Verifies the Flask routes in server.py against what the web UI actually calls:
auth gating, login (incl. rate limiting), and every /api/* endpoint the frontend
fetches. Everything external (Ollama, HA refresh, Edge TTS) is stubbed; the test
runs fully offline using Flask's test client.
"""
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- hermetic environment for the server import only; restored right after ---
# Module-level env/patching must not leak into other test files (they share a
# process under unittest discovery), so everything here is snapshotted and
# restored once server.py is imported.
_TMP = tempfile.mkdtemp(prefix="viki-server-test-")
_ENV_KEYS = ("VIKI_SESSION_DIR", "VIKI_SECRET_KEY", "VIKI_WEB_PASSWORD",
             "VIKI_AUTO_MEMORY", "VIKI_RETRIES")
_old_env = {k: os.environ.get(k) for k in _ENV_KEYS}
os.environ.update({"VIKI_SESSION_DIR": _TMP,
                   "VIKI_SECRET_KEY": "test-secret-key",
                   "VIKI_WEB_PASSWORD": "test-password",
                   "VIKI_AUTO_MEMORY": "off",
                   "VIKI_RETRIES": "0"})

# --- stub ollama (pattern used by other test files) ---
try:
    import ollama  # noqa: F401
    _ollama_stubbed = False
except ImportError:
    ollama = ModuleType("ollama")

    class ResponseError(Exception):
        pass

    ollama.ResponseError = ResponseError
    ollama.Client = None
    sys.modules["ollama"] = ollama
    _ollama_stubbed = True

# --- dotenv: other test files install a stub with only load_dotenv; server.py
# needs set_key too, so install a complete stub unconditionally for the import.
_dotenv_had = "dotenv" in sys.modules
_dotenv_old = sys.modules.get("dotenv")
_dotenv_stub = ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **k: False  # noqa: E731
_dotenv_stub.set_key = lambda *a, **k: False  # noqa: E731
sys.modules["dotenv"] = _dotenv_stub

# --- stub the agent + HA entity refresh BEFORE server.py imports them ---
# If anything in this chain is unavailable (flask, ddgs, ...), the import is
# skipped and every test below is skipped too — globals are restored either way
# so the rest of the suite is unaffected.
import agent as agent_mod

_real_VIKI = agent_mod.VIKI


class FakeVIKI:
    """Stand-in for the real Ollama-backed agent: instant, deterministic."""

    def __init__(self, toolbox):
        self.toolbox = toolbox
        self.model = "test-model"

    def warmup(self):
        pass  # no Ollama in tests

    def run(self, prompt, history, search, emit, approve, think, cancel, images=None):
        emit("status", "thinking")
        return {"type": "text", "content": f"echo: {prompt}", "tools": []}

    def extract_facts(self, *a, **k):
        return []


_SERVER_IMPORT_ERROR = None
server_mod = None
try:
    agent_mod.VIKI = FakeVIKI
    import tools as tools_mod  # noqa: E402

    _real_refresh = tools_mod.refresh_ha_entities
    tools_mod.refresh_ha_entities = lambda: None  # noqa: E731
    import server as _srv  # noqa: E402

    server_mod = _srv
except Exception as e:  # flask/ddgs/etc. missing: skip the server tests
    _SERVER_IMPORT_ERROR = e
finally:
    # restore globals no matter what — never poison the rest of the suite
    agent_mod.VIKI = _real_VIKI
    try:
        tools_mod.refresh_ha_entities = _real_refresh  # noqa: F821
    except NameError:
        pass
    for k, v in _old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if _ollama_stubbed:
        del sys.modules["ollama"]
    if _dotenv_had:
        sys.modules["dotenv"] = _dotenv_old
    else:
        sys.modules.pop("dotenv", None)


class _FakeTTS:
    """Context manager: make `from connectors.edge_tts import speak` succeed
    (returns fake MP3 bytes) only for the duration of a test."""

    def __enter__(self):
        self._had = "connectors.edge_tts" in sys.modules
        self._old = sys.modules.get("connectors.edge_tts")
        mod = ModuleType("connectors.edge_tts")
        mod.speak = lambda text: b"ID3-FAKE-MP3"  # noqa: E731
        sys.modules["connectors.edge_tts"] = mod
        return self

    def __exit__(self, *exc):
        if self._had:
            sys.modules["connectors.edge_tts"] = self._old
        else:
            sys.modules.pop("connectors.edge_tts", None)
        return False


@unittest.skipUnless(server_mod is not None,
                     f"server.py not importable here: {_SERVER_IMPORT_ERROR}")
class ServerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = server_mod.app
        cls.app.testing = True

    def setUp(self):
        server_mod._login_failures.clear()
        self.guest = self.app.test_client()  # no session cookie

    def _login(self, password="test-password"):
        client = self.app.test_client()
        r = client.post("/login", json={"password": password})
        return client, r

    # --- auth gating -----------------------------------------------------
    def test_api_requires_login(self):
        r = self.guest.get("/api/state")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.get_json()["error"], "login required")

    def test_index_serves_login_page_when_unauthed(self):
        r = self.guest.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("password", r.get_data(as_text=True).lower())

    def test_login_wrong_password(self):
        _, r = self._login("nope")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.get_json()["error"], "wrong password")

    def test_login_rate_limit(self):
        client = self.app.test_client()
        for _ in range(5):
            r = client.post("/login", json={"password": "nope"})
            self.assertEqual(r.status_code, 401)
        r = client.post("/login", json={"password": "nope"})
        self.assertEqual(r.status_code, 429)
        self.assertIn("try again", r.get_json()["error"])

    def test_login_success_and_state(self):
        client, r = self._login()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])
        r = client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        for key in ("model", "tools", "memory", "files", "warning"):
            self.assertIn(key, body)
        self.assertEqual(body["model"], "test-model")

    # --- chat ------------------------------------------------------------
    def test_chat_empty_message(self):
        client, _ = self._login()
        r = client.post("/api/chat", json={"session": "t1", "message": "   "})
        self.assertEqual(r.status_code, 400)

    def test_chat_streams_done_event(self):
        client, _ = self._login()
        r = client.post("/api/chat", json={"session": "t1", "message": "hello"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.content_type)
        text = r.get_data(as_text=True)
        self.assertIn('"kind": "done"', text)
        self.assertIn("echo: hello", text)

    # --- approvals / stop / reset ----------------------------------------
    def test_approve_unknown_id_still_ok(self):
        client, _ = self._login()
        r = client.post("/api/approve", json={"id": "nope", "ok": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])

    def test_approve_resolves_pending(self):
        client, _ = self._login()
        ev = threading.Event()
        server_mod.approvals["cid-1"] = {"event": ev, "ok": False}
        r = client.post("/api/approve", json={"id": "cid-1", "ok": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(ev.is_set())

    def test_stop_unknown_session(self):
        client, _ = self._login()
        r = client.post("/api/stop", json={"session": "nope"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])

    def test_reset_unknown_session(self):
        client, _ = self._login()
        r = client.post("/api/reset", json={"session": "nope"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])

    # --- sessions ---------------------------------------------------------
    def test_sessions_list(self):
        client, _ = self._login()
        r = client.get("/api/sessions")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.get_json()["sessions"], list)

    def test_session_load_404(self):
        client, _ = self._login()
        r = client.get("/api/sessions/does-not-exist")
        self.assertEqual(r.status_code, 404)

    # --- memory -----------------------------------------------------------
    def test_memory_crud(self):
        client, _ = self._login()
        r = client.get("/api/memory")
        self.assertEqual(r.status_code, 200)
        before = r.get_json()["memory"]
        client.post("/api/memory", json={"fact": "boss prefers tea over coffee"})
        r = client.get("/api/memory")
        after = r.get_json()["memory"]
        self.assertTrue(any("tea" in m for m in after))
        # store.remove() is 1-based (matches the web UI, which sends i+1)
        client.delete("/api/memory", json={"index": 1})
        r = client.get("/api/memory")
        self.assertEqual(r.get_json()["memory"], before)

    # --- tts --------------------------------------------------------------
    def test_tts_empty_text(self):
        client, _ = self._login()
        r = client.post("/api/tts", json={"text": "  "})
        self.assertEqual(r.status_code, 400)

    def test_tts_returns_audio(self):
        client, _ = self._login()
        with _FakeTTS():
            r = client.post("/api/tts", json={"text": "hello boss"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("audio/mpeg", r.content_type)
        self.assertEqual(r.get_data(), b"ID3-FAKE-MP3")


if __name__ == "__main__":
    unittest.main()
