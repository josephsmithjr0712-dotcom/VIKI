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

    def test_chat_stores_map_payloads_for_history(self):
        # v0.2.33: a map event during a turn must land on the stored
        # assistant message so the History tab can re-render the card.
        client, _ = self._login()
        payload = {"origin": {"name": "Millersburg", "lat": 40.55, "lon": -81.92},
                   "destination": {"name": "Killbuck", "lat": 40.49, "lon": -81.98},
                   "distance_m": 16093, "duration_s": 1500,
                   "geometry": [[40.55, -81.92], [40.49, -81.98]],
                   "links": {"google": "https://example.invalid/g",
                             "osm": "https://example.invalid/o"},
                   "traffic": []}

        class MapVIKI:
            model = "test-model"

            def __init__(self, toolbox):
                pass

            def warmup(self):
                pass

            def run(self, prompt, history, search, emit, approve, think, cancel, images=None):
                emit("map", payload)
                return {"type": "text", "content": "here is the route", "tools": []}

            def extract_facts(self, *a, **k):
                return []

        old = server_mod.viki
        try:
            server_mod.viki = MapVIKI(None)
            r = client.post("/api/chat", json={"session": "tmap", "message": "directions please"})
            self.assertEqual(r.status_code, 200)
            text = r.get_data(as_text=True)
            self.assertIn('"kind": "map"', text)
            msgs = server_mod.sessions["tmap"]["messages"]
            self.assertEqual(msgs[-1]["role"], "assistant")
            self.assertEqual(msgs[-1]["maps"], [payload])
            # a turn with no map event stores no "maps" key at all
            server_mod.viki = old
            r = client.post("/api/chat", json={"session": "tmap", "message": "plain follow-up"})
            self.assertEqual(r.status_code, 200)
            r.get_data(as_text=True)
            self.assertNotIn("maps", server_mod.sessions["tmap"]["messages"][-1])
        finally:
            server_mod.viki = old

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

    def test_malformed_json_body_is_handled(self):
        """Endpoints must not 500 on a non-JSON body; they return a graceful JSON response."""
        client, _ = self._login()
        bad = "{not valid json"
        for path in ("/api/chat", "/api/approve", "/api/stop", "/api/reset"):
            r = client.post(path, data=bad, content_type="application/json")
            self.assertNotEqual(r.status_code, 500)
            body = r.get_json()
            self.assertIsNotNone(body, f"{path} should return JSON on malformed input")

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

    # --- scheduled connection-check brief announcements (v0.3.7) --------
    def _run_job_with(self, announce_result):
        """Run a connection-check job with stubbed brief save + announce.

        Returns (result string, saved briefs, announced briefs)."""
        import smart_announce as _sa

        saved, announced = [], []

        def _fake_save(text):
            saved.append(text)

        def _fake_announce(brief, **_kwargs):
            announced.append(brief)
            return announce_result

        orig_check = server_mod.connection_check
        orig_save = server_mod._save_brief
        orig_announce = _sa.announce_brief
        try:
            server_mod.connection_check = lambda: "Unread mail:\nNone."
            server_mod._save_brief = _fake_save
            _sa.announce_brief = _fake_announce
            result = server_mod._run_connection_job(
                {"action": "connection-check"})
        finally:
            server_mod.connection_check = orig_check
            server_mod._save_brief = orig_save
            _sa.announce_brief = orig_announce
        return result, saved, announced

    def test_connection_job_saves_and_announces_brief(self):
        result, saved, announced = self._run_job_with((True, "Announced"))
        self.assertEqual(result, "connection brief saved; brief announced")
        self.assertEqual(saved, ["Unread mail:\nNone."])
        self.assertEqual(announced, ["Unread mail:\nNone."])

    def test_connection_job_records_skipped_announcement(self):
        result, saved, announced = self._run_job_with((False, "lights off"))
        self.assertEqual(
            result, "connection brief saved; brief not announced: lights off")
        self.assertEqual(saved, ["Unread mail:\nNone."])
        self.assertEqual(announced, ["Unread mail:\nNone."])

    def test_other_job_actions_never_announce(self):
        import smart_announce as _sa

        announced = []
        orig_announce = _sa.announce_brief
        try:
            _sa.announce_brief = lambda brief, **_k: announced.append(brief)
            result = server_mod._run_connection_job({"action": "nonsense"})
        finally:
            _sa.announce_brief = orig_announce
        self.assertIn("No built-in action", result)
        self.assertEqual(announced, [])

    # --- urgent-email-check jobs (v0.3.10) --------------------------------
    def _run_urgent_job(self, token=True, messages=None, search_error=False):
        import connectors.google as _g
        import smart_announce as _sa

        orig_token = _g.token_exists
        orig_search = _g.gmail_search_structured
        orig_announce = _sa.announce_urgent_emails
        try:
            _g.token_exists = lambda: token
            if search_error:
                def _boom(*_a, **_k):
                    raise RuntimeError("boom")
                _g.gmail_search_structured = _boom
            else:
                _g.gmail_search_structured = lambda *a, **k: messages or []
            _sa.announce_urgent_emails = lambda msgs: "urgent emails: 1 announced"
            return server_mod._run_connection_job({"action": "urgent-email-check"})
        finally:
            _g.token_exists = orig_token
            _g.gmail_search_structured = orig_search
            _sa.announce_urgent_emails = orig_announce

    def test_urgent_job_announces(self):
        result = self._run_urgent_job(messages=[{"from": "a@b.com"}])
        self.assertEqual(result, "urgent-email-check: urgent emails: 1 announced")

    def test_urgent_job_without_google_records_not_connected(self):
        result = self._run_urgent_job(token=False)
        self.assertEqual(result, "urgent-email-check: Google not connected")

    def test_urgent_job_search_failure_is_recorded(self):
        result = self._run_urgent_job(search_error=True)
        self.assertIn("urgent-email-check failed", result)

    # --- follow-up-nudges jobs (v0.3.11) -------------------------------
    def _run_nudge_job(self, token=True, threads=None, search_error=False):
        import connectors.google as _g

        orig_token = True
        orig_threads = _g.gmail_unanswered_threads
        import smart_announce as _sa

        orig_announce = _sa.announce_followup_nudges
        try:
            _sa.announce_followup_nudges = lambda ts: "follow-up nudges: 2 announced"
            _g.token_exists = lambda: token
            if search_error:
                def _boom(*_a, **_k):
                    raise RuntimeError("boom")
                _g.gmail_unanswered_threads = _boom
            else:
                _g.gmail_unanswered_threads = lambda *a, **k: threads or []
            return server_mod._run_connection_job({"action": "follow-up-nudges"})
        finally:
            _g.token_exists = orig_token
            _g.gmail_unanswered_threads = orig_threads
            _sa.announce_followup_nudges = orig_announce

    def test_nudge_job_lists_unanswered_threads(self):
        result = self._run_nudge_job(threads=[
            {"subject": "Q?", "from": "boss@example.com"},
            {"subject": "", "from": "c@d.com"}])
        self.assertEqual(
            result,
            "follow-up-nudges: 2 awaiting a reply: Q? (boss@example.com), "
            "(no subject) (c@d.com); follow-up nudges: 2 announced")

    def test_nudge_job_none_awaiting(self):
        result = self._run_nudge_job(threads=[])
        self.assertEqual(result, "follow-up-nudges: nothing awaiting a reply")

    def test_nudge_job_without_google_records_not_connected(self):
        result = self._run_nudge_job(token=False)
        self.assertEqual(result, "follow-up-nudges: Google not connected")

    def test_nudge_job_search_failure_is_recorded(self):
        result = self._run_nudge_job(search_error=True)
        self.assertIn("follow-up-nudges failed", result)

    def test_nudge_job_announcement_failure_is_recorded(self):
        import connectors.google as _g
        import smart_announce as _sa

        orig_token = _g.token_exists
        orig_threads = _g.gmail_unanswered_threads
        orig_announce = _sa.announce_followup_nudges
        try:
            _g.token_exists = lambda: True
            _g.gmail_unanswered_threads = lambda *a, **k: [
                {"subject": "Q?", "from": "a@b.com"}]

            def _boom(*_a, **_k):
                raise RuntimeError("speaker down")

            _sa.announce_followup_nudges = _boom
            result = server_mod._run_connection_job(
                {"action": "follow-up-nudges"})
        finally:
            _g.token_exists = orig_token
            _g.gmail_unanswered_threads = orig_threads
            _sa.announce_followup_nudges = orig_announce
        self.assertIn("follow-up-nudges: 1 awaiting a reply", result)
        self.assertIn("announcement failed: RuntimeError", result)

    def test_tasks_create_accepts_nudge_action(self):
        client, r = self._login()
        self.assertEqual(r.status_code, 200)
        r = client.post("/api/tasks", json={
            "name": "nudges", "kind": "interval", "every_minutes": 60,
            "action": "follow-up-nudges"})
        self.assertEqual(r.status_code, 200)
        r = client.post("/api/tasks", json={
            "name": "bad", "kind": "interval", "every_minutes": 60,
            "action": "nonsense"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
