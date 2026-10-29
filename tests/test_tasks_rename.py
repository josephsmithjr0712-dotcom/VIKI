"""Tests for renaming a scheduled task (v0.3.26).

Pause, run-now, and the create form all landed, but a typo in a
check's name still meant deleting it and recreating it — losing its
run history. Now Scheduler.rename, the rename_task chat tool,
PATCH /api/tasks/<id>, and a panel button all rename in place: the
id, schedule, and history stay exactly as they were.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import SessionStore
from scheduler import Scheduler
from tools import Toolbox


def toolbox():
    tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
    tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
    return tb


class TestRename(unittest.TestCase):
    def test_rename_keeps_id_schedule_and_history(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Mornign", "daily", at="09:00")
        s.run_one(job["id"], lambda j: "checked ok")
        self.assertTrue(s.rename(job["id"], "  Morning  check "))
        stored = s.list_tasks()[0]
        self.assertEqual(stored["id"], job["id"])
        self.assertEqual(stored["name"], "Morning check")
        self.assertEqual(stored["at"], "09:00")
        self.assertEqual(stored["run_count"], 1)
        self.assertEqual(stored["last_result"], "checked ok")

    def test_rename_unknown_id_returns_false(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        self.assertFalse(s.rename("nope", "New"))

    def test_rename_blank_name_raises(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", at="09:00")
        with self.assertRaises(ValueError):
            s.rename(job["id"], "   ")
        self.assertEqual(s.list_tasks()[0]["name"], "Morning")


class TestRenameTaskTool(unittest.TestCase):
    def test_tool_renames_and_reports(self):
        tb = toolbox()
        job = tb.scheduler.add("Mornign", "daily", at="09:00")
        text = tb._rename_task(job["id"], "Morning")
        self.assertEqual(text, "Renamed 'Mornign' to 'Morning'.")
        self.assertEqual(tb.scheduler.list_tasks()[0]["name"], "Morning")

    def test_tool_unknown_id(self):
        tb = toolbox()
        self.assertEqual(tb._rename_task("nope", "New"),
                         "No task with that id.")

    def test_tool_blank_name(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily", at="09:00")
        self.assertIn("Could not rename", tb._rename_task(job["id"], "  "))
        self.assertEqual(tb.scheduler.list_tasks()[0]["name"], "Morning")

    def test_tool_is_registered_and_risky(self):
        tb = toolbox()
        self.assertIn("rename_task", tb.funcs)
        self.assertTrue(tb.needs_approval("rename_task",
                                          {"job_id": "x", "name": "y"}))

    def test_panel_contract_has_rename_button(self):
        html = (Path(__file__).resolve().parent.parent / "static"
                / "index.html").read_text()
        self.assertIn('data-action="rename"', html)
        self.assertIn("prompt('Rename this check:'", html)
        self.assertIn("method:'PATCH'", html)


# Server endpoint tests need the repo's own import pattern (see
# tests/test_server.py): pin the web credentials to temp values so
# importing server.py never writes a .env into the repo, and stand in
# a FakeVIKI for the Ollama-backed agent, which is unavailable here.
import os as _os

_TMP = tempfile.mkdtemp(prefix="viki-rename-")
for _k, _v in {"VIKI_SESSION_DIR": _TMP,
               "VIKI_SECRET_KEY": "test-secret-key",
               "VIKI_WEB_PASSWORD": "test-password",
               "VIKI_AUTO_MEMORY": "off"}.items():
    _os.environ.setdefault(_k, _v)

try:
    import agent as _agent_mod

    class _FakeVIKI:
        def __init__(self, toolbox):
            self.toolbox = toolbox
            self.model = "test-model"

        def warmup(self):
            pass

    _real_VIKI = _agent_mod.VIKI
    _agent_mod.VIKI = _FakeVIKI
    try:
        import server as server_mod
    finally:
        _agent_mod.VIKI = _real_VIKI
    _HAS_SERVER = server_mod is not None
except Exception:
    server_mod = None
    _HAS_SERVER = False


@unittest.skipUnless(_HAS_SERVER, "server.py not importable here")
class TestTasksRenameEndpoint(unittest.TestCase):
    def _client(self):
        client = server_mod.app.test_client()
        with client.session_transaction() as sess:
            sess["viki_auth"] = True
        return client

    def test_patch_rename_and_errors(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        job = sched.add("Mornign", "daily", at="09:00")
        old_sched = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            client = self._client()
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"name": "Morning"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json()["name"], "Morning")
            self.assertEqual(sched.list_tasks()[0]["name"], "Morning")
            self.assertEqual(sched.list_tasks()[0]["at"], "09:00")
            self.assertEqual(client.patch("/api/tasks/nope",
                                          json={"name": "X"}).status_code,
                             404)
            self.assertEqual(client.patch(f"/api/tasks/{job['id']}",
                                          json={"name": "  "}).status_code,
                             400)
            self.assertEqual(client.patch(f"/api/tasks/{job['id']}",
                                          json={}).status_code, 400)
            # Toggle still works through the same PATCH route.
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"enabled": False})
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(resp.get_json()["enabled"])
        finally:
            server_mod.task_scheduler = old_sched


if __name__ == "__main__":
    unittest.main()
