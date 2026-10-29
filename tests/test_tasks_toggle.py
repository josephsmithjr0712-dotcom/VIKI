"""Tests for pausing/resuming scheduled tasks (v0.3.23).

Scheduler.set_enabled already existed, but nothing could reach it: no
chat tool, no API route, no panel button — the only way to stop a
check was to delete it. Now toggle_task (chat), PATCH /api/tasks/<id>
(web), and a pause/resume button (panel) all flip the same flag.
"""

import json
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


class TestToggleTask(unittest.TestCase):
    def test_toggle_pauses_then_resumes(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily", at="09:00")
        self.assertEqual(tb._toggle_task(job["id"]), "Paused 'Morning'.")
        self.assertFalse(tb.scheduler.list_tasks()[0]["enabled"])
        self.assertIn("off", tb._list_tasks())
        self.assertEqual(tb._toggle_task(job["id"]), "Resumed 'Morning'.")
        self.assertTrue(tb.scheduler.list_tasks()[0]["enabled"])
        self.assertIn("on", tb._list_tasks())

    def test_toggle_unknown_id(self):
        tb = toolbox()
        self.assertEqual(tb._toggle_task("nope"), "No task with that id.")

    def test_paused_job_does_not_run(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily", at="09:00")
        tb._toggle_task(job["id"])
        from datetime import datetime
        ran = tb.scheduler.run_due(lambda j: "ok", now=datetime(2026, 10, 28, 9, 0))
        self.assertEqual(ran, [])

    def test_toggle_tool_is_registered_and_risky(self):
        tb = toolbox()
        self.assertIn("toggle_task", tb.funcs)
        self.assertTrue(tb.needs_approval("toggle_task", {"job_id": "x"}))

    def test_panel_contract_has_toggle_button_and_patch(self):
        html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text()
        self.assertIn('data-action="toggle"', html)
        self.assertIn("method:'PATCH'", html)
        self.assertIn("/api/tasks/", html)


try:
    import server as server_mod
    _HAS_SERVER = server_mod is not None
except Exception:
    server_mod = None
    _HAS_SERVER = False


@unittest.skipUnless(_HAS_SERVER, "server.py not importable here")
class TestTasksToggleEndpoint(unittest.TestCase):
    def test_patch_toggles_enabled(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        job = sched.add("Morning", "daily", at="09:00")
        old = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            client = server_mod.app.test_client()
            with client.session_transaction() as sess:
                sess["viki_auth"] = True
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"enabled": False})
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(sched.list_tasks()[0]["enabled"])
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"enabled": True})
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(sched.list_tasks()[0]["enabled"])
        finally:
            server_mod.task_scheduler = old

    def test_patch_unknown_and_bad_body(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        old = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            client = server_mod.app.test_client()
            with client.session_transaction() as sess:
                sess["viki_auth"] = True
            self.assertEqual(client.patch("/api/tasks/nope",
                                          json={"enabled": False}).status_code, 404)
            self.assertEqual(client.patch("/api/tasks/nope",
                                          json={}).status_code, 400)
        finally:
            server_mod.task_scheduler = old


if __name__ == "__main__":
    unittest.main()
