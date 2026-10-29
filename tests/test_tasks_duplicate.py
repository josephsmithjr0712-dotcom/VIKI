"""Tests for duplicating a scheduled task (v0.3.29).

Rename and reschedule let you fix a check in place, but trying a
variation of one still meant recreating it by hand — and getting its
schedule or action slightly wrong. Now Scheduler.duplicate, the
duplicate_task chat tool, POST /api/tasks/<id>/duplicate, and a panel
button all copy a check: same schedule and action, fresh history,
named "<name> (copy)", with the original untouched.
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


class TestDuplicate(unittest.TestCase):
    def test_duplicate_keeps_schedule_action_and_clears_history(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", action="connection-check",
                    at="09:00")
        s.run_one(job["id"], lambda j: "checked ok")
        copy = s.duplicate(job["id"])
        self.assertNotEqual(copy["id"], job["id"])
        self.assertEqual(copy["name"], "Morning (copy)")
        self.assertEqual(copy["kind"], "daily")
        self.assertEqual(copy["at"], "09:00")
        self.assertEqual(copy["action"], "connection-check")
        self.assertIsNone(copy["last_run"])
        self.assertEqual(copy["run_count"], 0)
        self.assertIsNone(copy["last_result"])
        self.assertIsNone(copy["last_error"])
        # The original is untouched, history included.
        original = s.list_tasks()[0]
        self.assertEqual(original["name"], "Morning")
        self.assertEqual(original["run_count"], 1)
        self.assertEqual(original["last_result"], "checked ok")

    def test_duplicate_interval_and_paused_state(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Frequent", "interval", action="connection-check",
                    every_minutes=15)
        s.set_enabled(job["id"], False)
        copy = s.duplicate(job["id"])
        self.assertEqual(copy["every_minutes"], 15)
        self.assertFalse(copy["enabled"])
        self.assertEqual(len(s.list_tasks()), 2)

    def test_duplicate_unknown_id_returns_none(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        self.assertIsNone(s.duplicate("nope"))


class TestDuplicateTaskTool(unittest.TestCase):
    def test_tool_duplicates_and_reports(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily", at="09:00")
        text = tb._duplicate_task(job["id"])
        self.assertTrue(text.startswith("Duplicated 'Morning' as 'Morning (copy)'"))
        self.assertEqual(len(tb.scheduler.list_tasks()), 2)

    def test_tool_unknown_id(self):
        tb = toolbox()
        self.assertEqual(tb._duplicate_task("nope"),
                         "No task with that id.")

    def test_tool_is_registered_and_risky(self):
        tb = toolbox()
        self.assertIn("duplicate_task", tb.funcs)
        self.assertTrue(tb.needs_approval("duplicate_task",
                                          {"job_id": "x"}))

    def test_panel_contract_has_duplicate_button(self):
        html = (Path(__file__).resolve().parent.parent / "static"
                / "index.html").read_text()
        self.assertIn('data-action="duplicate"', html)
        self.assertIn("/duplicate", html)


# Server endpoint tests need the repo's own import pattern (see
# tests/test_server.py): pin the web credentials to temp values so
# importing server.py never writes a .env into the repo, and stand in
# a FakeVIKI for the Ollama-backed agent, which is unavailable here.
import os as _os

_TMP = tempfile.mkdtemp(prefix="viki-duplicate-")
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
class TestTasksDuplicateEndpoint(unittest.TestCase):
    def _client(self):
        client = server_mod.app.test_client()
        with client.session_transaction() as sess:
            sess["viki_auth"] = True
        return client

    def test_duplicate_endpoint_and_404(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        job = sched.add("Morning", "daily", at="09:00")
        sched.run_one(job["id"], lambda j: "checked ok")
        old_sched = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            client = self._client()
            resp = client.post(f"/api/tasks/{job['id']}/duplicate")
            self.assertEqual(resp.status_code, 200)
            copy = resp.get_json()["task"]
            self.assertEqual(copy["name"], "Morning (copy)")
            self.assertNotEqual(copy["id"], job["id"])
            self.assertEqual(copy["run_count"], 0)
            self.assertEqual(len(sched.list_tasks()), 2)
            self.assertEqual(client.post("/api/tasks/nope/duplicate").status_code,
                             404)
        finally:
            server_mod.task_scheduler = old_sched


if __name__ == "__main__":
    unittest.main()
