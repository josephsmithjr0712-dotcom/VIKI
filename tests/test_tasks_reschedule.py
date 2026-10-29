"""Tests for rescheduling a scheduled task (v0.3.28).

Rename and pause landed, but changing *when* a check runs still meant
deleting it and recreating it — losing its run history. Now
Scheduler.reschedule, the reschedule_task chat tool, PATCH
/api/tasks/<id>, and a panel button all change the schedule in place:
the id, name, action, and history stay exactly as they were, and the
new spec is validated before anything is touched, so a bad schedule
leaves the old one alone.
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


class TestReschedule(unittest.TestCase):
    def test_reschedule_keeps_id_name_and_history(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", at="09:00")
        s.run_one(job["id"], lambda j: "checked ok")
        self.assertTrue(s.reschedule(job["id"], "interval",
                                     every_minutes=30))
        stored = s.list_tasks()[0]
        self.assertEqual(stored["id"], job["id"])
        self.assertEqual(stored["name"], "Morning")
        self.assertEqual(stored["kind"], "interval")
        self.assertEqual(stored["every_minutes"], 30)
        self.assertNotIn("at", stored)  # stale daily spec removed
        self.assertEqual(stored["run_count"], 1)
        self.assertEqual(stored["last_result"], "checked ok")

    def test_reschedule_to_daily_and_once(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Check", "interval", every_minutes=15)
        self.assertTrue(s.reschedule(job["id"], "daily", at="07:30"))
        stored = s.list_tasks()[0]
        self.assertEqual(stored["at"], "07:30")
        self.assertNotIn("every_minutes", stored)
        self.assertTrue(s.reschedule(job["id"], "once",
                                     at="2026-11-02 08:00"))
        stored = s.list_tasks()[0]
        self.assertEqual(stored["kind"], "once")
        self.assertEqual(stored["at"], "2026-11-02 08:00")

    def test_reschedule_unknown_id_returns_false(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        self.assertFalse(s.reschedule("nope", "daily", at="09:00"))

    def test_reschedule_bad_spec_raises_and_changes_nothing(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", at="09:00")
        for kind, kwargs in [("daily", {"at": "25:00"}),
                             ("interval", {"every_minutes": 0}),
                             ("once", {"at": "tomorrow"}),
                             ("weekly", {})]:
            with self.assertRaises(ValueError):
                s.reschedule(job["id"], kind, **kwargs)
        stored = s.list_tasks()[0]
        self.assertEqual(stored["kind"], "daily")
        self.assertEqual(stored["at"], "09:00")

    def test_add_still_validates_via_shared_spec(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        with self.assertRaises(ValueError):
            s.add("Bad", "daily", at="9am")
        self.assertEqual(s.list_tasks(), [])


class TestRescheduleTaskTool(unittest.TestCase):
    def test_tool_reschedules_and_reports(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily", at="09:00")
        text = tb._reschedule_task(job["id"], "interval",
                                   every_minutes=30)
        self.assertEqual(text, "Rescheduled 'Morning' to every 30 minutes.")
        stored = tb.scheduler.list_tasks()[0]
        self.assertEqual(stored["kind"], "interval")
        self.assertEqual(stored["every_minutes"], 30)

    def test_tool_unknown_id(self):
        tb = toolbox()
        self.assertEqual(tb._reschedule_task("nope", "daily", at="09:00"),
                         "No task with that id.")

    def test_tool_bad_spec_and_bad_minutes(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily", at="09:00")
        self.assertIn("Could not reschedule",
                      tb._reschedule_task(job["id"], "daily", at="25:00"))
        self.assertIn("whole number",
                      tb._reschedule_task(job["id"], "interval",
                                          every_minutes="soon"))
        self.assertEqual(tb.scheduler.list_tasks()[0]["at"], "09:00")

    def test_tool_is_registered_and_risky(self):
        tb = toolbox()
        self.assertIn("reschedule_task", tb.funcs)
        self.assertTrue(tb.needs_approval("reschedule_task",
                                          {"job_id": "x", "kind": "daily"}))

    def test_panel_contract_has_reschedule_button(self):
        html = (Path(__file__).resolve().parent.parent / "static"
                / "index.html").read_text()
        self.assertIn('data-action="reschedule"', html)
        self.assertIn("New schedule kind (daily, interval, or once):", html)
        self.assertIn("method:'PATCH'", html)


# Server endpoint tests need the repo's own import pattern (see
# tests/test_server.py): pin the web credentials to temp values so
# importing server.py never writes a .env into the repo, and stand in
# a FakeVIKI for the Ollama-backed agent, which is unavailable here.
import os as _os

_TMP = tempfile.mkdtemp(prefix="viki-reschedule-")
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
class TestTasksRescheduleEndpoint(unittest.TestCase):
    def _client(self):
        client = server_mod.app.test_client()
        with client.session_transaction() as sess:
            sess["viki_auth"] = True
        return client

    def test_patch_reschedule_and_errors(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        job = sched.add("Morning", "daily", at="09:00")
        sched.run_one(job["id"], lambda j: "checked ok")
        old_sched = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            client = self._client()
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"kind": "interval",
                                      "every_minutes": 30})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json()["kind"], "interval")
            stored = sched.list_tasks()[0]
            self.assertEqual(stored["every_minutes"], 30)
            self.assertEqual(stored["name"], "Morning")
            self.assertEqual(stored["run_count"], 1)
            # Bad spec is rejected and changes nothing.
            self.assertEqual(client.patch(f"/api/tasks/{job['id']}",
                                          json={"kind": "daily"}).status_code,
                             400)
            self.assertEqual(sched.list_tasks()[0]["kind"], "interval")
            self.assertEqual(client.patch(f"/api/tasks/{job['id']}",
                                          json={"kind": "weekly"}).status_code,
                             400)
            self.assertEqual(client.patch("/api/tasks/nope",
                                          json={"kind": "daily",
                                                "at": "09:00"}).status_code,
                             404)
            # Rename + reschedule together through the same route.
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"name": "Early",
                                      "kind": "daily", "at": "07:30"})
            self.assertEqual(resp.status_code, 200)
            stored = sched.list_tasks()[0]
            self.assertEqual(stored["name"], "Early")
            self.assertEqual(stored["at"], "07:30")
        finally:
            server_mod.task_scheduler = old_sched


if __name__ == "__main__":
    unittest.main()
