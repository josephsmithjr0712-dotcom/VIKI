"""Tests for surfacing a scheduled task's next run (v0.3.21).

Scheduler.next_run existed but nothing showed it: the chat list_tasks
text, the /api/tasks endpoint, and the web tasks panel all showed only
the schedule spec. Now tasks_with_next_run() attaches the computed
timestamp (or None for disabled / already-fired jobs) and every
surface renders it.
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import SessionStore
from scheduler import Scheduler
from tools import Toolbox


def fresh():
    return Scheduler(folder=tempfile.mkdtemp())


def toolbox():
    tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
    tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
    return tb


NOW = datetime(2026, 10, 28, 8, 0)


class TestTasksWithNextRun(unittest.TestCase):
    def test_daily_next_run_today(self):
        s = fresh()
        s.add("Morning", "daily", at="09:00")
        tasks = s.tasks_with_next_run(now=NOW)
        self.assertEqual(tasks[0]["next_run"], "2026-10-28T09:00:00")

    def test_daily_already_ran_today_rolls_to_tomorrow(self):
        s = fresh()
        job = s.add("Morning", "daily", at="07:00")
        s.run_due(lambda j: "ok", now=datetime(2026, 10, 28, 7, 0))
        tasks = s.tasks_with_next_run(now=NOW)
        self.assertEqual(tasks[0]["next_run"], "2026-10-29T07:00:00")

    def test_interval_next_run_from_last_run(self):
        s = fresh()
        s.add("Poll", "interval", every_minutes=60)
        s.run_due(lambda j: "ok", now=datetime(2026, 10, 28, 7, 30))
        tasks = s.tasks_with_next_run(now=NOW)
        self.assertEqual(tasks[0]["next_run"], "2026-10-28T08:30:00")

    def test_disabled_job_reports_none(self):
        s = fresh()
        job = s.add("Off", "daily", at="09:00")
        s.set_enabled(job["id"], False)
        self.assertIsNone(s.tasks_with_next_run(now=NOW)[0]["next_run"])

    def test_fired_once_job_reports_none(self):
        s = fresh()
        s.add("One-shot", "once", at="2026-10-28 07:00")
        s.run_due(lambda j: "ok", now=datetime(2026, 10, 28, 7, 0))
        task = s.tasks_with_next_run(now=NOW)[0]
        self.assertFalse(task["enabled"])
        self.assertIsNone(task["next_run"])

    def test_pending_once_job_reports_its_time(self):
        s = fresh()
        s.add("One-shot", "once", at="2026-11-02 07:00")
        self.assertEqual(s.tasks_with_next_run(now=NOW)[0]["next_run"],
                         "2026-11-02T07:00:00")

    def test_list_tasks_unchanged(self):
        # Plain list_tasks keeps the stored shape (no next_run key added),
        # so existing callers and persistence are unaffected.
        s = fresh()
        s.add("Morning", "daily", at="09:00")
        self.assertNotIn("next_run", s.list_tasks()[0])


class TestListTasksText(unittest.TestCase):
    def test_listing_shows_next_run(self):
        tb = toolbox()
        tb.run("schedule_task", {"name": "Morning", "kind": "daily",
                                "action": "connection-check", "at": "09:00"})
        listing = tb.run("list_tasks", {})
        self.assertIn("daily at 09:00", listing)
        self.assertIn("next run", listing)


try:
    import server as server_mod
    _HAS_SERVER = server_mod is not None
except Exception:
    server_mod = None
    _HAS_SERVER = False


@unittest.skipUnless(_HAS_SERVER, "server.py not importable here")
class TestTasksEndpointNextRun(unittest.TestCase):
    def test_api_tasks_include_next_run(self):
        # Wiring test only: the endpoint must return exactly what
        # Scheduler.tasks_with_next_run computes (exact timestamps are
        # covered deterministically in TestTasksWithNextRun above, since
        # the endpoint itself uses wall-clock now).
        folder = tempfile.mkdtemp()
        sched = Scheduler(folder=folder)
        sched.add("Morning", "daily", at="09:00")
        old = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            with server_mod.app.test_request_context():
                resp = server_mod.tasks_list()
            data = json.loads(resp.get_data(as_text=True))
            expected = sched.tasks_with_next_run()[0]["next_run"]
        finally:
            server_mod.task_scheduler = old
        self.assertEqual(len(data["tasks"]), 1)
        self.assertEqual(data["tasks"][0]["name"], "Morning")
        self.assertIn("next_run", data["tasks"][0])
        got = datetime.fromisoformat(data["tasks"][0]["next_run"])
        want = datetime.fromisoformat(expected)
        self.assertLessEqual(abs((got - want).total_seconds()), 2)


if __name__ == "__main__":
    unittest.main()
