"""Tests for showing a scheduled check's action and run count (v0.3.32).

run_count has been tracked since the scheduler began, but it was shown
nowhere: the chat list omitted both the action and the count, and the
web panel omitted the count. So a check that had run fifty times looked
identical to one that had never run, and in chat you could not even tell
what a check does. Now the chat list shows action + run count, and the
panel shows the run count too. No server change: /api/tasks already
returned run_count inside each job dict.
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import SessionStore
from scheduler import Scheduler
from tools import Toolbox

HTML = (Path(__file__).resolve().parent.parent
        / "static" / "index.html").read_text(errors="ignore")


def toolbox():
    tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
    tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
    return tb


def scheduled(tb, kind="daily", **kw):
    args = {"name": "Morning", "kind": kind, "action": "connection-check"}
    args.update(kw)
    if kind == "daily":
        args.setdefault("at", "09:00")
    elif kind == "interval":
        args.setdefault("every_minutes", 60)
    tb.run("schedule_task", args)
    return tb.scheduler


class TestListTasksShowRuns(unittest.TestCase):
    def test_never_ran_shows_action_and_zero_runs(self):
        tb = toolbox()
        scheduled(tb)
        listing = tb.run("list_tasks", {})
        self.assertIn("action connection-check", listing)
        self.assertIn("0 runs", listing)

    def test_one_run_shows_singular(self):
        tb = toolbox()
        sched = scheduled(tb)
        sched.run_due(lambda j: "ok", now=datetime(2026, 10, 29, 9, 0))
        listing = tb.run("list_tasks", {})
        self.assertIn("1 run", listing)
        self.assertNotIn("1 runs", listing)

    def test_two_runs_show_plural(self):
        tb = toolbox()
        sched = scheduled(tb, kind="interval")
        sched.run_due(lambda j: "ok", now=datetime(2026, 10, 29, 9, 0))
        sched.run_due(lambda j: "ok",
                      now=datetime(2026, 10, 29, 9, 0) + timedelta(minutes=61))
        listing = tb.run("list_tasks", {})
        self.assertIn("2 runs", listing)

    def test_duplicate_copy_starts_at_zero(self):
        tb = toolbox()
        sched = scheduled(tb)
        sched.run_due(lambda j: "ok", now=datetime(2026, 10, 29, 9, 0))
        tb.run("duplicate_task", {"job_id": sched.list_tasks()[0]["id"]})
        listing = tb.run("list_tasks", {})
        self.assertIn("1 run", listing)
        self.assertIn("0 runs", listing)


class TestTasksPanelShowRuns(unittest.TestCase):
    def test_panel_renders_run_count(self):
        self.assertIn("t.run_count", HTML)


try:
    import server as server_mod
    _HAS_SERVER = server_mod is not None
except Exception:
    server_mod = None
    _HAS_SERVER = False


@unittest.skipUnless(_HAS_SERVER, "server.py not importable here")
class TestTasksEndpointShowRuns(unittest.TestCase):
    def test_api_tasks_include_run_count(self):
        folder = tempfile.mkdtemp()
        sched = Scheduler(folder=folder)
        sched.add("Morning", "daily", at="09:00")
        sched.run_due(lambda j: "ok", now=datetime(2026, 10, 29, 9, 0))
        old = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            with server_mod.app.test_request_context():
                resp = server_mod.tasks_list()
            data = json.loads(resp.get_data(as_text=True))
        finally:
            server_mod.task_scheduler = old
        self.assertEqual(data["tasks"][0]["run_count"], 1)


if __name__ == "__main__":
    unittest.main()
