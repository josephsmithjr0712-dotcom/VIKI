"""Tests for surfacing a scheduled task's last outcome (v0.3.22).

v0.3.21 showed the *next* run everywhere, but the *last* outcome was
still half-hidden: chat list_tasks omitted last_result (despite its
spec promising "last result"), and the web tasks panel showed neither
the last run nor its result/error — so a scheduled check that failed
(or found something) looked identical to one that never ran. Now both
surfaces render the last run plus its result or error, with long
results truncated so neither surface blows up.
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

HTML = (Path(__file__).resolve().parent.parent
        / "static" / "index.html").read_text(errors="ignore")


def toolbox():
    tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
    tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
    return tb


def scheduled(tb):
    tb.run("schedule_task", {"name": "Morning", "kind": "daily",
                             "action": "connection-check", "at": "09:00"})
    return tb.scheduler


class TestListTasksLastOutcome(unittest.TestCase):
    def test_listing_shows_last_result(self):
        tb = toolbox()
        sched = scheduled(tb)
        sched.run_due(lambda j: "connection brief saved; brief announced",
                      now=datetime(2026, 10, 28, 9, 0))
        listing = tb.run("list_tasks", {})
        self.assertIn("last run 2026-10-28", listing)
        self.assertIn("last result: connection brief saved; brief announced",
                      listing)

    def test_listing_shows_last_error_not_result(self):
        tb = toolbox()
        sched = scheduled(tb)

        def boom(job):
            raise RuntimeError("boom")

        sched.run_due(boom, now=datetime(2026, 10, 28, 9, 0))
        listing = tb.run("list_tasks", {})
        self.assertIn("last error: RuntimeError: boom", listing)
        self.assertNotIn("last result:", listing)

    def test_long_result_truncated(self):
        tb = toolbox()
        sched = scheduled(tb)
        sched.run_due(lambda j: "x" * 300, now=datetime(2026, 10, 28, 9, 0))
        listing = tb.run("list_tasks", {})
        self.assertIn("last result:", listing)
        self.assertIn("…", listing)
        self.assertNotIn("x" * 201, listing)

    def test_never_ran_shows_no_last_outcome(self):
        tb = toolbox()
        scheduled(tb)
        listing = tb.run("list_tasks", {})
        self.assertNotIn("last run", listing)
        self.assertNotIn("last result", listing)
        self.assertNotIn("last error", listing)


class TestTasksPanelLastOutcome(unittest.TestCase):
    def test_panel_renders_last_run_result_and_error(self):
        # Contract test (same pattern as test_handsfree): the panel's
        # loadTasks renderer must reference all three stored fields.
        self.assertIn("t.last_run", HTML)
        self.assertIn("t.last_result", HTML)
        self.assertIn("t.last_error", HTML)


try:
    import server as server_mod
    _HAS_SERVER = server_mod is not None
except Exception:
    server_mod = None
    _HAS_SERVER = False


@unittest.skipUnless(_HAS_SERVER, "server.py not importable here")
class TestTasksEndpointLastOutcome(unittest.TestCase):
    def test_api_tasks_include_last_outcome(self):
        folder = tempfile.mkdtemp()
        sched = Scheduler(folder=folder)
        sched.add("Morning", "daily", at="09:00")
        sched.run_due(lambda j: "connection brief saved; brief announced",
                      now=datetime(2026, 10, 28, 9, 0))
        old = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            with server_mod.app.test_request_context():
                resp = server_mod.tasks_list()
            data = json.loads(resp.get_data(as_text=True))
        finally:
            server_mod.task_scheduler = old
        task = data["tasks"][0]
        self.assertEqual(task["last_result"],
                         "connection brief saved; brief announced")
        self.assertIsNone(task["last_error"])


if __name__ == "__main__":
    unittest.main()
