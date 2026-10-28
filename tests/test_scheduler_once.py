"""Tests for one-shot ("once") scheduler jobs: a job that runs a single
time at a full date+time, then switches itself off so it never repeats.
Covers Scheduler validation/due/next-run/run-due/persistence plus the
Toolbox schedule_task/list_tasks surface (which needs no server.py change:
the /api/tasks endpoint already passes kind/at through to Scheduler.add).
"""
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


WHEN = "2026-11-02 07:00"
WHEN_DT = datetime(2026, 11, 2, 7, 0)


class TestOnceAdd(unittest.TestCase):
    def test_add_and_normalize(self):
        job = fresh().add("One-shot brief", "once", at=WHEN)
        self.assertEqual(job["kind"], "once")
        self.assertEqual(job["at"], WHEN)
        self.assertTrue(job["enabled"])

    def test_t_separator_normalizes(self):
        job = fresh().add("T form", "once", at="2026-11-02T07:00")
        self.assertEqual(job["at"], WHEN)

    def test_bad_at_rejected(self):
        s = fresh()
        for bad in ["", "tomorrow", "08:30", "2026-13-02 07:00",
                    "2026-11-02", "2026/11/02 07:00"]:
            with self.assertRaises(ValueError, msg=bad):
                s.add("x", "once", at=bad)

    def test_missing_at_rejected(self):
        with self.assertRaises(ValueError):
            fresh().add("x", "once")


class TestOnceDue(unittest.TestCase):
    def test_not_due_before(self):
        s = fresh()
        job = s.add("Brief", "once", at=WHEN)
        self.assertFalse(s.is_due(job, now=datetime(2026, 11, 2, 6, 59)))
        self.assertFalse(s.is_due(job, now=datetime(2026, 11, 1, 7, 0)))

    def test_due_at_and_after(self):
        s = fresh()
        job = s.add("Brief", "once", at=WHEN)
        self.assertTrue(s.is_due(job, now=WHEN_DT))
        self.assertTrue(s.is_due(job, now=datetime(2026, 11, 3, 7, 0)))

    def test_disabled_never_due(self):
        s = fresh()
        job = s.add("Brief", "once", at=WHEN)
        s.set_enabled(job["id"], False)
        job = [t for t in s.list_tasks() if t["id"] == job["id"]][0]
        self.assertFalse(s.is_due(job, now=datetime(2026, 11, 3, 7, 0)))

    def test_next_run(self):
        s = fresh()
        job = s.add("Brief", "once", at=WHEN)
        self.assertEqual(s.next_run(job, now=datetime(2026, 11, 1, 7, 0)), WHEN_DT)
        # Overdue and never run: due now, not at the stale time.
        self.assertEqual(s.next_run(job, now=datetime(2026, 11, 3, 7, 0)),
                         datetime(2026, 11, 3, 7, 0))

    def test_fires_exactly_once(self):
        s = fresh()
        job = s.add("Brief", "once", at=WHEN)
        fired = []
        out = s.run_due(lambda j: fired.append(j["id"]), now=WHEN_DT)
        self.assertEqual(len(out), 1)
        self.assertEqual(fired, [job["id"]])
        stored = [t for t in s.list_tasks() if t["id"] == job["id"]][0]
        self.assertFalse(stored["enabled"])  # switched off after firing
        self.assertEqual(stored["run_count"], 1)
        self.assertEqual(s.run_due(lambda j: fired.append(j["id"]),
                                   now=datetime(2026, 11, 3, 7, 0)), [])
        self.assertEqual(len(fired), 1)
        self.assertIsNone(s.next_run(stored, now=datetime(2026, 11, 3, 7, 0)))

    def test_persists_across_reload(self):
        folder = tempfile.mkdtemp()
        job = Scheduler(folder=folder).add("Brief", "once", at=WHEN)
        reloaded = [t for t in Scheduler(folder=folder).list_tasks()
                    if t["id"] == job["id"]][0]
        self.assertEqual(reloaded["kind"], "once")
        self.assertEqual(reloaded["at"], WHEN)


class TestOnceToolbox(unittest.TestCase):
    def test_schedule_and_list(self):
        tb = toolbox()
        msg = tb.run("schedule_task", {"name": "One-shot brief", "kind": "once",
                                       "action": "connection-check",
                                       "at": WHEN})
        self.assertIn("once at 2026-11-02 07:00", msg)
        listing = tb.run("list_tasks", {})
        self.assertIn("once at 2026-11-02 07:00", listing)

    def test_bad_once_via_tool(self):
        tb = toolbox()
        msg = tb.run("schedule_task", {"name": "Bad", "kind": "once",
                                       "action": "connection-check",
                                       "at": "tomorrow"})
        self.assertIn("Could not schedule", msg)

    def test_other_kinds_unchanged(self):
        tb = toolbox()
        msg = tb.run("schedule_task", {"name": "Daily", "kind": "daily",
                                       "action": "connection-check",
                                       "at": "08:30"})
        self.assertIn("daily at 08:30", msg)


if __name__ == "__main__":
    unittest.main()
