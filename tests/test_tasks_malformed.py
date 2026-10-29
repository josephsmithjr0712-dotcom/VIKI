"""Tests for malformed scheduled jobs (v0.3.35).

A hand-edited tasks.json can hold a job whose spec is missing or
garbage (no every_minutes, a non-numeric one, a bad daily time).
Before this fix, one such job crashed is_due/next_run with KeyError /
ValueError / TypeError — which broke the whole tick (run_due) and the
whole task list (tasks_with_next_run), so one bad job silenced every
good one. Now a malformed job simply never fires and reports no next
run, and the healthy jobs around it keep working.
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


NOW = datetime(2026, 10, 29, 8, 0)


def inject(s, **overrides):
    """Append a job straight to the store, bypassing add()'s validation,
    the way a hand-edited tasks.json would."""
    job = {"id": "bad-1", "name": "Bad", "kind": "interval",
           "action": "", "enabled": True, "created": "2026-10-29T00:00:00",
           "last_run": None, "run_count": 0, "last_result": None,
           "last_error": None}
    job.update(overrides)
    s._tasks.append(job)
    return job


class TestMalformedJobs(unittest.TestCase):
    def test_interval_missing_every_minutes_never_due(self):
        s = fresh()
        job = inject(s)  # no every_minutes key at all
        self.assertFalse(s.is_due(job, now=NOW))
        self.assertIsNone(s.next_run(job, now=NOW))

    def test_interval_string_every_minutes_never_due(self):
        s = fresh()
        job = inject(s, every_minutes="60")
        self.assertFalse(s.is_due(job, now=NOW))
        self.assertIsNone(s.next_run(job, now=NOW))

    def test_interval_zero_every_minutes_never_due(self):
        s = fresh()
        job = inject(s, every_minutes=0)
        self.assertFalse(s.is_due(job, now=NOW))
        self.assertIsNone(s.next_run(job, now=NOW))

    def test_daily_bad_time_never_due(self):
        s = fresh()
        job = inject(s, kind="daily", at="99:99")
        self.assertFalse(s.is_due(job, now=NOW))
        self.assertIsNone(s.next_run(job, now=NOW))

    def test_daily_missing_at_never_due(self):
        s = fresh()
        job = inject(s, kind="daily")  # no at key at all
        self.assertFalse(s.is_due(job, now=NOW))
        self.assertIsNone(s.next_run(job, now=NOW))

    def test_tick_and_list_survive_bad_jobs(self):
        s = fresh()
        inject(s, id="bad-1")  # interval, missing every_minutes
        inject(s, id="bad-2", kind="daily", at="nope")
        good = s.add("Good", "interval", every_minutes=60)
        # The list doesn't crash and reports no next run for bad jobs.
        listed = {t["id"]: t for t in s.tasks_with_next_run(now=NOW)}
        self.assertIsNone(listed["bad-1"]["next_run"])
        self.assertIsNone(listed["bad-2"]["next_run"])
        self.assertIsNotNone(listed[good["id"]]["next_run"])
        # The tick doesn't crash and runs only the good job.
        ran = []
        outcomes = s.run_due(lambda job: ran.append(job["id"]), now=NOW)
        self.assertEqual(ran, [good["id"]])
        self.assertEqual([o["id"] for o in outcomes], [good["id"]])

    def test_chat_list_survives_bad_jobs(self):
        tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
        inject(tb.scheduler, id="bad-1")  # interval, missing every_minutes
        inject(tb.scheduler, id="bad-2", kind="daily")  # daily, missing at
        text = tb._list_tasks()
        self.assertIn("bad-1", text)
        self.assertIn("bad-2", text)


if __name__ == "__main__":
    unittest.main()
