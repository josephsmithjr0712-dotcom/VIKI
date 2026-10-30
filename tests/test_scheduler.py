"""Tests for scheduler.py: job validation, interval/daily due logic,
once-per-day and anti-repeat guarantees, executor failures, and persistence."""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler import Scheduler


def fresh():
    return Scheduler(folder=tempfile.mkdtemp())


def at(hour, minute, day=15):
    return datetime(2026, 10, day, hour, minute)


class TestAdd(unittest.TestCase):
    def test_interval_job(self):
        s = fresh()
        job = s.add("Check email", "interval", action="connection-check", every_minutes=15)
        self.assertEqual(job["kind"], "interval")
        self.assertTrue(job["enabled"])
        self.assertEqual(job["run_count"], 0)
        self.assertIsNone(job["last_run"])

    def test_daily_job(self):
        job = fresh().add("Morning brief", "daily", at="08:30")
        self.assertEqual(job["at"], "08:30")

    def test_bad_inputs_rejected(self):
        s = fresh()
        for bad in [
            lambda: s.add("", "interval", every_minutes=5),
            lambda: s.add("x", "weekly"),
            lambda: s.add("x", "interval", every_minutes=0),
            lambda: s.add("x", "interval"),
            lambda: s.add("x", "daily", at="8:30"),
            lambda: s.add("x", "daily", at="25:00"),
            lambda: s.add("x", "daily"),
        ]:
            with self.assertRaises(ValueError):
                bad()

    def test_remove_and_disable(self):
        s = fresh()
        job = s.add("Check email", "interval", every_minutes=15)
        self.assertFalse(s.remove("nope"))
        self.assertTrue(s.set_enabled(job["id"], False))
        self.assertEqual(s.list_tasks()[0]["enabled"], False)
        self.assertTrue(s.remove(job["id"]))
        self.assertEqual(s.list_tasks(), [])


class TestDueLogic(unittest.TestCase):
    def test_new_interval_job_due_immediately(self):
        s = fresh()
        job = s.add("Check email", "interval", every_minutes=15)
        self.assertTrue(s.is_due(job, at(9, 0)))

    def test_interval_not_due_until_elapsed(self):
        s = fresh()
        s.add("Check email", "interval", every_minutes=15)
        s.run_due(lambda j: "ok", now=at(9, 0))
        job = s.list_tasks()[0]
        self.assertFalse(s.is_due(job, at(9, 14)))
        self.assertTrue(s.is_due(job, at(9, 15)))

    def test_interval_no_catchup_burst(self):
        s = fresh()
        s.add("Check email", "interval", every_minutes=15)
        calls = []
        s.run_due(lambda j: calls.append(j["name"]), now=at(9, 0))
        # Ticked again much later: fires once, not once per missed interval.
        s.run_due(lambda j: calls.append(j["name"]), now=at(12, 0))
        self.assertEqual(calls, ["Check email", "Check email"])

    def test_daily_due_once_per_day(self):
        s = fresh()
        job = s.add("Morning brief", "daily", at="08:30")
        self.assertFalse(s.is_due(job, at(8, 29)))
        self.assertTrue(s.is_due(job, at(8, 30)))
        s.run_due(lambda j: "ok", now=at(8, 30))
        job = s.list_tasks()[0]
        self.assertFalse(s.is_due(job, at(8, 45)))  # no repeat today
        self.assertTrue(s.is_due(job, at(8, 30, day=16)))  # due again tomorrow

    def test_daily_missed_day_fires_once(self):
        s = fresh()
        job = s.add("Morning brief", "daily", at="08:30")
        # First tick is days later, after the scheduled time: due once.
        self.assertTrue(s.is_due(job, at(9, 0, day=20)))
        s.run_due(lambda j: "ok", now=at(9, 0, day=20))
        job = s.list_tasks()[0]  # re-fetch: run_due stamped last_run
        self.assertFalse(s.is_due(job, at(9, 30, day=20)))

    def test_disabled_never_due(self):
        s = fresh()
        job = s.add("Check email", "interval", every_minutes=15)
        s.set_enabled(job["id"], False)
        self.assertFalse(s.is_due(s.list_tasks()[0], at(9, 0)))

    def test_next_run(self):
        s = fresh()
        job = s.add("Check email", "interval", every_minutes=15)
        self.assertEqual(s.next_run(job, at(9, 0)), at(9, 0))
        s.run_due(lambda j: "ok", now=at(9, 0))
        self.assertEqual(s.next_run(s.list_tasks()[0], at(9, 5)), at(9, 15))
        daily = s.add("Brief", "daily", at="08:30")
        self.assertEqual(s.next_run(daily, at(7, 0)), at(8, 30))


class TestRunDue(unittest.TestCase):
    def test_executor_result_recorded(self):
        s = fresh()
        s.add("Check email", "interval", every_minutes=15)
        out = s.run_due(lambda j: "found 0 urgent", now=at(9, 0))
        self.assertEqual(out[0]["result"], "found 0 urgent")
        job = s.list_tasks()[0]
        self.assertEqual(job["run_count"], 1)
        self.assertEqual(job["last_result"], "found 0 urgent")
        self.assertIsNone(job["last_error"])

    def test_executor_failure_recorded_not_raised(self):
        s = fresh()
        s.add("Flaky", "interval", every_minutes=5)
        s.add("Steady", "interval", every_minutes=5)

        def boom(job):
            if job["name"] == "Flaky":
                raise RuntimeError("mailbox down")
            return "ok"

        out = s.run_due(boom, now=at(9, 0))
        self.assertEqual(len(out), 2)
        flaky = s.list_tasks()[0]
        self.assertIn("mailbox down", flaky["last_error"])
        self.assertEqual(s.list_tasks()[1]["last_result"], "ok")
        # Claimed anyway: no immediate re-fire on the next tick.
        self.assertEqual(s.run_due(boom, now=at(9, 1)), [])

    def test_nothing_due_returns_empty_and_writes_nothing(self):
        s = fresh()
        s.add("Check email", "interval", every_minutes=60)
        s.run_due(lambda j: "ok", now=at(9, 0))
        self.assertEqual(s.run_due(lambda j: "ok", now=at(9, 30)), [])


class TestPersistence(unittest.TestCase):
    def test_jobs_survive_reload(self):
        folder = tempfile.mkdtemp()
        s = Scheduler(folder=folder)
        s.add("Check email", "interval", every_minutes=15)
        s.run_due(lambda j: "ok", now=at(9, 0))
        s2 = Scheduler(folder=folder)
        job = s2.list_tasks()[0]
        self.assertEqual(job["run_count"], 1)
        self.assertFalse(s2.is_due(job, at(9, 10)))

    def test_corrupt_file_recovered(self):
        folder = tempfile.mkdtemp()
        Path(folder, "tasks.json").write_text("{not json")
        s = Scheduler(folder=folder)
        self.assertIsNotNone(s.warning)
        self.assertEqual(s.list_tasks(), [])
        s.add("Fresh", "interval", every_minutes=10)
        self.assertEqual(len(s.list_tasks()), 1)


if __name__ == "__main__":
    unittest.main()
