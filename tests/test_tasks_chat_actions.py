"""Tests for chat task tools accepting every built-in check action (v0.3.33).

The web panel always offered all three executor actions, but the chat
tools lagged: schedule_task refused anything but 'connection-check'
("aren't wired yet" — stale, the tick runs them all), and run_task_now
always ran a connection check even for an urgent-email or follow-up
job, recording the wrong outcome under that job's name. Now chat
matches the panel: schedule_task takes any built-in action, and
run_task_now dispatches on the job's own action.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import smart_announce
import tools as tools_mod
from memory import SessionStore
from scheduler import Scheduler
from tools import Toolbox


def toolbox():
    tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
    tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
    return tb


class TestScheduleTaskActions(unittest.TestCase):
    def test_accepts_urgent_email_check(self):
        tb = toolbox()
        msg = tb.run("schedule_task", {"name": "Urgent watch", "kind": "daily",
                                       "at": "08:00", "action": "urgent-email-check"})
        self.assertIn("Scheduled", msg)
        self.assertEqual(tb.scheduler.list_tasks()[0]["action"], "urgent-email-check")

    def test_accepts_follow_up_nudges(self):
        tb = toolbox()
        msg = tb.run("schedule_task", {"name": "Nudge watch", "kind": "daily",
                                       "at": "08:00", "action": "follow-up-nudges"})
        self.assertIn("Scheduled", msg)
        self.assertEqual(tb.scheduler.list_tasks()[0]["action"], "follow-up-nudges")

    def test_accepts_connection_check(self):
        tb = toolbox()
        msg = tb.run("schedule_task", {"name": "Morning", "kind": "daily",
                                       "at": "08:00", "action": "connection-check"})
        self.assertIn("Scheduled", msg)
        self.assertEqual(tb.scheduler.list_tasks()[0]["action"], "connection-check")

    def test_rejects_unknown_action(self):
        tb = toolbox()
        msg = tb.run("schedule_task", {"name": "Bogus", "kind": "daily",
                                       "at": "08:00", "action": "weather-check"})
        self.assertIn("Unknown check action", msg)
        self.assertIn("urgent-email-check", msg)
        self.assertEqual(tb.scheduler.list_tasks(), [])


class TestRunTaskNowDispatch(unittest.TestCase):
    def test_urgent_email_check_runs_its_own_action(self):
        tb = toolbox()
        job = tb.scheduler.add("Urgent watch", "daily", at="08:00",
                               action="urgent-email-check")
        old = (tools_mod.google_connected, tools_mod.gmail_search_structured,
               smart_announce.announce_urgent_emails)
        tools_mod.google_connected = lambda: True
        tools_mod.gmail_search_structured = lambda q, max_results=20: [{"subject": "x"}]
        smart_announce.announce_urgent_emails = lambda messages: "announced 1 urgent"
        try:
            text = tb._run_task_now(job["id"])
        finally:
            (tools_mod.google_connected, tools_mod.gmail_search_structured,
             smart_announce.announce_urgent_emails) = old
        self.assertIn("urgent-email-check: announced 1 urgent", text)
        self.assertIn("announced 1 urgent",
                      tb.scheduler.list_tasks()[0]["last_result"])

    def test_urgent_email_check_google_not_connected(self):
        tb = toolbox()
        job = tb.scheduler.add("Urgent watch", "daily", at="08:00",
                               action="urgent-email-check")
        old = tools_mod.google_connected
        tools_mod.google_connected = lambda: False
        try:
            text = tb._run_task_now(job["id"])
        finally:
            tools_mod.google_connected = old
        self.assertIn("urgent-email-check: Google not connected", text)

    def test_follow_up_nudges_runs_its_own_action(self):
        tb = toolbox()
        job = tb.scheduler.add("Nudge watch", "daily", at="08:00",
                               action="follow-up-nudges")
        old = (tools_mod.google_connected, tools_mod.gmail_unanswered_threads,
               smart_announce.announce_followup_nudges)
        tools_mod.google_connected = lambda: True
        tools_mod.gmail_unanswered_threads = lambda **kw: [
            {"subject": "Re: plans", "from": "a@example.com"}]
        smart_announce.announce_followup_nudges = lambda threads: "spoken 1 nudge"
        try:
            text = tb._run_task_now(job["id"])
        finally:
            (tools_mod.google_connected, tools_mod.gmail_unanswered_threads,
             smart_announce.announce_followup_nudges) = old
        self.assertIn("follow-up-nudges: 1 awaiting a reply", text)
        self.assertIn("spoken 1 nudge", text)

    def test_follow_up_nudges_nothing_awaiting(self):
        tb = toolbox()
        job = tb.scheduler.add("Nudge watch", "daily", at="08:00",
                               action="follow-up-nudges")
        old = (tools_mod.google_connected, tools_mod.gmail_unanswered_threads)
        tools_mod.google_connected = lambda: True
        tools_mod.gmail_unanswered_threads = lambda **kw: []
        try:
            text = tb._run_task_now(job["id"])
        finally:
            (tools_mod.google_connected, tools_mod.gmail_unanswered_threads) = old
        self.assertIn("follow-up-nudges: nothing awaiting a reply", text)

    def test_connection_check_job_still_uses_connection_check(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily", at="08:00",
                               action="connection-check")
        old = tools_mod.connection_check
        tools_mod.connection_check = lambda: {"gmail": "ok"}
        try:
            text = tb._run_task_now(job["id"])
        finally:
            tools_mod.connection_check = old
        self.assertIn("Ran 'Morning' now", text)
        self.assertIn("ok", text)


if __name__ == "__main__":
    unittest.main()
