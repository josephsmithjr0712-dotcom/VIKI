"""Tests for queued one-off announcements (v0.3.15).

A scheduled one-off announcement skipped by a gate (quiet hours,
phone away, lights off) is queued in the announcement log and retried
on every scheduler tick until it speaks, is spoken another way, or
expires after 24 hours — so it is delayed, not lost.

Pure logic + injected states / fake speaker / temp log: no network,
no Home Assistant, no real speaker.
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from announcements import AnnouncementLog  # noqa: E402
from smart_announce import (  # noqa: E402
    retry_pending_announcements,
    run_smart_job,
)

# Monday 2026-10-26 (weekday), 4pm: home + after 3pm.
MON_4PM = datetime(2026, 10, 26, 16, 0)


def _tracker(state="home"):
    return {"entity_id": "device_tracker.joeys_phone", "state": state,
            "attributes": {"friendly_name": "Joeys Phone"}}


def _light(state="on"):
    return {"entity_id": "light.living_room", "state": state,
            "attributes": {"friendly_name": "Living Room"}}


def _home_states():
    return [_tracker("home"), _light("on")]


def _away_states():
    return [_tracker("not_home"), _light("on")]


class _Speaker:
    """Fake announce_fn: records calls, returns success."""

    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return "Announced on test speaker."


class PendingQueueTest(unittest.TestCase):
    def test_queue_and_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            self.assertTrue(log.queue_pending("Reminder: call Amy."))
            self.assertEqual(log.pending_texts(), ["Reminder: call Amy."])

    def test_duplicate_queue_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            self.assertTrue(log.queue_pending("Reminder: call Amy."))
            self.assertFalse(log.queue_pending("  reminder: CALL amy. "))
            self.assertEqual(len(log.pending_texts()), 1)

    def test_already_announced_is_not_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            log.mark_announced("Reminder: call Amy.")
            self.assertFalse(log.queue_pending("Reminder: call Amy."))
            self.assertEqual(log.pending_texts(), [])

    def test_empty_text_is_not_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            self.assertFalse(log.queue_pending("   "))
            self.assertEqual(log.pending_texts(), [])

    def test_drop_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            log.queue_pending("Reminder: call Amy.")
            self.assertTrue(log.drop_pending("Reminder: call Amy."))
            self.assertFalse(log.drop_pending("Reminder: call Amy."))
            self.assertEqual(log.pending_texts(), [])

    def test_pending_expires_after_24_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            # Relative to real now: _save() prunes on every queue, so
            # the base must be real now, not a fixed test date.
            now = datetime.now()
            log.queue_pending("Stale reminder.",
                              when=now - timedelta(hours=25))
            log.queue_pending("Fresh reminder.",
                              when=now - timedelta(hours=23))
            self.assertEqual(log.pending_texts(now=now),
                             ["Fresh reminder."])

    def test_queue_persists_across_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "announced.json"
            AnnouncementLog(path).queue_pending("Reminder: call Amy.")
            reloaded = AnnouncementLog(path)
            self.assertEqual(reloaded.pending_texts(),
                             ["Reminder: call Amy."])


class RetryPendingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log = AnnouncementLog(Path(self._tmp.name) / "announced.json")
        # Pin quiet hours off unless a test says otherwise.
        self._quiet_off = mock.patch(
            "smart_announce.is_quiet_hours", return_value=False)
        self._quiet_off.start()

    def tearDown(self):
        self._quiet_off.stop()
        self._tmp.cleanup()

    def test_speaks_pending_when_home_and_drops_it(self):
        speaker = _Speaker()
        self.log.queue_pending("Reminder: call Amy.")
        summary = retry_pending_announcements(
            announce_fn=speaker, log=self.log, states=_home_states(),
            now=MON_4PM)
        self.assertEqual(speaker.calls, ["Reminder: call Amy."])
        self.assertEqual(self.log.pending_texts(), [])
        self.assertTrue(self.log.already_announced("Reminder: call Amy."))
        self.assertIn("1 spoken", summary)
        self.assertEqual(
            retry_pending_announcements(
                announce_fn=speaker, log=self.log, states=_home_states(),
                now=MON_4PM),
            "no pending announcements")

    def test_leaves_pending_when_away(self):
        speaker = _Speaker()
        self.log.queue_pending("Reminder: call Amy.")
        summary = retry_pending_announcements(
            announce_fn=speaker, log=self.log, states=_away_states(),
            now=MON_4PM)
        self.assertEqual(speaker.calls, [])
        self.assertEqual(self.log.pending_texts(), ["Reminder: call Amy."])
        self.assertIn("still waiting", summary)

    def test_leaves_pending_during_quiet_hours(self):
        speaker = _Speaker()
        self.log.queue_pending("Reminder: call Amy.")
        with mock.patch("smart_announce.is_quiet_hours",
                        return_value=True):
            summary = retry_pending_announcements(
                announce_fn=speaker, log=self.log, states=_home_states(),
                now=MON_4PM)
        self.assertEqual(speaker.calls, [])
        self.assertEqual(self.log.pending_texts(), ["Reminder: call Amy."])
        self.assertIn("quiet hours", summary)

    def test_drops_pending_spoken_another_way(self):
        speaker = _Speaker()
        self.log.queue_pending("Reminder: call Amy.")
        self.log.mark_announced("Reminder: call Amy.")
        summary = retry_pending_announcements(
            announce_fn=speaker, log=self.log, states=_home_states(),
            now=MON_4PM)
        self.assertEqual(speaker.calls, [])
        self.assertEqual(self.log.pending_texts(), [])
        self.assertIn("0 spoken", summary)

    def test_empty_queue_reports_nothing_pending(self):
        self.assertEqual(
            retry_pending_announcements(
                announce_fn=_Speaker(), log=self.log,
                states=_home_states(), now=MON_4PM),
            "no pending announcements")


class RunSmartJobQueueTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log = AnnouncementLog(Path(self._tmp.name) / "announced.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_quiet_hours_skip_queues_for_later(self):
        speaker = _Speaker()
        with mock.patch("smart_announce.is_quiet_hours",
                        return_value=True):
            result = run_smart_job(
                {"action": "announce-if-home:Reminder: call Amy."},
                announce_fn=speaker, log=self.log, states=_home_states(),
                now=MON_4PM)
        self.assertEqual(speaker.calls, [])
        self.assertIn("queued for later", result)
        self.assertEqual(self.log.pending_texts(), ["Reminder: call Amy."])

    def test_spoken_job_is_not_queued(self):
        speaker = _Speaker()
        with mock.patch("smart_announce.is_quiet_hours",
                        return_value=False):
            result = run_smart_job(
                {"action": "announce-if-home:Reminder: call Amy."},
                announce_fn=speaker, log=self.log, states=_home_states(),
                now=MON_4PM)
        self.assertEqual(speaker.calls, ["Reminder: call Amy."])
        self.assertIn("spoken", result)
        self.assertEqual(self.log.pending_texts(), [])

    def test_already_announced_job_is_not_queued(self):
        speaker = _Speaker()
        self.log.mark_announced("Reminder: call Amy.")
        with mock.patch("smart_announce.is_quiet_hours",
                        return_value=False):
            result = run_smart_job(
                {"action": "announce-if-home:Reminder: call Amy."},
                announce_fn=speaker, log=self.log, states=_home_states(),
                now=MON_4PM)
        self.assertEqual(speaker.calls, [])
        self.assertIn("already announced", result)
        self.assertEqual(self.log.pending_texts(), [])


if __name__ == "__main__":
    unittest.main()
