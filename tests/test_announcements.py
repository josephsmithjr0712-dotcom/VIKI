"""Tests for the smart-announcement guard (v0.2.24).

Pure logic + a temp JSON file: no network, no Home Assistant, no TTS.
"""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from announcements import AnnouncementLog, fingerprint  # noqa: E402


def _log(tmp):
    return AnnouncementLog(Path(tmp) / "announced.json")


class TestFingerprint(unittest.TestCase):
    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(
            fingerprint("Urgent mail from Amy"),
            fingerprint("  urgent   MAIL from amy "),
        )

    def test_different_messages_differ(self):
        self.assertNotEqual(
            fingerprint("Urgent mail from Amy"),
            fingerprint("Urgent mail from Bob"),
        )


class TestDedupe(unittest.TestCase):
    def test_repeat_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _log(tmp)
            self.assertEqual(log.gate("Urgent mail landed"), (True, "ok"))
            log.mark_announced("Urgent mail landed")
            self.assertEqual(log.gate("urgent mail landed"), (False, "already announced"))

    def test_new_message_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _log(tmp)
            log.mark_announced("Urgent mail landed")
            self.assertEqual(log.gate("Dinner is ready"), (True, "ok"))

    def test_record_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            _log(tmp).mark_announced("Urgent mail landed")
            reloaded = _log(tmp)
            self.assertTrue(reloaded.already_announced("Urgent mail landed"))


class TestWelcomeHome(unittest.TestCase):
    def _friday_4pm(self):
        # 2026-10-23 was a Friday.
        return datetime(2026, 10, 23, 16, 0)

    def test_happy_path_then_once_per_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _log(tmp)
            now = self._friday_4pm()
            self.assertTrue(log.welcome_home_due(now, True, True))
            log.mark_welcome_home(now)
            self.assertFalse(log.welcome_home_due(now, True, True))

    def test_next_day_due_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _log(tmp)
            log.mark_welcome_home(self._friday_4pm())
            # Monday 2026-10-26, 4pm.
            self.assertTrue(
                log.welcome_home_due(datetime(2026, 10, 26, 16, 0), True, True)
            )

    def test_before_3pm_not_due(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _log(tmp)
            self.assertFalse(
                log.welcome_home_due(datetime(2026, 10, 23, 14, 59), True, True)
            )

    def test_weekend_not_due(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _log(tmp)
            # Saturday 2026-10-24, 4pm.
            self.assertFalse(
                log.welcome_home_due(datetime(2026, 10, 24, 16, 0), True, True)
            )

    def test_phone_away_or_lights_off_not_due(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _log(tmp)
            now = self._friday_4pm()
            self.assertFalse(log.welcome_home_due(now, False, True))
            self.assertFalse(log.welcome_home_due(now, True, False))


if __name__ == "__main__":
    unittest.main()
