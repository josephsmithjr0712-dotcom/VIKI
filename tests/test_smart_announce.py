"""Tests for presence-gated smart announcements (v0.2.31).

Pure logic + injected states / fake speaker / temp log: no network,
no Home Assistant, no real speaker.
"""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from announcements import AnnouncementLog  # noqa: E402
from smart_announce import (  # noqa: E402
    WELCOME_HOME_TEXT,
    announce_if_home,
    run_smart_job,
    welcome_home,
)

# Monday 2026-10-26 (weekday), Saturday 2026-10-31 (weekend).
MON_AFTER_3PM = datetime(2026, 10, 26, 16, 0)
MON_BEFORE_3PM = datetime(2026, 10, 26, 14, 0)
SAT_AFTER_3PM = datetime(2026, 10, 31, 16, 0)


def _tracker(state="home"):
    return {"entity_id": "device_tracker.joeys_phone", "state": state,
            "attributes": {"friendly_name": "Joeys Phone"}}


def _light(state="on"):
    return {"entity_id": "light.living_room", "state": state,
            "attributes": {"friendly_name": "Living Room"}}


def _home_states():
    return [_tracker("home"), _light("on")]


class _Speaker:
    """Fake announce_fn: records calls, returns success or failure."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, text):
        self.calls.append(text)
        if self.fail:
            return "Speaker unreachable."
        return "Announced on test speaker."


class SmartAnnounceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log = AnnouncementLog(Path(self._tmp.name) / "announced.json")

    def tearDown(self):
        self._tmp.cleanup()

    # -- announce_if_home -------------------------------------------------
    def test_speaks_when_home_and_fresh(self):
        speaker = _Speaker()
        ok, _ = announce_if_home("Urgent mail from HR.", announce_fn=speaker,
                                 log=self.log, states=_home_states())
        self.assertTrue(ok)
        self.assertEqual(speaker.calls, ["Urgent mail from HR."])
        self.assertTrue(self.log.already_announced("Urgent mail from HR."))

    def test_skips_when_phone_away(self):
        speaker = _Speaker()
        states = [_tracker("not_home"), _light("on")]
        ok, reason = announce_if_home("Hi.", announce_fn=speaker,
                                      log=self.log, states=states)
        self.assertFalse(ok)
        self.assertEqual(reason, "phone not home")
        self.assertEqual(speaker.calls, [])

    def test_skips_when_lights_off(self):
        speaker = _Speaker()
        states = [_tracker("home"), _light("off")]
        ok, reason = announce_if_home("Hi.", announce_fn=speaker,
                                      log=self.log, states=states)
        self.assertFalse(ok)
        self.assertEqual(reason, "lights off")
        self.assertEqual(speaker.calls, [])

    def test_never_repeats(self):
        speaker = _Speaker()
        announce_if_home("Same news.", announce_fn=speaker,
                         log=self.log, states=_home_states())
        ok, reason = announce_if_home("Same news.", announce_fn=speaker,
                                      log=self.log, states=_home_states())
        self.assertFalse(ok)
        self.assertEqual(reason, "already announced")
        self.assertEqual(len(speaker.calls), 1)

    def test_failed_announce_is_not_recorded(self):
        failing = _Speaker(fail=True)
        ok, _ = announce_if_home("Try me.", announce_fn=failing,
                                 log=self.log, states=_home_states())
        self.assertFalse(ok)
        self.assertFalse(self.log.already_announced("Try me."))
        # Retry with a working speaker succeeds.
        working = _Speaker()
        ok, _ = announce_if_home("Try me.", announce_fn=working,
                                 log=self.log, states=_home_states())
        self.assertTrue(ok)

    def test_empty_text_never_speaks(self):
        speaker = _Speaker()
        ok, reason = announce_if_home("   ", announce_fn=speaker,
                                      log=self.log, states=_home_states())
        self.assertFalse(ok)
        self.assertEqual(reason, "nothing to announce")
        self.assertEqual(speaker.calls, [])

    # -- welcome_home ------------------------------------------------------
    def test_welcome_home_speaks_once_per_day(self):
        speaker = _Speaker()
        ok, _ = welcome_home(announce_fn=speaker, log=self.log,
                             now=MON_AFTER_3PM, states=_home_states())
        self.assertTrue(ok)
        self.assertEqual(speaker.calls, [WELCOME_HOME_TEXT])
        # Second check the same day: no repeat.
        ok, reason = welcome_home(announce_fn=speaker, log=self.log,
                                  now=MON_AFTER_3PM, states=_home_states())
        self.assertFalse(ok)
        self.assertEqual(reason, "already greeted today")
        self.assertEqual(len(speaker.calls), 1)

    def test_welcome_home_before_3pm_skipped(self):
        speaker = _Speaker()
        ok, reason = welcome_home(announce_fn=speaker, log=self.log,
                                  now=MON_BEFORE_3PM, states=_home_states())
        self.assertFalse(ok)
        self.assertEqual(reason, "before 3pm")
        self.assertEqual(speaker.calls, [])

    def test_welcome_home_weekend_skipped(self):
        speaker = _Speaker()
        ok, reason = welcome_home(announce_fn=speaker, log=self.log,
                                  now=SAT_AFTER_3PM, states=_home_states())
        self.assertFalse(ok)
        self.assertEqual(reason, "weekend: no greeting")
        self.assertEqual(speaker.calls, [])

    def test_failed_greeting_does_not_burn_the_day(self):
        failing = _Speaker(fail=True)
        ok, _ = welcome_home(announce_fn=failing, log=self.log,
                             now=MON_AFTER_3PM, states=_home_states())
        self.assertFalse(ok)
        working = _Speaker()
        ok, _ = welcome_home(announce_fn=working, log=self.log,
                             now=MON_AFTER_3PM, states=_home_states())
        self.assertTrue(ok)
        self.assertEqual(working.calls, [WELCOME_HOME_TEXT])

    # -- run_smart_job ------------------------------------------------------
    def test_run_smart_job_dispatch(self):
        speaker = _Speaker()
        out = run_smart_job({"action": "welcome-home"}, announce_fn=speaker,
                            log=self.log, now=MON_AFTER_3PM,
                            states=_home_states())
        self.assertIn("spoken", out)
        out = run_smart_job({"action": "announce-if-home:Doorbell."},
                            announce_fn=speaker, log=self.log,
                            states=_home_states())
        self.assertIn("spoken", out)
        self.assertEqual(speaker.calls, [WELCOME_HOME_TEXT, "Doorbell."])
        out = run_smart_job({"action": "do-something-else"},
                            announce_fn=speaker, log=self.log,
                            states=_home_states())
        self.assertIn("unknown smart-announcement job", out)


if __name__ == "__main__":
    unittest.main()
