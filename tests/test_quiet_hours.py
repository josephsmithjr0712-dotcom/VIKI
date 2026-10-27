"""Tests for quiet-hours gating of smart announcements (v0.3.14).

No network, no Home Assistant, no real speaker: injected states,
a fake speaker, and a temp announcement log, mirroring
test_smart_announce.py. Quiet hours default to 10pm–7am and are
overridable with VIKI_QUIET_HOURS.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from announcements import AnnouncementLog  # noqa: E402
from smart_announce import (  # noqa: E402
    WELCOME_HOME_TEXT,
    _quiet_hours,
    announce_followup_nudges,
    announce_if_home,
    announce_urgent_emails,
    is_quiet_hours,
    run_smart_job,
    welcome_home,
)

# Monday 2026-10-26: daytime, night, and the window edges.
DAY = datetime(2026, 10, 26, 16, 0)
NIGHT = datetime(2026, 10, 26, 23, 30)
EARLY = datetime(2026, 10, 26, 6, 30)
AT_7AM = datetime(2026, 10, 26, 7, 0)
AT_10PM = datetime(2026, 10, 26, 22, 0)
BEFORE_10PM = datetime(2026, 10, 26, 21, 59)


def _tracker(state="home"):
    return {"entity_id": "device_tracker.joeys_phone", "state": state,
            "attributes": {"friendly_name": "Joeys Phone"}}


def _light(state="on"):
    return {"entity_id": "light.living_room", "state": state,
            "attributes": {"friendly_name": "Living Room"}}


def _home_states():
    return [_tracker("home"), _light("on")]


class _Speaker:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return "Announced on test speaker."


class QuietHoursTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log = AnnouncementLog(Path(self._tmp.name) / "announced.json")

    def tearDown(self):
        self._tmp.cleanup()

    # -- window parsing ----------------------------------------------------
    def test_default_window_is_10pm_to_7am(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VIKI_QUIET_HOURS", None)
            self.assertEqual(_quiet_hours(), (22, 7))

    def test_custom_window_parsed(self):
        with mock.patch.dict(os.environ, {"VIKI_QUIET_HOURS": "21-6"}):
            self.assertEqual(_quiet_hours(), (21, 6))

    def test_off_disables(self):
        with mock.patch.dict(os.environ, {"VIKI_QUIET_HOURS": "off"}):
            self.assertIsNone(_quiet_hours())
            self.assertFalse(is_quiet_hours(now=NIGHT))

    def test_malformed_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"VIKI_QUIET_HOURS": "banana"}):
            self.assertEqual(_quiet_hours(), (22, 7))

    def test_out_of_range_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"VIKI_QUIET_HOURS": "25-7"}):
            self.assertEqual(_quiet_hours(), (22, 7))

    def test_equal_start_end_means_empty_window(self):
        with mock.patch.dict(os.environ, {"VIKI_QUIET_HOURS": "8-8"}):
            self.assertIsNone(_quiet_hours())

    # -- window membership ---------------------------------------------------
    def test_wraps_midnight(self):
        self.assertTrue(is_quiet_hours(now=NIGHT))
        self.assertTrue(is_quiet_hours(now=EARLY))
        self.assertFalse(is_quiet_hours(now=DAY))

    def test_edges_are_half_open(self):
        self.assertTrue(is_quiet_hours(now=AT_10PM))
        self.assertFalse(is_quiet_hours(now=BEFORE_10PM))
        self.assertFalse(is_quiet_hours(now=AT_7AM))

    def test_non_wrapping_window(self):
        with mock.patch.dict(os.environ, {"VIKI_QUIET_HOURS": "1-3"}):
            self.assertTrue(is_quiet_hours(now=datetime(2026, 10, 26, 2, 0)))
            self.assertFalse(is_quiet_hours(now=DAY))

    # -- announce gating -------------------------------------------------------
    def test_night_announcement_skipped_not_spoken_not_recorded(self):
        speaker = _Speaker()
        ok, reason = announce_if_home("Urgent mail.", announce_fn=speaker,
                                     log=self.log, states=_home_states(),
                                     now=NIGHT)
        self.assertFalse(ok)
        self.assertEqual(reason, "quiet hours")
        self.assertEqual(speaker.calls, [])
        self.assertFalse(self.log.already_announced("Urgent mail."))
        # Same message speaks fine in the daytime: nothing was burned.
        ok, _ = announce_if_home("Urgent mail.", announce_fn=speaker,
                                 log=self.log, states=_home_states(),
                                 now=DAY)
        self.assertTrue(ok)
        self.assertEqual(speaker.calls, ["Urgent mail."])

    def test_presence_reasons_still_win_at_night(self):
        # Quiet hours sit behind the presence gate, so the reason for a
        # skip stays honest when Joey is away or the lights are off.
        speaker = _Speaker()
        states = [_tracker("not_home"), _light("on")]
        _, reason = announce_if_home("Hi.", announce_fn=speaker,
                                     log=self.log, states=states, now=NIGHT)
        self.assertEqual(reason, "phone not home")

    def test_welcome_home_not_burned_by_quiet_hours(self):
        speaker = _Speaker()
        ok, reason = welcome_home(announce_fn=speaker, log=self.log,
                                  now=datetime(2026, 10, 26, 23, 30),
                                  states=_home_states())
        # 23:30 is a weekday after 3pm, so it is due — but quiet.
        self.assertFalse(ok)
        self.assertEqual(reason, "quiet hours")
        self.assertEqual(speaker.calls, [])
        # The day's greeting is still available (not marked).
        ok, _ = welcome_home(announce_fn=speaker, log=self.log,
                             now=DAY, states=_home_states())
        self.assertTrue(ok)
        self.assertEqual(speaker.calls, [WELCOME_HOME_TEXT])

    def test_urgent_and_nudges_report_quiet_hours(self):
        speaker = _Speaker()
        messages = [{"from": "boss@example.com",
                     "subject": "URGENT: server down", "snippet": ""}]
        out = announce_urgent_emails(messages, announce_fn=speaker,
                                     log=self.log, states=_home_states(),
                                     now=NIGHT)
        self.assertIn("0 announced", out)
        self.assertIn("quiet hours", out)
        self.assertEqual(speaker.calls, [])
        out = announce_followup_nudges(
            [{"from": "a@b.com", "subject": "Hi"}], announce_fn=speaker,
            log=self.log, states=_home_states(), now=NIGHT)
        self.assertIn("quiet hours", out)

    def test_run_smart_job_reports_quiet_hours(self):
        speaker = _Speaker()
        out = run_smart_job({"action": "announce-if-home:Doorbell."},
                            announce_fn=speaker, log=self.log,
                            now=NIGHT, states=_home_states())
        self.assertIn("quiet hours", out)
        self.assertEqual(speaker.calls, [])


if __name__ == "__main__":
    unittest.main()
