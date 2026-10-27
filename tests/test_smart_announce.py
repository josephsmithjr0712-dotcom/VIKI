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
    announce_brief,
    announce_followup_nudges,
    announce_if_home,
    announce_urgent_emails,
    is_urgent,
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

    # -- announce_brief ----------------------------------------------------
    def test_brief_speaks_collapsed_when_home(self):
        speaker = _Speaker()
        brief = "Unread mail (last day):\nNo unread mail.\n\nCalendar (today):\nNo upcoming events."
        ok, _ = announce_brief(brief, announce_fn=speaker,
                               log=self.log, states=_home_states())
        self.assertTrue(ok)
        self.assertEqual(
            speaker.calls,
            ["Here's your brief. Unread mail (last day): No unread mail. "
             "Calendar (today): No upcoming events."])

    def test_brief_skips_when_lights_off(self):
        speaker = _Speaker()
        states = [_tracker("home"), _light("off")]
        ok, reason = announce_brief("Some brief.", announce_fn=speaker,
                                    log=self.log, states=states)
        self.assertFalse(ok)
        self.assertEqual(reason, "lights off")
        self.assertEqual(speaker.calls, [])

    def test_brief_empty_never_speaks(self):
        speaker = _Speaker()
        ok, reason = announce_brief("  \n ", announce_fn=speaker,
                                    log=self.log, states=_home_states())
        self.assertFalse(ok)
        self.assertEqual(reason, "nothing to announce")
        self.assertEqual(speaker.calls, [])

    def test_brief_never_repeats(self):
        speaker = _Speaker()
        ok, _ = announce_brief("Same brief.", announce_fn=speaker,
                               log=self.log, states=_home_states())
        self.assertTrue(ok)
        ok, reason = announce_brief("Same brief.", announce_fn=speaker,
                                    log=self.log, states=_home_states())
        self.assertFalse(ok)
        self.assertEqual(reason, "already announced")
        self.assertEqual(len(speaker.calls), 1)

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

    # -- urgent emails ------------------------------------------------------
    def test_is_urgent_keyword_in_subject(self):
        self.assertTrue(is_urgent({"from": "a@b.com", "subject": "URGENT: read this",
                                   "snippet": ""}))
        self.assertTrue(is_urgent({"from": "a@b.com", "subject": "Weekly digest",
                                   "snippet": "action required by Friday"}))

    def test_is_urgent_sender_match_is_substring(self):
        self.assertTrue(is_urgent({"from": "HR Team <hr@gerresheimer.com>",
                                   "subject": "Hello", "snippet": ""},
                                  senders=["gerresheimer.com"], keywords=[]))

    def test_is_urgent_rejects_plain_mail(self):
        self.assertFalse(is_urgent({"from": "news@shop.com", "subject": "Sale!",
                                    "snippet": "20% off today"}, keywords=["urgent"]))

    def test_announce_urgent_emails_speaks_only_urgent(self):
        speaker = _Speaker()
        messages = [
            {"from": "boss@example.com", "subject": "URGENT: server down",
             "snippet": ""},
            {"from": "news@shop.com", "subject": "Sale!", "snippet": ""},
        ]
        out = announce_urgent_emails(messages, announce_fn=speaker,
                                     log=self.log, states=_home_states())
        self.assertEqual(out, "urgent emails: 1 announced")
        self.assertEqual(len(speaker.calls), 1)
        self.assertIn("server down", speaker.calls[0])

    def test_announce_urgent_emails_never_repeats(self):
        speaker = _Speaker()
        messages = [{"from": "boss@example.com", "subject": "URGENT: server down",
                     "snippet": ""}]
        announce_urgent_emails(messages, announce_fn=speaker,
                               log=self.log, states=_home_states())
        out = announce_urgent_emails(messages, announce_fn=speaker,
                                     log=self.log, states=_home_states())
        self.assertIn("0 announced", out)
        self.assertIn("already announced", out)
        self.assertEqual(len(speaker.calls), 1)

    def test_announce_urgent_emails_empty(self):
        self.assertEqual(announce_urgent_emails([], announce_fn=_Speaker(),
                                                log=self.log,
                                                states=_home_states()),
                         "no urgent emails")

    def test_announce_followup_nudges_speaks_each_thread(self):
        speaker = _Speaker()
        threads = [
            {"from": "boss@example.com", "subject": "Q?"},
            {"from": "HR Team <hr@gerresheimer.com>", "subject": ""},
        ]
        out = announce_followup_nudges(threads, announce_fn=speaker,
                                       log=self.log, states=_home_states())
        self.assertEqual(out, "follow-up nudges: 2 announced")
        self.assertEqual(len(speaker.calls), 2)
        self.assertIn("boss@example.com", speaker.calls[0])
        self.assertIn("Q?", speaker.calls[0])
        self.assertIn("(no subject)", speaker.calls[1])

    def test_announce_followup_nudges_never_repeats(self):
        speaker = _Speaker()
        threads = [{"from": "a@b.com", "subject": "Hello"}]
        announce_followup_nudges(threads, announce_fn=speaker,
                                 log=self.log, states=_home_states())
        out = announce_followup_nudges(threads, announce_fn=speaker,
                                       log=self.log, states=_home_states())
        self.assertIn("0 announced", out)
        self.assertIn("already announced", out)
        self.assertEqual(len(speaker.calls), 1)

    def test_announce_followup_nudges_empty(self):
        self.assertEqual(announce_followup_nudges(
            [], announce_fn=_Speaker(), log=self.log,
            states=_home_states()), "no follow-up nudges")

    def test_announce_followup_nudges_skips_when_away(self):
        speaker = _Speaker()
        states = [_tracker("not_home"), _light("on")]
        out = announce_followup_nudges(
            [{"from": "a@b.com", "subject": "Hi"}], announce_fn=speaker,
            log=self.log, states=states)
        self.assertIn("0 announced", out)
        self.assertIn("phone not home", out)
        self.assertEqual(speaker.calls, [])


if __name__ == "__main__":
    unittest.main()
