"""Tests for presence detection (v0.2.29).

Pure logic + injected HA states / fake ping: no network, no Home
Assistant, no real phone.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from announcements import AnnouncementLog  # noqa: E402
from presence import home_status, lights_on, phone_home, welcome_home_check  # noqa: E402


def _tracker(state="home", entity="device_tracker.joeys_phone"):
    return {"entity_id": entity, "state": state,
            "attributes": {"friendly_name": "Joeys Phone"}}


def _light(state="on", entity="light.living_room"):
    return {"entity_id": entity, "state": state,
            "attributes": {"friendly_name": "Living Room"}}


class TestPhoneHome(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("VIKI_PHONE_TRACKER", "VIKI_PHONE_IP", "HA_URL", "HA_TOKEN")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_tracker_home(self):
        self.assertTrue(phone_home(states=[_tracker("home")]))

    def test_tracker_away(self):
        self.assertFalse(phone_home(states=[_tracker("not_home")]))

    def test_tracker_ignored_when_not_phone(self):
        # Another person's tracker must not count as Joey home.
        other = {"entity_id": "device_tracker.someone_else", "state": "home",
                 "attributes": {"friendly_name": "Someone Else"}}
        self.assertFalse(phone_home(states=[other]))

    def test_ping_fallback(self):
        self.assertTrue(phone_home(states=[], ping=lambda ip: ip == "100.1.2.3",
                                   phone_ip="100.1.2.3"))
        self.assertFalse(phone_home(states=[], ping=lambda ip: False,
                                    phone_ip="100.1.2.3"))

    def test_nothing_configured_is_not_home(self):
        self.assertFalse(phone_home(states=[]))

    def test_ping_error_is_not_home(self):
        def boom(ip):
            raise RuntimeError("no ping here")
        self.assertFalse(phone_home(states=[], ping=boom, phone_ip="100.1.2.3"))


class TestLightsOn(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("LIGHT_ENTITY")
        os.environ.pop("LIGHT_ENTITY", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("LIGHT_ENTITY", None)
        else:
            os.environ["LIGHT_ENTITY"] = self._saved

    def test_any_light_on(self):
        self.assertTrue(lights_on(states=[_light("off"), _light("on")]))

    def test_all_off(self):
        self.assertFalse(lights_on(states=[_light("off")]))

    def test_no_lights(self):
        self.assertFalse(lights_on(states=[]))

    def test_pinned_lamp(self):
        os.environ["LIGHT_ENTITY"] = "light.kitchen"
        self.assertFalse(lights_on(states=[_light("on", "light.living_room")]))
        self.assertTrue(lights_on(states=[_light("on", "light.kitchen")]))


class TestHomeStatus(unittest.TestCase):
    def test_ready_only_when_both(self):
        home = [_tracker("home"), _light("on")]
        self.assertEqual(home_status(states=home),
                         {"phone_home": True, "lights_on": True, "ready": True})
        self.assertFalse(home_status(states=[_tracker("home"), _light("off")])["ready"])
        self.assertFalse(home_status(states=[_tracker("not_home"), _light("on")])["ready"])


class TestWelcomeHomeCheck(unittest.TestCase):
    def test_weekday_after_3pm_home(self):
        # Monday 2026-10-26, 4pm: all gates pass.
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            ok, reason = welcome_home_check(
                now=datetime(2026, 10, 26, 16, 0), log=log,
                states=[_tracker("home"), _light("on")])
            self.assertTrue(ok, reason)

    def test_weekend_blocked(self):
        # Saturday 2026-10-24, 4pm.
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            ok, reason = welcome_home_check(
                now=datetime(2026, 10, 24, 16, 0), log=log,
                states=[_tracker("home"), _light("on")])
            self.assertFalse(ok)
            self.assertEqual(reason, "weekend: no greeting")

    def test_before_3pm_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            ok, reason = welcome_home_check(
                now=datetime(2026, 10, 26, 14, 59), log=log,
                states=[_tracker("home"), _light("on")])
            self.assertFalse(ok)
            self.assertEqual(reason, "before 3pm")

    def test_lights_off_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            ok, reason = welcome_home_check(
                now=datetime(2026, 10, 26, 16, 0), log=log,
                states=[_tracker("home"), _light("off")])
            self.assertFalse(ok)
            self.assertEqual(reason, "lights off")

    def test_phone_away_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            ok, reason = welcome_home_check(
                now=datetime(2026, 10, 26, 16, 0), log=log,
                states=[_tracker("not_home"), _light("on")])
            self.assertFalse(ok)
            self.assertEqual(reason, "phone not home")

    def test_already_greeted_today_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AnnouncementLog(Path(tmp) / "announced.json")
            log.mark_welcome_home(when=datetime(2026, 10, 26, 16, 30))
            ok, reason = welcome_home_check(
                now=datetime(2026, 10, 26, 17, 0), log=log,
                states=[_tracker("home"), _light("on")])
            self.assertFalse(ok)
            self.assertEqual(reason, "already greeted today")


if __name__ == "__main__":
    unittest.main()
