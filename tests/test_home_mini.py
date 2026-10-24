"""Tests for Home Mini announcements via Home Assistant.
HA API and Edge TTS are mocked; the HTTP file server is exercised for real."""
import sys
import tempfile
import urllib.request
from functools import partial
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.home_mini as hm
from announcements import AnnouncementLog

FAKE_MP3 = b"\xff\xfb\x90\x00" * 100  # fake MP3 frame bytes

FAKE_STATES = [
    {"entity_id": "media_player.living_room_mini",
     "attributes": {"friendly_name": "Living Room Mini"}},
    {"entity_id": "media_player.kitchen_display",
     "attributes": {"friendly_name": "Kitchen Display"}},
]


class TestAnnounceHA(TestCase):
    def setUp(self):
        # announce() now dedupes via AnnouncementLog by default; point
        # that log at a fresh temp dir per test so tests stay hermetic
        # (a spoken message in one test must not skip the next test).
        self._tmpdir = tempfile.mkdtemp()
        self.patches = [
            patch.object(hm, "_ha_states", return_value=FAKE_STATES),
            patch.object(hm, "_ha_service"),
            patch.object(hm, "_lan_ip", return_value="127.0.0.1"),
            patch("connectors.edge_tts.speak", return_value=FAKE_MP3),
            patch.object(hm, "AnnouncementLog",
                         partial(AnnouncementLog,
                                 path=Path(self._tmpdir) / "announced.json")),
        ]
        self.mock_states = self.patches[0].start()
        self.mock_service = self.patches[1].start()
        for p in self.patches[2:]:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def test_happy_path_ha(self):
        out = hm.announce("Dinner is ready")
        self.assertIn("via Home Assistant", out)
        self.mock_service.assert_called_once()
        domain, service, data = self.mock_service.call_args[0]
        self.assertEqual((domain, service), ("media_player", "play_media"))
        self.assertEqual(data["entity_id"], "media_player.living_room_mini")
        self.assertIn("http://127.0.0.1:", data["media_content_id"])
        self.assertTrue(data["media_content_id"].endswith(".mp3"))

    def test_device_matching(self):
        out = hm.announce("hello", device="kitchen")
        self.assertIn("via Home Assistant", out)
        data = self.mock_service.call_args[0][2]
        self.assertEqual(data["entity_id"], "media_player.kitchen_display")

    def test_device_no_match_falls_back(self):
        # No HA match -> find_mini_entity returns None -> HA raises ->
        # falls back to Cast, which is also mocked to fail gracefully
        with patch.object(hm, "_announce_cast", return_value="cast fallback ok"):
            out = hm.announce("hello", device="garage")
        self.assertEqual(out, "cast fallback ok")

    def test_no_speakers_in_ha(self):
        self.mock_states.return_value = []
        with patch.object(hm, "_announce_cast", return_value="cast fallback ok"):
            out = hm.announce("hello")
        self.assertEqual(out, "cast fallback ok")

    def test_mini_entity_override(self):
        with patch.dict("os.environ", {"MINI_ENTITY": "media_player.custom_mini"}):
            out = hm.announce("hello")
        data = self.mock_service.call_args[0][2]
        self.assertEqual(data["entity_id"], "media_player.custom_mini")

    def test_empty_and_long(self):
        self.assertEqual(hm.announce("   "), "Nothing to announce.")
        self.assertIn("too long", hm.announce("x" * 501))

    def test_wait_until_idle_unconfigured_never_raises(self):
        # Contract: "Never raises" — must hold even when HA_URL/HA_TOKEN
        # are unset (regression: used to propagate _ha()'s RuntimeError).
        with patch.dict("os.environ", {"HA_URL": "", "HA_TOKEN": ""}):
            hm._wait_until_idle("media_player.test_mini", timeout=1)  # must not raise

    def test_wait_until_idle_polls_ha(self):
        # Mini is playing, then goes idle -> _wait_until_idle returns
        states = [{"state": "playing"}, {"state": "playing"}, {"state": "idle"}]
        with patch.object(hm, "_ha", return_value=("http://ha:8123", "tok")), \
             patch.object(hm.time, "sleep"), \
             patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = MagicMock()
            import json as json_mod
            mock_urlopen.return_value.__enter__.return_value.read.return_value = b""
            # json.load reads from the response object
            with patch("json.load", side_effect=lambda r: states.pop(0)):
                hm._wait_until_idle("media_player.test_mini", timeout=10)
            self.assertEqual(len(states), 0)  # consumed all three polls

    def test_http_server_serves_file(self):
        port = hm._ensure_server()
        p = hm.ANNOUNCE_DIR / "probe.txt"
        p.write_text("served-ok")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/probe.txt", timeout=5) as r:
                self.assertEqual(r.read().decode(), "served-ok")
        finally:
            p.unlink(missing_ok=True)

    def test_toolbox_wiring(self):
        from tools import Toolbox
        from memory import SessionStore
        import tempfile

        tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        self.assertIn("announce", tb.enabled)
        self.assertIn("cast_devices", tb.enabled)
        self.assertFalse(tb.needs_approval("announce", {"text": "hi"}))


class TestAnnounceDedupe(TestCase):
    """The v0.2.26 wiring: announce() itself enforces the no-repeat guard."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.log = AnnouncementLog(path=self.tmpdir / "announced.json")
        self.patches = [
            patch.object(hm, "_ha_states", return_value=FAKE_STATES),
            patch.object(hm, "_ha_service"),
            patch.object(hm, "_lan_ip", return_value="127.0.0.1"),
            patch("connectors.edge_tts.speak", return_value=FAKE_MP3),
        ]
        self.mock_service = self.patches[1].start()
        for p in (self.patches[0], self.patches[2], self.patches[3]):
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def test_repeat_is_skipped(self):
        out1 = hm.announce("Dinner is ready", log=self.log)
        self.assertIn("via Home Assistant", out1)
        out2 = hm.announce("Dinner is ready", log=self.log)
        self.assertIn("Already announced", out2)
        self.assertEqual(self.mock_service.call_count, 1)

    def test_rewording_is_skipped(self):
        hm.announce("Dinner is ready!", log=self.log)
        out = hm.announce("  dinner IS ready ", log=self.log)
        self.assertIn("Already announced", out)
        self.assertEqual(self.mock_service.call_count, 1)

    def test_dedupe_off_repeats(self):
        hm.announce("Dinner is ready", log=self.log, dedupe=False)
        out = hm.announce("Dinner is ready", log=self.log, dedupe=False)
        self.assertIn("via Home Assistant", out)
        self.assertEqual(self.mock_service.call_count, 2)

    def test_failed_announce_is_not_recorded(self):
        # No HA speakers, Cast reports failure: nothing is marked, so a
        # retry actually retries instead of being swallowed by the guard.
        with patch.object(hm, "_ha_states", return_value=[]), \
                patch.object(hm, "_announce_cast",
                             return_value="No Cast devices found.") as mock_cast:
            hm.announce("Hello?", log=self.log)
            hm.announce("Hello?", log=self.log)
            self.assertEqual(mock_cast.call_count, 2)
        self.assertFalse(self.log.already_announced("Hello?"))


if __name__ == "__main__":
    main(verbosity=2)
