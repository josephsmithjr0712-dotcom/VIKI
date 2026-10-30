"""Tests for the Spotify connector. spotipy is mocked."""
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.spotify as spm


def fake_track(name="Song", artist="Band"):
    return {"name": name, "uri": "spotify:track:abc",
            "artists": [{"name": artist}]}


def fake_sp():
    return MagicMock()


class TestSpotify(TestCase):
    def _sp(self, sp):
        return patch.object(spm, "_sp", return_value=sp)

    def test_search_tracks(self):
        sp = fake_sp()
        sp.search.return_value = {"tracks": {"items": [fake_track()]}}
        with self._sp(sp):
            out = spm.spotify_search("song")
        sp.search.assert_called_once_with("song", limit=5, type="track")
        self.assertIn("Song — Band", out)
        self.assertIn("spotify:track:abc", out)

    def test_search_bad_kind(self):
        with self._sp(fake_sp()):
            self.assertIn("track, album, artist, or playlist",
                          spm.spotify_search("x", kind="movie"))

    def test_play_picks_device(self):
        sp = fake_sp()
        sp.search.return_value = {"tracks": {"items": [fake_track()]}}
        sp.devices.return_value = {"devices": [
            {"id": "d1", "name": "Phone", "type": "Smartphone", "is_active": False},
            {"id": "d2", "name": "Kitchen Mini", "type": "Speaker", "is_active": True},
        ]}
        with self._sp(sp):
            out = spm.spotify_play("song", device="kitchen")
        sp.start_playback.assert_called_once_with(device_id="d2",
                                                  uris=["spotify:track:abc"])
        self.assertIn("Kitchen Mini", out)

    def test_play_defaults_to_active(self):
        sp = fake_sp()
        sp.search.return_value = {"tracks": {"items": [fake_track()]}}
        sp.devices.return_value = {"devices": [
            {"id": "d1", "name": "Phone", "type": "Smartphone", "is_active": True},
        ]}
        with self._sp(sp):
            spm.spotify_play("song")
        sp.start_playback.assert_called_once_with(device_id="d1",
                                                  uris=["spotify:track:abc"])

    def test_play_no_match(self):
        sp = fake_sp()
        sp.search.return_value = {"tracks": {"items": []}}
        with self._sp(sp):
            out = spm.spotify_play("zzz")
        self.assertIn("No Spotify tracks found", out)

    def test_play_uri_track_vs_context(self):
        sp = fake_sp()
        sp.devices.return_value = {"devices": [
            {"id": "d1", "name": "Phone", "type": "Smartphone", "is_active": True}]}
        with self._sp(sp):
            spm.spotify_play_uri("spotify:track:abc")
        sp.start_playback.assert_called_with(device_id="d1", uris=["spotify:track:abc"])
        with self._sp(sp):
            spm.spotify_play_uri("spotify:playlist:xyz")
        sp.start_playback.assert_called_with(device_id="d1",
                                             context_uri="spotify:playlist:xyz")

    def test_play_uri_bad(self):
        with self._sp(fake_sp()):
            self.assertIn("Spotify URI", spm.spotify_play_uri("http://x"))

    def test_controls(self):
        sp = fake_sp()
        with self._sp(sp):
            self.assertEqual(spm.spotify_pause(), "Paused.")
            self.assertEqual(spm.spotify_resume(), "Resumed.")
            self.assertEqual(spm.spotify_next(), "Skipped.")
            self.assertEqual(spm.spotify_previous(), "Went back.")
        sp.pause_playback.assert_called_once()
        sp.next_track.assert_called_once()
        sp.previous_track.assert_called_once()

    def test_devices(self):
        sp = fake_sp()
        sp.devices.return_value = {"devices": [
            {"id": "d1", "name": "Phone", "type": "Smartphone", "is_active": True}]}
        with self._sp(sp):
            out = spm.spotify_devices()
        self.assertIn("Phone (Smartphone) [active]", out)

    def test_now_playing(self):
        sp = fake_sp()
        sp.current_playback.return_value = {
            "item": fake_track("Song", "Band"),
            "is_playing": True, "device": {"name": "Kitchen Mini"}}
        with self._sp(sp):
            out = spm.spotify_now_playing()
        self.assertIn("Song — Band", out)
        self.assertIn("playing on Kitchen Mini", out)

    def test_now_playing_nothing(self):
        sp = fake_sp()
        sp.current_playback.return_value = None
        with self._sp(sp):
            self.assertEqual(spm.spotify_now_playing(), "Nothing playing.")

    def test_volume(self):
        sp = fake_sp()
        sp.devices.return_value = {"devices": [
            {"id": "d1", "name": "Phone", "type": "Smartphone", "is_active": True}]}
        with self._sp(sp):
            out = spm.spotify_volume("75")
        sp.volume.assert_called_once_with(75, device_id="d1")
        self.assertIn("75%", out)

    def test_volume_clamped(self):
        sp = fake_sp()
        sp.devices.return_value = {"devices": [
            {"id": "d1", "name": "Phone", "type": "Smartphone", "is_active": True}]}
        with self._sp(sp):
            spm.spotify_volume("150")
        sp.volume.assert_called_once_with(100, device_id="d1")

    def test_volume_named_device(self):
        # Finding 30, second half: optional device name, matched the
        # same substring way as spotify_play.
        sp = fake_sp()
        sp.devices.return_value = {"devices": [
            {"id": "d1", "name": "Phone", "type": "Smartphone", "is_active": True},
            {"id": "d2", "name": "Living Room TV", "type": "TV", "is_active": False}]}
        with self._sp(sp):
            out = spm.spotify_volume("40", device="living room")
        sp.volume.assert_called_once_with(40, device_id="d2")
        self.assertIn("Living Room TV", out)

    def test_no_active_device_message(self):
        sp = fake_sp()
        sp.pause_playback.side_effect = Exception("NO_ACTIVE_DEVICE")
        with self._sp(sp):
            out = spm.spotify_pause()
        self.assertIn("No active Spotify device", out)

    def test_toolbox_gating(self):
        from tools import Toolbox, SPOTIFY_TOOLS
        from memory import SessionStore
        import tools as tools_mod

        with patch.object(tools_mod, "spotify_connected", return_value=False):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        for t in SPOTIFY_TOOLS:
            self.assertNotIn(t, tb.enabled)
        with patch.object(tools_mod, "spotify_connected", return_value=True):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s2"))
        for t in SPOTIFY_TOOLS:
            self.assertIn(t, tb.enabled)
        self.assertFalse(tb.needs_approval("spotify_play", {"query": "x"}))


if __name__ == "__main__":
    main(verbosity=2)
