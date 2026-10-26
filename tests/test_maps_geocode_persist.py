"""Tests for the disk-backed geocode cache in connectors/maps.py (v0.2.36).

All HTTP is mocked — no real network calls. Covers: a lookup is written
to disk, a fresh process (simulated) serves it from disk with no HTTP
call, corrupt files fail soft, malformed entries are skipped, and
clear_geocode_cache removes the file.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.maps as maps  # noqa: E402


def _row(lat, lon, name):
    return [{"lat": str(lat), "lon": str(lon), "display_name": name}]


class GeocodePersistTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cache_file = Path(self._tmp.name) / "geocode_cache.json"
        self._orig_file = maps._CACHE_FILE
        maps._CACHE_FILE = self._cache_file
        maps.clear_geocode_cache()

    def tearDown(self):
        maps._CACHE_FILE = self._orig_file
        maps.clear_geocode_cache()
        self._tmp.cleanup()

    def _simulate_restart(self):
        """Wipe in-memory state but keep the disk file, like a fresh process."""
        maps._GEOCODE_CACHE.clear()
        maps._CACHE_LOADED = False

    def test_lookup_written_to_disk(self):
        with patch.object(
            maps, "_http_get_json", return_value=_row(40.5, -81.9, "Millersburg")
        ):
            maps.geocode("Millersburg")
        raw = json.loads(self._cache_file.read_text())
        self.assertEqual(raw["millersburg"], [40.5, -81.9, "Millersburg"])

    def test_restart_serves_from_disk_without_http(self):
        with patch.object(
            maps, "_http_get_json", return_value=_row(40.5, -81.9, "Millersburg")
        ):
            maps.geocode("Millersburg")
        self._simulate_restart()
        with patch.object(
            maps, "_http_get_json", side_effect=AssertionError("must not hit network")
        ) as get:
            result, err = maps.geocode("Millersburg")
        self.assertIsNone(err)
        self.assertEqual(result, (40.5, -81.9, "Millersburg"))
        get.assert_not_called()

    def test_corrupt_file_fails_soft(self):
        self._cache_file.write_text("{not json")
        self._simulate_restart()
        with patch.object(
            maps, "_http_get_json", return_value=_row(40.5, -81.9, "Millersburg")
        ) as get:
            result, err = maps.geocode("Millersburg")
        self.assertIsNone(err)
        self.assertEqual(result, (40.5, -81.9, "Millersburg"))
        self.assertEqual(get.call_count, 1)

    def test_malformed_entries_skipped(self):
        self._cache_file.write_text(
            json.dumps(
                {
                    "good": [40.5, -81.9, "Millersburg"],
                    "bad": "nope",
                    "alsobad": [1],
                }
            )
        )
        self._simulate_restart()
        with patch.object(
            maps, "_http_get_json", return_value=_row(40.5, -81.9, "Millersburg")
        ):
            result, err = maps.geocode("good")
        self.assertIsNone(err)
        self.assertEqual(result, (40.5, -81.9, "Millersburg"))
        # "bad" was skipped on load, so it falls through to a live lookup.
        with patch.object(
            maps, "_http_get_json", return_value=_row(41.0, -82.0, "Bad Town")
        ) as get:
            result, err = maps.geocode("bad")
        self.assertIsNone(err)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(result, (41.0, -82.0, "Bad Town"))

    def test_clear_removes_file(self):
        with patch.object(
            maps, "_http_get_json", return_value=_row(40.5, -81.9, "Millersburg")
        ):
            maps.geocode("Millersburg")
        self.assertTrue(self._cache_file.exists())
        maps.clear_geocode_cache()
        self.assertFalse(self._cache_file.exists())


if __name__ == "__main__":
    unittest.main()
