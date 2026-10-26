"""Tests for the geocode cache in connectors/maps.py (v0.2.35).

All HTTP is mocked — no real network calls. Covers: repeat lookups hit
the cache (one HTTP call), different places are looked up separately,
failures are never cached, and the cache stays bounded.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.maps as maps  # noqa: E402


def _row(lat, lon, name):
    return [{"lat": str(lat), "lon": str(lon), "display_name": name}]


class GeocodeCacheTests(unittest.TestCase):
    def setUp(self):
        maps.clear_geocode_cache()

    def tearDown(self):
        maps.clear_geocode_cache()

    def test_repeat_lookup_uses_cache(self):
        with patch.object(
            maps, "_http_get_json", return_value=_row(40.5, -81.9, "Millersburg")
        ) as get:
            first, err1 = maps.geocode("Millersburg")
            second, err2 = maps.geocode("millersburg")  # case-insensitive key
        self.assertIsNone(err1)
        self.assertIsNone(err2)
        self.assertEqual(first, second)
        self.assertEqual(get.call_count, 1)

    def test_different_places_lookup_separately(self):
        def fake(url, timeout=12):
            if "Killbuck" in url:
                return _row(40.5, -81.98, "Killbuck")
            return _row(40.5, -81.9, "Millersburg")

        with patch.object(maps, "_http_get_json", side_effect=fake) as get:
            maps.geocode("Millersburg")
            maps.geocode("Killbuck")
        self.assertEqual(get.call_count, 2)

    def test_failures_are_not_cached(self):
        with patch.object(
            maps, "_http_get_json", side_effect=Exception("boom")
        ) as get:
            result, err = maps.geocode("Millersburg")
            self.assertIsNone(result)
            self.assertIn("Could not look up", err)
        with patch.object(
            maps, "_http_get_json", return_value=_row(40.5, -81.9, "Millersburg")
        ) as get2:
            result, err = maps.geocode("Millersburg")
            self.assertIsNone(err)
            self.assertIsNotNone(result)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get2.call_count, 1)

    def test_cache_stays_bounded(self):
        old_max = maps._GEOCODE_CACHE_MAX
        maps._GEOCODE_CACHE_MAX = 3
        try:
            def fake(url, timeout=12):
                return _row(40.0, -81.0, url)

            with patch.object(maps, "_http_get_json", side_effect=fake):
                for place in ("A", "B", "C", "D"):
                    maps.geocode(place)
            self.assertLessEqual(len(maps._GEOCODE_CACHE), 3)
            self.assertNotIn("a", maps._GEOCODE_CACHE)  # oldest evicted
            self.assertIn("d", maps._GEOCODE_CACHE)
        finally:
            maps._GEOCODE_CACHE_MAX = old_max


if __name__ == "__main__":
    unittest.main()
