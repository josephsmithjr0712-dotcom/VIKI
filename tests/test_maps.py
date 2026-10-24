"""Tests for connectors/maps.py (v0.2.25).

All HTTP is mocked — no real network calls. Covers geocode parsing,
route parsing (distance/duration/steps), construction filtering,
traffic degradation without a key, and HTTP failure handling.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.maps as maps  # noqa: E402


def _nominatim(lat, lon, name):
    return [{"lat": str(lat), "lon": str(lon), "display_name": name}]


_OSRM = {
    "routes": [
        {
            "distance": 16093.44,
            "duration": 1500,
            "legs": [
                {
                    "steps": [
                        {
                            "maneuver": {"type": "depart"},
                            "name": "Main St",
                            "distance": 500,
                        },
                        {
                            "maneuver": {"type": "turn", "modifier": "left"},
                            "name": "Oak Ave",
                            "distance": 15593.44,
                        },
                        {
                            "maneuver": {"type": "arrive"},
                            "name": "",
                            "distance": 0,
                        },
                    ]
                }
            ],
            "geometry": {
                "coordinates": [[-81.9, 40.4], [-81.8, 40.45], [-81.7, 40.5]]
            },
        }
    ]
}

_OVERPASS = {
    "elements": [
        {"tags": {"highway": "construction", "name": "State Route 83"}},
        {"tags": {"highway": "residential", "construction": "resurfacing"}},
        {"tags": {"highway": "construction", "name": "State Route 83"}},
    ]
}


def _route_side_effect(*args, **kwargs):
    return _nominatim(40.0, -81.0, "Start Town")


class TestGeocode(unittest.TestCase):
    def test_parses_result(self):
        with patch.object(
            maps, "_http_get_json",
            return_value=_nominatim(40.123, -81.456, "Millersburg, Ohio"),
        ):
            (coords, err) = maps.geocode("Millersburg OH")
        self.assertIsNone(err)
        self.assertEqual(coords[0], 40.123)
        self.assertEqual(coords[1], -81.456)
        self.assertIn("Millersburg", coords[2])

    def test_no_match(self):
        with patch.object(maps, "_http_get_json", return_value=[]):
            (coords, err) = maps.geocode("Nowhere XYZ")
        self.assertIsNone(coords)
        self.assertIn("No match", err)

    def test_http_failure_is_error_string(self):
        with patch.object(
            maps, "_http_get_json", side_effect=Exception("boom")
        ):
            (coords, err) = maps.geocode("Millersburg OH")
        self.assertIsNone(coords)
        self.assertIn("Could not look up", err)


class TestRoute(unittest.TestCase):
    def _mocked(self, mock_get):
        mock_get.side_effect = [
            _nominatim(40.0, -81.9, "Start"),
            _nominatim(40.5, -81.7, "End"),
            _OSRM,
        ]

    def test_parses_distance_duration_steps(self):
        with patch.object(maps, "_http_get_json") as mock_get:
            self._mocked(mock_get)
            (r, err) = maps.route("A", "B")
        self.assertIsNone(err)
        self.assertAlmostEqual(r["distance_m"], 16093.44)
        self.assertEqual(r["duration_s"], 1500)
        self.assertIn("Head out onto Main St", r["steps"][0])
        self.assertIn("Turn left onto Oak Ave", r["steps"][1])
        self.assertEqual(r["steps"][2], "Arrive at your destination (0.0 miles)")

    def test_no_route(self):
        with patch.object(maps, "_http_get_json") as mock_get:
            mock_get.side_effect = [
                _nominatim(40.0, -81.9, "Start"),
                _nominatim(40.5, -81.7, "End"),
                {"routes": []},
            ]
            (r, err) = maps.route("A", "B")
        self.assertIsNone(r)
        self.assertIn("No driving route", err)

    def test_http_failure_is_error_string(self):
        with patch.object(
            maps, "_http_get_json", side_effect=Exception("down")
        ):
            (r, err) = maps.route("A", "B")
        self.assertIsNone(r)
        self.assertIn("Could not look up", err)


class TestTools(unittest.TestCase):
    def test_distance_output(self):
        with patch.object(maps, "_http_get_json") as mock_get:
            mock_get.side_effect = [
                _nominatim(40.0, -81.9, "Start"),
                _nominatim(40.5, -81.7, "End"),
                _OSRM,
            ]
            out = maps.maps_distance("A", "B")
        self.assertIn("10.0 miles", out)
        self.assertIn("25 min", out)

    def test_directions_include_steps_and_construction(self):
        with patch.object(maps, "_http_get_json") as mock_get, patch.object(
            maps, "_http_post", return_value=_OVERPASS
        ):
            mock_get.side_effect = [
                _nominatim(40.0, -81.9, "Start"),
                _nominatim(40.5, -81.7, "End"),
                _OSRM,
            ]
            out = maps.maps_directions("A", "B")
        self.assertIn("1. Head out onto Main St", out)
        self.assertIn("Construction / roadworks", out)
        self.assertIn("State Route 83", out)
        # Dedupe: the same road listed twice in Overpass appears once.
        self.assertEqual(out.count("State Route 83"), 1)

    def test_directions_construction_failure_is_note(self):
        with patch.object(maps, "_http_get_json") as mock_get, patch.object(
            maps, "_http_post", side_effect=Exception("overpass down")
        ):
            mock_get.side_effect = [
                _nominatim(40.0, -81.9, "Start"),
                _nominatim(40.5, -81.7, "End"),
                _OSRM,
            ]
            out = maps.maps_directions("A", "B")
        self.assertIn("Construction note:", out)
        self.assertNotIn("Traceback", out)


class TestTraffic(unittest.TestCase):
    def test_no_key_degrades_gracefully(self):
        with patch.dict(os.environ, {"VIKI_TOMTOM_KEY": ""}):
            (summary, note) = maps.traffic_at(40.0, -81.9)
        self.assertIsNone(summary)
        self.assertIn("TomTom", note)
        self.assertIn("VIKI_TOMTOM_KEY", note)

    def test_with_key_slow_and_free(self):
        with patch.dict(os.environ, {"VIKI_TOMTOM_KEY": "fake-key"}):
            with patch.object(
                maps, "_http_get_json",
                return_value={"flowSegmentData": {
                    "currentSpeed": 25, "freeFlowSpeed": 55}},
            ):
                (summary, note) = maps.traffic_at(40.0, -81.9)
            self.assertIsNone(note)
            self.assertIn("Slow traffic", summary)
            with patch.object(
                maps, "_http_get_json",
                return_value={"flowSegmentData": {
                    "currentSpeed": 55, "freeFlowSpeed": 55}},
            ):
                (summary, note) = maps.traffic_at(40.0, -81.9)
            self.assertIn("Flowing freely", summary)

    def test_traffic_along_route_without_key(self):
        with patch.dict(os.environ, {"VIKI_TOMTOM_KEY": ""}):
            out = maps.traffic_along_route([(-81.9, 40.0)])
        self.assertIn("VIKI_TOMTOM_KEY", out)


if __name__ == "__main__":
    unittest.main()
