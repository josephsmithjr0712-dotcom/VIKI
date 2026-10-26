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

    def test_construction_retries_once_then_succeeds(self):
        with patch.object(
            maps, "_http_post",
            side_effect=[Exception("timeout"), _OVERPASS],
        ) as mock_post:
            out = maps.construction_along_route([(-81.9, 40.0), (-81.8, 40.1)])
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("State Route 83 (roadwork)", out)

    def test_construction_gives_up_after_two_failures(self):
        with patch.object(
            maps, "_http_post", side_effect=Exception("overpass down")
        ) as mock_post:
            out = maps.construction_along_route([(-81.9, 40.0)])
        self.assertEqual(mock_post.call_count, 2)
        self.assertIsInstance(out, str)
        self.assertIn("Construction check failed", out)

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


class TestMapPayload(unittest.TestCase):
    def _run_directions(self, mock_get, mock_post):
        mock_get.side_effect = [
            _nominatim(40.0, -81.9, "Start Town"),
            _nominatim(40.5, -81.7, "End Town"),
            _OSRM,
        ]
        mock_post.return_value = _OVERPASS
        return maps.maps_directions("A", "B")

    def test_route_carries_endpoint_coords(self):
        with patch.object(maps, "_http_get_json") as mock_get:
            mock_get.side_effect = [
                _nominatim(40.0, -81.9, "Start"),
                _nominatim(40.5, -81.7, "End"),
                _OSRM,
            ]
            (r, err) = maps.route("A", "B")
        self.assertIsNone(err)
        self.assertEqual((r["from_lat"], r["from_lon"]), (40.0, -81.9))
        self.assertEqual((r["to_lat"], r["to_lon"]), (40.5, -81.7))

    def test_payload_geometry_links_and_no_traffic_without_key(self):
        with patch.dict(os.environ, {"VIKI_TOMTOM_KEY": ""}):
            with patch.object(maps, "_http_get_json") as mock_get, patch.object(
                maps, "_http_post"
            ) as mock_post:
                self._run_directions(mock_get, mock_post)
                payload = maps.take_map_payload("A", "B")
        self.assertIsNotNone(payload)
        # Leaflet order: [lat, lon].
        self.assertEqual(payload["geometry"][0], [40.4, -81.9])
        self.assertEqual(payload["origin"]["lat"], 40.0)
        self.assertEqual(payload["destination"]["name"], "End Town")
        self.assertAlmostEqual(payload["distance_m"], 16093.44)
        self.assertIn("google.com/maps/dir", payload["links"]["google"])
        self.assertIn("openstreetmap.org/directions", payload["links"]["osm"])
        self.assertIsNone(payload["traffic"])

    def test_payload_stale_endpoints_return_none(self):
        with patch.object(maps, "_http_get_json") as mock_get, patch.object(
            maps, "_http_post"
        ) as mock_post:
            self._run_directions(mock_get, mock_post)
        self.assertIsNone(maps.take_map_payload("A", "Somewhere Else"))
        self.assertIsNone(maps.take_map_payload("Elsewhere", "B"))

    def test_simplify_keeps_first_last_and_caps_points(self):
        coords = [(float(-81.0 - i * 0.001), 40.0) for i in range(500)]
        out = maps._simplify(coords)
        self.assertLessEqual(len(out), 200)
        self.assertEqual(out[0], coords[0])
        self.assertEqual(out[-1], coords[-1])

    def test_traffic_segments_with_key(self):
        with patch.dict(os.environ, {"VIKI_TOMTOM_KEY": "fake-key"}):
            with patch.object(
                maps, "_http_get_json",
                return_value={"flowSegmentData": {
                    "currentSpeed": 27, "freeFlowSpeed": 55}},
            ):
                segs = maps.traffic_segments([(-81.9, 40.0), (-81.8, 40.1)])
        self.assertIsNotNone(segs)
        self.assertAlmostEqual(segs[0]["ratio"], 27 / 55, places=2)
        self.assertFalse(segs[0]["closed"])

    def test_directions_text_mentions_map_card(self):
        with patch.object(maps, "_http_get_json") as mock_get, patch.object(
            maps, "_http_post"
        ) as mock_post:
            out = self._run_directions(mock_get, mock_post)
        self.assertIn("map card", out)


class TestTrafficAdjustedDuration(unittest.TestCase):
    def test_no_traffic_returns_none(self):
        with patch.object(maps, "traffic_segments", return_value=None):
            self.assertIsNone(maps.traffic_adjusted_duration([], 1500))

    def test_free_flow_returns_none(self):
        segs = [{"lat": 40.0, "lon": -81.9, "ratio": 1.0, "closed": False}]
        with patch.object(maps, "traffic_segments", return_value=segs):
            self.assertIsNone(maps.traffic_adjusted_duration([], 1500))

    def test_slow_traffic_scales_duration(self):
        segs = [{"lat": 40.0, "lon": -81.9, "ratio": 0.5, "closed": False}]
        with patch.object(maps, "traffic_segments", return_value=segs):
            self.assertEqual(maps.traffic_adjusted_duration([], 1500), 3000)

    def test_closed_segments_ignored(self):
        segs = [
            {"lat": 40.0, "lon": -81.9, "ratio": 0.0, "closed": True},
            {"lat": 40.1, "lon": -81.8, "ratio": 0.5, "closed": False},
        ]
        with patch.object(maps, "traffic_segments", return_value=segs):
            self.assertEqual(maps.traffic_adjusted_duration([], 1500), 3000)

    def test_all_closed_returns_none(self):
        segs = [{"lat": 40.0, "lon": -81.9, "ratio": 0.0, "closed": True}]
        with patch.object(maps, "traffic_segments", return_value=segs):
            self.assertIsNone(maps.traffic_adjusted_duration([], 1500))

    def test_ratio_floor_caps_at_4x(self):
        segs = [{"lat": 40.0, "lon": -81.9, "ratio": 0.1, "closed": False}]
        with patch.object(maps, "traffic_segments", return_value=segs):
            self.assertEqual(maps.traffic_adjusted_duration([], 1500), 6000)

    def test_distance_text_shows_adjusted_time(self):
        segs = [{"lat": 40.0, "lon": -81.9, "ratio": 0.5, "closed": False}]
        with patch.object(maps, "_http_get_json") as mock_get, patch.object(
            maps, "traffic_segments", return_value=segs
        ):
            mock_get.side_effect = [
                _nominatim(40.0, -81.9, "Start"),
                _nominatim(40.5, -81.7, "End"),
                _OSRM,
            ]
            out = maps.maps_distance("A", "B")
        self.assertIn("about 50 min drive with current traffic", out)
        self.assertIn("25 min free flow", out)

    def test_distance_text_free_flow_unchanged(self):
        segs = [{"lat": 40.0, "lon": -81.9, "ratio": 1.0, "closed": False}]
        with patch.object(maps, "_http_get_json") as mock_get, patch.object(
            maps, "traffic_segments", return_value=segs
        ):
            mock_get.side_effect = [
                _nominatim(40.0, -81.9, "Start"),
                _nominatim(40.5, -81.7, "End"),
                _OSRM,
            ]
            out = maps.maps_distance("A", "B")
        self.assertIn("about 25 min drive", out)
        self.assertNotIn("current traffic", out)

    def test_directions_header_shows_adjusted_time(self):
        segs = [{"lat": 40.0, "lon": -81.9, "ratio": 0.5, "closed": False}]
        with patch.object(maps, "_http_get_json") as mock_get, patch.object(
            maps, "_http_post"
        ) as mock_post, patch.object(
            maps, "traffic_segments", return_value=segs
        ):
            mock_get.side_effect = [
                _nominatim(40.0, -81.9, "Start"),
                _nominatim(40.5, -81.7, "End"),
                _OSRM,
            ]
            mock_post.return_value = _OVERPASS
            out = maps.maps_directions("A", "B")
        self.assertIn("about 50 min with current traffic", out)
        self.assertIn("25 min free flow", out)


if __name__ == "__main__":
    unittest.main()
