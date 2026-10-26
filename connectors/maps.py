"""Maps for VIKI (v0.2.27): distances, directions, construction, traffic,
plus the structured map payload the web UI renders as an in-chat map card.

Key-free by default, built on OpenStreetMap services:
- Nominatim for geocoding (address/place -> coordinates).
- OSRM demo server for routing (distance, duration, turn-by-turn steps).
- Overpass API for construction/roadworks near the route.

Live traffic is the one exception: it needs a TomTom API key. If
VIKI_TOMTOM_KEY is not set, traffic calls degrade gracefully with a
clear note — traffic is never faked and never crashes the answer.

Every public text function returns a plain-language string (the
agent-tool convention); network failures become a readable error,
never an exception to the user. All HTTP calls carry timeouts. Tests
mock _http_get_json / _http_post, so no test hits the real network.

Map cards: route() caches its result in _LAST_ROUTE. After the agent
runs a maps tool, it calls take_map_payload(origin, destination) to
get the structured payload (geometry, endpoints, links, traffic
segments) for the chat's map card — but only when the cached route
matches those exact endpoints, so a stale route is never shown.

Geocode lookups are cached in memory and on disk
(workspace/geocode_cache.json), so repeat directions stay fast across
server restarts.
"""

import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "VIKI/0.2.27"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/0/json"

DEFAULT_TIMEOUT = 12
OVERPASS_TIMEOUT = 30
# Map-card geometry is downsampled so the SSE payload stays small.
MAP_MAX_POINTS = 200
# Traffic is sampled at a few points along the route, not every vertex.
TRAFFIC_SAMPLE_POINTS = 5

# Last successful route(), keyed by the (origin, destination) strings the
# agent passed. Read by take_map_payload(); never shown unless it matches.
_LAST_ROUTE = None


def _http_get_json(url, timeout=DEFAULT_TIMEOUT):
    """GET a URL and parse the JSON body. Separated out so tests can mock it."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _http_post(url, data, timeout=OVERPASS_TIMEOUT):
    """POST form data and parse the JSON body. Separated out for tests."""
    body = urllib.parse.urlencode({"data": data}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _fmt_miles(meters):
    return f"{meters / 1609.344:.1f} miles"


def _fmt_minutes(seconds):
    mins = int(round(seconds / 60))
    if mins < 60:
        return f"{mins} min"
    hours, mins = divmod(mins, 60)
    return f"{hours} hr {mins} min" if mins else f"{hours} hr"


# --- geocoding -----------------------------------------------------------

# Geocode cache: normalized place string -> (lat, lon, display_name).
# Every route() call otherwise re-hits Nominatim for both endpoints, so
# repeat directions (same places, asked again) pay two lookups each time.
# Caching keeps those fast and stays polite to the free service. Only
# successful lookups are cached, so a transient failure never sticks.
# The cache is also written to disk, so repeat lookups stay fast across
# server restarts. workspace/ survives VIKI's self-update, so the file
# lives there; VIKI_GEOCODE_CACHE_FILE overrides the path (tests).
_GEOCODE_CACHE = {}
_GEOCODE_CACHE_MAX = 200
_CACHE_FILE = (
    Path(os.environ["VIKI_GEOCODE_CACHE_FILE"])
    if os.environ.get("VIKI_GEOCODE_CACHE_FILE")
    else Path(__file__).resolve().parent.parent / "workspace" / "geocode_cache.json"
)
_CACHE_LOADED = False


def _load_geocode_cache():
    """Load the disk cache once; a missing or corrupt file just means empty."""
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    try:
        raw = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    for key, val in list(raw.items())[:_GEOCODE_CACHE_MAX]:
        try:
            lat, lon, name = val
            _GEOCODE_CACHE[str(key)] = (float(lat), float(lon), str(name))
        except (TypeError, ValueError):
            continue


def _save_geocode_cache():
    """Best-effort atomic write; a cache failure must never break directions."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_GEOCODE_CACHE), encoding="utf-8")
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        pass


def clear_geocode_cache():
    """Drop all cached place lookups (tests, or a stale-entry reset)."""
    global _CACHE_LOADED
    _GEOCODE_CACHE.clear()
    _CACHE_LOADED = True
    try:
        _CACHE_FILE.unlink()
    except Exception:
        pass


def geocode(place):
    """Resolve a place/address to (lat, lon, display_name).

    Returns (None, error_string) on failure.
    """
    if not place or not str(place).strip():
        return None, "No place given to look up."
    _load_geocode_cache()
    place_key = str(place).strip().lower()
    if place_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[place_key], None
    query = urllib.parse.urlencode(
        {"q": str(place).strip(), "format": "json", "limit": 1}
    )
    try:
        rows = _http_get_json(f"{NOMINATIM_URL}?{query}")
    except Exception as e:
        return None, f"Could not look up {place!r}: {e}"
    if not rows:
        return None, f"No match found for {place!r}."
    row = rows[0]
    try:
        result = (
            float(row["lat"]),
            float(row["lon"]),
            row.get("display_name", str(place)),
        )
    except (KeyError, TypeError, ValueError):
        return None, f"Got an unexpected result looking up {place!r}."
    if len(_GEOCODE_CACHE) >= _GEOCODE_CACHE_MAX:
        _GEOCODE_CACHE.pop(next(iter(_GEOCODE_CACHE)))
    _GEOCODE_CACHE[place_key] = result
    _save_geocode_cache()
    return result, None


# --- routing -------------------------------------------------------------

def route(origin, destination):
    """Route between two places.

    Returns (route_dict, None) or (None, error_string). route_dict has
    distance_m, duration_s, steps (list of instruction strings), and
    coords (list of (lon, lat) geometry points).
    """
    start, err = geocode(origin)
    if err:
        return None, err
    end, err = geocode(destination)
    if err:
        return None, err
    (slat, slon, sname) = start
    (elat, elon, ename) = end
    coords = f"{slon},{slat};{elon},{elat}"
    url = (
        f"{OSRM_URL}/{coords}?overview=full&geometries=geojson"
        f"&steps=true&annotations=false"
    )
    try:
        data = _http_get_json(url)
    except Exception as e:
        return None, f"Could not plan a route: {e}"
    routes = data.get("routes") or []
    if not routes:
        return None, f"No driving route found from {origin!r} to {destination!r}."
    leg = routes[0]
    steps = []
    for leg_part in leg.get("legs", []):
        for step in leg_part.get("steps", []):
            man = step.get("maneuver", {}) or {}
            mtype = man.get("type", "")
            mod = man.get("modifier", "")
            name = step.get("name", "")
            dist = step.get("distance", 0)
            if mtype == "arrive":
                text = "Arrive at your destination"
            elif mtype == "depart":
                text = "Head out"
            elif mtype == "roundabout":
                text = "Go through the roundabout"
            elif mtype == "turn":
                text = f"Turn {mod}" if mod else "Turn"
            elif mtype == "merge":
                text = "Merge"
            elif mtype == "on ramp":
                text = "Take the ramp"
            elif mtype == "off ramp":
                text = "Take the exit"
            elif mtype == "fork":
                text = f"Keep {mod}" if mod else "Keep"
            elif mtype == "end of road":
                text = f"At the end of the road, turn {mod}" if mod else "At the end of the road"
            else:
                text = "Continue"
            if name and mtype != "arrive":
                text = f"{text} onto {name}"
            steps.append(f"{text} ({_fmt_miles(dist)})")
    geom = (leg.get("geometry") or {}).get("coordinates") or []
    route_dict = {
        "distance_m": leg.get("distance", 0),
        "duration_s": leg.get("duration", 0),
        "steps": steps,
        "coords": [(float(c[0]), float(c[1])) for c in geom if len(c) >= 2],
        "from_name": sname,
        "to_name": ename,
        "from_lat": slat,
        "from_lon": slon,
        "to_lat": elat,
        "to_lon": elon,
    }
    global _LAST_ROUTE
    _LAST_ROUTE = {"origin": origin, "destination": destination, "route": route_dict}
    return route_dict, None


# --- construction / roadworks ---------------------------------------------

def _bbox(coords, pad_deg=0.01):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (
        min(lats) - pad_deg,
        min(lons) - pad_deg,
        max(lats) + pad_deg,
        max(lons) + pad_deg,
    )


def construction_along_route(coords):
    """Find OSM construction/roadworks near a route.

    Returns a list of short descriptions (possibly empty), or an error
    string on failure. Never raises.
    """
    if not coords:
        return []
    south, west, north, east = _bbox(coords)
    query = (
        "[out:json][timeout:25];"
        "("
        f'way["highway"="construction"]({south},{west},{north},{east});'
        f'way["construction"]({south},{west},{north},{east});'
        ");"
        "out tags center 50;"
    )
    # Overpass is the flakiest of the map services (it timed out on a
    # Millersburg -> Killbuck check): one retry before giving up, so a
    # transient timeout doesn't blank the construction section.
    data = None
    last_error = None
    for _attempt in range(2):
        try:
            data = _http_post(OVERPASS_URL, query)
            break
        except Exception as e:
            last_error = e
    if data is None:
        return f"Construction check failed: {last_error}"
    found = []
    for el in data.get("elements", [])[:50]:
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or tags.get("ref") or "unnamed road"
        kind = tags.get("construction") or "roadwork"
        found.append(f"{name} ({kind})")
    # Dedupe, keep order.
    seen, out = set(), []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --- live traffic (TomTom, key-gated) --------------------------------------

def _tomtom_key():
    return os.getenv("VIKI_TOMTOM_KEY", "").strip()


def _traffic_speeds(lat, lon):
    """Raw TomTom flow data for one point.

    Returns (current_mph, freeflow_mph, road_closed) or (None, error_string).
    Key-gating lives in the callers: this helper assumes a key is set.
    """
    key = _tomtom_key()
    url = (
        f"{TOMTOM_URL}?point={lat},{lon}&unit=MPH"
        f"&key={urllib.parse.quote(key)}"
    )
    try:
        data = _http_get_json(url)
    except Exception as e:
        return None, f"Traffic lookup failed: {e}"
    seg = data.get("flowSegmentData") or {}
    if not seg:
        return None, "No traffic data for that point."
    cur, free = seg.get("currentSpeed"), seg.get("freeFlowSpeed")
    if cur is None or free is None:
        return None, "Traffic data was incomplete for that point."
    return (cur, free, bool(seg.get("roadClosure"))), None


def traffic_at(lat, lon):
    """Traffic at one point via TomTom flow data.

    Returns (summary_string, None) or (None, note_string). When no key is
    set, returns (None, key-note) — never faked data.
    """
    if not _tomtom_key():
        return None, (
            "Live traffic needs a TomTom API key "
            "(set VIKI_TOMTOM_KEY in .env)."
        )
    speeds, err = _traffic_speeds(lat, lon)
    if err:
        return None, err
    cur, free, closed = speeds
    if closed:
        return "Road closed on this segment.", None
    if cur >= free - 2:
        return f"Flowing freely ({cur} mph vs {free} mph free flow).", None
    return f"Slow traffic ({cur} mph vs {free} mph free flow).", None


def _sample_points(coords, max_points=5):
    """Evenly sample up to max_points along a route's geometry."""
    if not coords:
        return []
    if len(coords) <= max_points:
        return coords
    step = (len(coords) - 1) / (max_points - 1)
    return [coords[int(round(i * step))] for i in range(max_points)]


def traffic_along_route(coords):
    """Traffic summary along a route, or a key-note string if unavailable.

    Returns a single summary string (never raises).
    """
    if not _tomtom_key():
        return (
            "Live traffic unavailable: set VIKI_TOMTOM_KEY in .env "
            "to enable TomTom traffic."
        )
    points = _sample_points(coords)
    if not points:
        return "No route geometry to check traffic against."
    notes, slow = [], 0
    for lon, lat in points:
        summary, note = traffic_at(lat, lon)
        if summary:
            notes.append(summary)
            if summary.startswith("Slow") or "closed" in summary:
                slow += 1
        elif note and "needs a TomTom" not in note and "failed" in note:
            return note
    if not notes:
        return "No traffic data along this route."
    if slow == 0:
        return "Traffic is flowing freely along the route."
    return f"Traffic along the route: {slow} of {len(notes)} checked points slow. " + "; ".join(notes)


def traffic_segments(coords):
    """Per-point traffic speeds along a route, for the map card.

    Returns a list of {lat, lon, ratio, closed} (ratio = current/free
    flow, 1.0 = free flow) or None when traffic is unavailable (no key,
    no geometry, or every lookup failed). Never raises, never fakes.
    """
    if not _tomtom_key():
        return None
    points = _sample_points(coords, TRAFFIC_SAMPLE_POINTS)
    if not points:
        return None
    segments = []
    for lon, lat in points:
        speeds, err = _traffic_speeds(lat, lon)
        if err or not speeds:
            continue
        cur, free, closed = speeds
        ratio = (cur / free) if free else 1.0
        segments.append({"lat": lat, "lon": lon, "ratio": round(ratio, 2),
                         "closed": closed})
    return segments or None


def _adjusted_from_segments(segments, duration_s):
    """Adjusted duration from already-sampled traffic segments, or None.

    Shared by traffic_adjusted_duration() and take_map_payload() so the
    map card and the text answer agree: same samples, same scaling.
    Returns None when there are no usable samples or traffic is
    essentially free-flowing (mean ratio >= 0.98).
    """
    if not segments:
        return None
    ratios = [s["ratio"] for s in segments
              if not s.get("closed") and s["ratio"] > 0]
    if not ratios:
        return None
    mean_ratio = sum(ratios) / len(ratios)
    if mean_ratio >= 0.98:
        return None
    return duration_s / max(mean_ratio, 0.25)


def traffic_adjusted_duration(coords, duration_s):
    """Traffic-adjusted drive time for a route, or None.

    Averages the sampled TomTom ratios from traffic_segments()
    (ratio = current/free-flow speed; below 1 means slower than free
    flow) and scales the free-flow duration: adjusted =
    duration_s / mean_ratio. Returns None when traffic is unavailable
    (no key, no usable samples, every sample a closure) or when
    traffic is essentially free-flowing (mean ratio >= 0.98) — callers
    then just report the free-flow time. Ratios are floored at 0.25
    as a sanity cap so one bad sample can't inflate the estimate past
    4x free flow. Never raises, never fakes.
    """
    return _adjusted_from_segments(traffic_segments(coords), duration_s)


# --- map-card payload ------------------------------------------------------

def _simplify(coords, max_points=MAP_MAX_POINTS):
    """Downsample geometry for the map card, keeping first/last points."""
    if len(coords) <= max_points:
        return list(coords)
    step = (len(coords) - 1) / (max_points - 1)
    return [coords[int(round(i * step))] for i in range(max_points)]


def map_links(origin, destination, from_lat, from_lon, to_lat, to_lon):
    """Clickable map links for a route (fallback when the card can't render)."""
    google = (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={urllib.parse.quote(str(origin))}"
        f"&destination={urllib.parse.quote(str(destination))}"
    )
    osm = (
        "https://www.openstreetmap.org/directions?engine=fossgis_osrm_car"
        f"&route={from_lat},{from_lon}%3B{to_lat},{to_lon}"
    )
    return {"google": google, "osm": osm}


def take_map_payload(origin, destination):
    """Structured map payload for the most recent route() call.

    Returns None unless the cached route matches these exact endpoints,
    so the chat never shows a stale route. Geometry is [lat, lon] pairs
    (Leaflet order), downsampled. Traffic is None without a TomTom key.
    duration_traffic_s is the traffic-adjusted ETA (None when traffic
    is free-flowing or unavailable), matching the text answer.
    """
    cached = _LAST_ROUTE
    if not cached or cached.get("origin") != origin or cached.get("destination") != destination:
        return None
    r = cached["route"]
    geom = _simplify(r["coords"])
    # Sample traffic once and derive the adjusted ETA from those same
    # samples, so the card header matches the text answer (previously
    # the card always showed the free-flow time, even when the text
    # said "50 min with current traffic").
    segments = traffic_segments(r["coords"])
    adjusted = _adjusted_from_segments(segments, r["duration_s"])
    return {
        "origin": {"lat": r["from_lat"], "lon": r["from_lon"], "name": r["from_name"]},
        "destination": {"lat": r["to_lat"], "lon": r["to_lon"], "name": r["to_name"]},
        "geometry": [[lat, lon] for lon, lat in geom],
        "distance_m": r["distance_m"],
        "duration_s": r["duration_s"],
        "duration_traffic_s": int(round(adjusted)) if adjusted else None,
        "links": map_links(origin, destination,
                           r["from_lat"], r["from_lon"], r["to_lat"], r["to_lon"]),
        "traffic": segments,
    }


# --- agent tools (plain-language strings) ----------------------------------

def maps_distance(origin, destination):
    """Distance and drive time from one place to another.

    The drive time is traffic-adjusted when a TomTom key is set (see
    traffic_adjusted_duration); without one it stays the free-flow time.
    """
    r, err = route(origin, destination)
    if err:
        return err
    adjusted = traffic_adjusted_duration(r["coords"], r["duration_s"])
    if adjusted:
        time_txt = (f"about {_fmt_minutes(adjusted)} drive with current "
                    f"traffic ({_fmt_minutes(r['duration_s'])} free flow)")
    else:
        time_txt = f"about {_fmt_minutes(r['duration_s'])} drive"
    return (
        f"{origin} to {destination}: {_fmt_miles(r['distance_m'])} "
        f"({time_txt})."
    )


def maps_directions(origin, destination):
    """Turn-by-turn directions, plus construction on the route.

    The header drive time is traffic-adjusted when a TomTom key is set
    (see traffic_adjusted_duration); without one it stays free-flow.
    """
    r, err = route(origin, destination)
    if err:
        return err
    adjusted = traffic_adjusted_duration(r["coords"], r["duration_s"])
    if adjusted:
        header = (f"Route: {origin} to {destination} — "
                  f"{_fmt_miles(r['distance_m'])}, about "
                  f"{_fmt_minutes(adjusted)} with current traffic "
                  f"({_fmt_minutes(r['duration_s'])} free flow).")
    else:
        header = (f"Route: {origin} to {destination} — "
                  f"{_fmt_miles(r['distance_m'])}, about "
                  f"{_fmt_minutes(r['duration_s'])}.")
    lines = [header]
    for i, step in enumerate(r["steps"], 1):
        lines.append(f"{i}. {step}")
    if not r["steps"]:
        lines.append("(No step-by-step directions returned.)")
    works = construction_along_route(r["coords"])
    if isinstance(works, str):
        lines.append(f"Construction note: {works}")
    elif works:
        lines.append("Construction / roadworks on this route:")
        lines.extend(f"- {w}" for w in works)
    else:
        lines.append("No construction reported on this route.")
    lines.append("The route is also shown on the map card in this chat.")
    return "\n".join(lines)


def maps_traffic(origin, destination):
    """Live traffic along the route between two places."""
    r, err = route(origin, destination)
    if err:
        return err
    summary = traffic_along_route(r["coords"])
    return (
        f"Traffic for {origin} to {destination} "
        f"({_fmt_miles(r['distance_m'])}, about {_fmt_minutes(r['duration_s'])}):\n"
        f"{summary}"
    )


def maps_configured():
    """Maps tools need no key for geocoding, routing, or construction."""
    return True
