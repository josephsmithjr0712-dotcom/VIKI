"""Spotify connector for VIKI: search and playback control.

Setup:
1. Create an app at https://developer.spotify.com/dashboard, add
   http://127.0.0.1:8888/callback as a Redirect URI.
2. Set SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET in .env.
3. Run: python -m connectors.spotify_auth  (one-time browser approval)

Note: controlling playback via the API needs Spotify Premium.

Tools:
  spotify_search(query)       - search tracks/artists/albums/playlists
  spotify_play(query, device) - play the top track match
  spotify_play_uri(uri, device) - play any Spotify URI (album/playlist/track)
  spotify_pause() / spotify_resume()
  spotify_next() / spotify_previous()
  spotify_devices()           - speakers/phones/apps Spotify can play on
  spotify_now_playing()
  spotify_volume(percent)
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".spotify_cache"
SCOPES = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
REDIRECT = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")


def connected() -> bool:
    return bool(CACHE.exists() and os.getenv("SPOTIFY_CLIENT_ID")
                and os.getenv("SPOTIFY_CLIENT_SECRET"))


def _sp():
    if not os.getenv("SPOTIFY_CLIENT_ID") or not os.getenv("SPOTIFY_CLIENT_SECRET"):
        raise RuntimeError(
            "Spotify is not connected. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET, then run: python -m connectors.spotify_auth"
        )
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError:
        raise RuntimeError("spotipy is not installed. Run: pip install spotipy")
    auth = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=REDIRECT,
        scope=SCOPES,
        cache_path=str(CACHE),
        open_browser=False,
    )
    if not auth.get_cached_token():
        raise RuntimeError(
            "Spotify needs approval first. Run: python -m connectors.spotify_auth")
    return spotipy.Spotify(auth_manager=auth)


def _run(fn):
    try:
        return fn()
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        msg = str(e)
        if "NO_ACTIVE_DEVICE" in msg or "404" in msg:
            return ("No active Spotify device. Open Spotify on a speaker/phone "
                    "first, or name one with the device argument.")
        return f"Spotify action failed: {type(e).__name__}: {e}"


def _pick_device(sp, name):
    devs = sp.devices().get("devices", [])
    if not devs:
        raise RuntimeError("No Spotify devices found. Open Spotify somewhere first.")
    name = (name or "").strip().lower()
    if name:
        for d in devs:
            if name in d["name"].lower():
                return d
        raise RuntimeError("No Spotify device matching '%s'. Available: %s"
                           % (name, ", ".join(d["name"] for d in devs)))
    for d in devs:
        if d.get("is_active"):
            return d
    return devs[0]


def _track_line(t):
    artists = ", ".join(a["name"] for a in t.get("artists", []))
    return f"{t['name']} — {artists}"


def spotify_search(query, kind="track"):
    def go(q=query, k=kind):
        sp = _sp()
        q = (q or "").strip()
        k = (k or "track").strip().lower()
        if k not in ("track", "album", "artist", "playlist"):
            return "kind must be track, album, artist, or playlist."
        if not q:
            return "Give me something to search for."
        res = sp.search(q, limit=5, type=k)
        items = res.get(k + "s", {}).get("items", [])
        if not items:
            return f"No Spotify {k} matches for '{q}'."
        lines = []
        for it in items:
            if k == "track":
                lines.append(f"{_track_line(it)}\n  {it['uri']}")
            elif k == "artist":
                lines.append(f"{it['name']}\n  {it['uri']}")
            else:
                owner = it.get("owner", {}).get("display_name", "")
                lines.append(f"{it['name']}" + (f" by {owner}" if owner else "")
                             + f"\n  {it['uri']}")
        return f"Spotify {k} matches for '{q}':\n" + "\n".join(lines)
    return _run(go)


def spotify_play(query, device=""):
    def go(q=query, d=device):
        sp = _sp()
        q = (q or "").strip()
        if not q:
            return "Give me a song to play."
        res = sp.search(q, limit=1, type="track")
        items = res.get("tracks", {}).get("items", [])
        if not items:
            return f"No Spotify tracks found for '{q}'."
        track = items[0]
        dev = _pick_device(sp, d)
        sp.start_playback(device_id=dev["id"], uris=[track["uri"]])
        return f"Playing '{_track_line(track)}' on {dev['name']}."
    return _run(go)


def spotify_play_uri(uri, device=""):
    def go(u=uri, d=device):
        sp = _sp()
        u = (u or "").strip()
        if not u.startswith("spotify:"):
            return "Give me a Spotify URI (spotify:track:... / spotify:album:... / spotify:playlist:...)."
        dev = _pick_device(sp, d)
        if ":track:" in u:
            sp.start_playback(device_id=dev["id"], uris=[u])
        else:
            sp.start_playback(device_id=dev["id"], context_uri=u)
        return f"Playing on {dev['name']}."
    return _run(go)


def spotify_pause():
    def go():
        _sp().pause_playback()
        return "Paused."
    return _run(go)


def spotify_resume():
    def go():
        _sp().start_playback()
        return "Resumed."
    return _run(go)


def spotify_next():
    def go():
        _sp().next_track()
        return "Skipped."
    return _run(go)


def spotify_previous():
    def go():
        _sp().previous_track()
        return "Went back."
    return _run(go)


def spotify_devices():
    def go():
        sp = _sp()
        devs = sp.devices().get("devices", [])
        if not devs:
            return "No Spotify devices found. Open Spotify somewhere first."
        return "Spotify devices:\n" + "\n".join(
            f"- {d['name']} ({d['type']})" + (" [active]" if d.get("is_active") else "")
            for d in devs)
    return _run(go)


def spotify_now_playing():
    def go():
        pb = _sp().current_playback()
        if not pb or not pb.get("item"):
            return "Nothing playing."
        item = pb["item"]
        dev = (pb.get("device") or {}).get("name", "?")
        state = "playing" if pb.get("is_playing") else "paused"
        return f"{_track_line(item)} — {state} on {dev}."
    return _run(go)


def spotify_volume(percent):
    def go(p=percent):
        sp = _sp()
        try:
            p = int(p)
        except (TypeError, ValueError):
            return "Volume must be 0-100."
        p = max(0, min(100, p))
        dev = _pick_device(sp, "")
        sp.volume(p, device_id=dev["id"])
        return f"Volume set to {p}% on {dev['name']}."
    return _run(go)
