"""Plex connector for VIKI: search the library, list players, play media.

Setup: PLEX_URL (e.g. http://192.168.1.10:32400) + PLEX_TOKEN in .env.
Get the token: Plex Web > Settings > authorized devices won't show it; the
reliable way is https://support.plex.tv/articles/204059436-finding-an-auth-token-x-plex-token/
(open any library item's XML in Plex Web and copy X-Plex-Token from the URL).
Optional: PLEX_PLAYER = default player name (e.g. "Living Room Roku").

Tools:
  plex_libraries()            - movie/show/music libraries on the server
  plex_search(query)          - search movies & episodes by title
  plex_on_deck()              - continue-watching items
  plex_players()              - Plex clients currently seen on the network
  plex_play(title, player)    - play the first matching movie/episode
  plex_pause(player)          - pause a player

Playing uses Plex Companion: the command goes straight from the server to
the player, so the Plex app just needs to be signed in on the same network.
If Home Assistant is connected and the player is a Roku, the Plex app is
launched on the Roku first so there's something to play on.
"""
import os

PLEX_TIMEOUT = 15


def configured() -> bool:
    return bool(os.getenv("PLEX_URL") and os.getenv("PLEX_TOKEN"))


def _server():
    if not configured():
        raise RuntimeError(
            "Plex is not connected. Set PLEX_URL and PLEX_TOKEN in .env."
        )
    try:
        from plexapi.server import PlexServer
    except ImportError:
        raise RuntimeError("plexapi is not installed. Run: pip install plexapi")
    return PlexServer(os.getenv("PLEX_URL"), os.getenv("PLEX_TOKEN"),
                      timeout=PLEX_TIMEOUT)


def _run(fn):
    try:
        return fn()
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        return f"Plex action failed: {type(e).__name__}: {e}"


def _describe(item) -> str:
    title = getattr(item, "title", "?")
    year = f" ({item.year})" if getattr(item, "year", None) else ""
    kind = getattr(item, "TYPE", None) or type(item).__name__
    extra = ""
    if kind == "episode":
        s, e = getattr(item, "seasonNumber", None), getattr(item, "episodeNumber", None)
        if s is not None and e is not None:
            extra = f" S{s:02d}E{e:02d}"
    return f"{title}{year}{extra} [{kind}]"


def plex_libraries():
    def go():
        server = _server()
        sections = server.library.sections()
        if not sections:
            return "No libraries found."
        return "Plex libraries:\n" + "\n".join(
            f"- {s.title} ({s.type})" for s in sections)
    return _run(go)


def plex_search(query):
    def go(q=query):
        server = _server()
        q = (q or "").strip()
        if not q:
            return "Give me a title to search for."
        results = server.library.search(q)
        if not results:
            return f"No Plex matches for '{q}'."
        return f"Plex matches for '{q}':\n" + "\n".join(
            f"- {_describe(r)}" for r in results[:10])
    return _run(go)


def plex_on_deck():
    def go():
        server = _server()
        items = server.library.onDeck()
        if not items:
            return "Nothing on deck."
        return "Continue watching:\n" + "\n".join(f"- {_describe(i)}" for i in items[:10])
    return _run(go)


def plex_players():
    def go():
        server = _server()
        clients = server.clients()
        if not clients:
            return ("No Plex players seen. Open the Plex app on the device "
                    "(signed in, same network) and try again.")
        return "Plex players:\n" + "\n".join(
            f"- {c.title} ({c.product}, {c.address})" for c in clients)
    return _run(go)


def _pick_player(server, name):
    name = (name or os.getenv("PLEX_PLAYER", "")).strip()
    clients = server.clients()
    if not clients:
        raise RuntimeError("No Plex players seen. Open the Plex app first.")
    if name:
        for c in clients:
            if name.lower() in c.title.lower():
                return c
        raise RuntimeError(
            f"No Plex player matching '{name}'. Available: "
            + ", ".join(c.title for c in clients))
    return clients[0]


def _maybe_launch_plex_on_roku(player_title):
    """If HA is connected and the player looks like a Roku, open the Plex app.

    Returns None on success (or when no launch is needed), else a
    short failure note for the caller to surface — a failed launch
    must not silently read as "Playing"."""
    if "roku" not in player_title.lower():
        return None
    try:
        from connectors import roku as roku_mod
        if roku_mod.ha_connected():
            result = roku_mod.roku_launch("Plex")
            if isinstance(result, str) and ("failed" in result or "No Roku" in result):
                return result
    except Exception as e:
        return f"could not open Plex on the Roku ({type(e).__name__})"
    return None


def plex_play(title, player=""):
    """Play the first Plex match for a title on a player."""
    def go(t=title, p=player):
        server = _server()
        t = (t or "").strip()
        if not t:
            return "Give me a title to play."
        results = server.library.search(t)
        if not results:
            return f"No Plex matches for '{t}'."
        item = results[0]
        client = _pick_player(server, p)
        launch_note = _maybe_launch_plex_on_roku(client.title)
        item.play(client)
        msg = f"Playing '{_describe(item)}' on {client.title}."
        if launch_note:
            msg += f" Note: {launch_note}"
        return msg
    return _run(go)


def plex_pause(player=""):
    def go():
        server = _server()
        client = _pick_player(server, player)
        client.pause()
        return f"Paused {client.title}."
    return _run(go)
