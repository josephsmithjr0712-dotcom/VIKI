"""VIKI's toolbox. Each tool is a plain function plus a JSON schema the model sees."""
import ast, json, math, operator, os, re, threading, urllib.parse, urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from connectors.google import (
    calendar_create,
    calendar_list,
    gmail_read,
    gmail_search,
    gmail_send,
    token_exists as google_connected,
)
from connectors.github import (
    github_create_issue,
    github_list_issues,
    github_read_file,
    github_search_code,
    github_search_repos,
    token_exists as github_connected,
)
from connectors.browser import (
    browser_click,
    browser_fill,
    browser_goto,
    browser_screenshot,
    browser_snapshot,
    browser_text,
)
from connectors.home_mini import announce as home_announce, list_devices as cast_list_devices
from connectors.maps import maps_directions, maps_distance, maps_traffic
from connectors.roku import (
    ha_connected,
    roku_apps,
    roku_key,
    roku_launch,
    roku_play_movie,
    roku_search,
    roku_status,
    roku_type,
)
from connectors.plex import (
    configured as plex_configured,
    plex_libraries,
    plex_on_deck,
    plex_pause,
    plex_play,
    plex_players,
    plex_search,
)
from connectors.spotify import (
    connected as spotify_connected,
    spotify_devices,
    spotify_next,
    spotify_now_playing,
    spotify_pause,
    spotify_play,
    spotify_play_uri,
    spotify_previous,
    spotify_resume,
    spotify_search,
    spotify_volume,
)

WORKSPACE = Path(os.getenv("VIKI_WORKSPACE") or Path(__file__).resolve().parent / "workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (VIKI agent)"}


def _safe_path(name):
    p = (WORKSPACE / name).resolve()
    if WORKSPACE.resolve() not in p.parents and p != WORKSPACE.resolve():
        raise ValueError("path is outside the workspace")
    return p


class _Text(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "svg", "noscript"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, a):
        self.skip += tag in self.SKIP

    def handle_endtag(self, tag):
        self.skip = max(0, self.skip - (tag in self.SKIP))

    def handle_data(self, d):
        if not self.skip and d.strip():
            self.out.append(d.strip())


# --- tool implementations -------------------------------------------------
def web_search(query, max_results=6):
    from ddgs import DDGS
    try:
        rows = DDGS().text(query, max_results=int(max_results), backend="lite")
    except Exception as e:
        return f"Search failed: {e}"
    return "\n\n".join(f"{r.get('title')}\n{r.get('body')}\n{r.get('href')}" for r in rows) or "No results."


def read_webpage(url, max_chars=6000):
    if not re.match(r"https?://", url):
        return "Only http(s) URLs are supported."
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=12) as r:
            html = r.read(1_500_000).decode("utf-8", "ignore")
        p = _Text(); p.feed(html)
        return " ".join(p.out)[: int(max_chars)] or "Page had no readable text."
    except Exception as e:
        return f"Could not read {url}: {e}"


def get_weather(location, unit="Fahrenheit"):
    try:
        url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=10) as r:
            d = json.load(r)
        c, f = d["current_condition"][0], unit.lower().startswith("f")
        days = ", ".join(
            f"{w['date']}: {w['mintemp'+('F' if f else 'C')]}-{w['maxtemp'+('F' if f else 'C')]}"
            for w in d.get("weather", [])[:3]
        )
        return (f"{location}: {c['weatherDesc'][0]['value']}, {c['temp_F' if f else 'temp_C']}°{'F' if f else 'C'}, "
                f"humidity {c['humidity']}%, wind {c['windspeedMiles' if f else 'windspeedKmph']} {'mph' if f else 'km/h'}. "
                f"Next days: {days}")
    except Exception as e:
        return f"Weather lookup failed for {location}: {e}"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg}


def calculate(expression):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("sqrt", "log", "sin", "cos", "tan", "abs", "round"):
            fn = {"abs": abs, "round": round}.get(n.func.id) or getattr(math, n.func.id)
            return fn(*map(ev, n.args))
        if isinstance(n, ast.Name) and n.id in ("pi", "e"):
            return getattr(math, n.id)
        raise ValueError("unsupported expression")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Cannot calculate: {e}"


def current_time():
    return datetime.now().astimezone().strftime("%A, %B %d %Y, %I:%M %p %Z")


def list_files():
    return "\n".join(str(p.relative_to(WORKSPACE)) for p in sorted(WORKSPACE.rglob("*")) if p.is_file()) or "(workspace is empty)"


def read_file(path):
    try:
        return _safe_path(path).read_text(encoding="utf-8")[:12000]
    except Exception as e:
        return f"Cannot read {path}: {e}"


def write_file(path, content):
    try:
        p = _safe_path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Cannot write {path}: {e}"


# Cached entity list, refreshed at startup (see refresh_ha_entities).
# Gives the model a map of what's actually in the house.
# "entities" = compact list (id, name, state) for the system prompt.
# "full" = complete entity states with all attributes, for the dump action.
HA_ENTITIES_CACHE = {"updated": None, "entities": [], "full": []}


def _ha_request(path, body=None, timeout=10):
    """Low-level HA API call. Returns parsed JSON."""
    base, token = os.getenv("HA_URL", "").rstrip("/"), os.getenv("HA_TOKEN", "")
    hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(f"{base}{path}", data=data, headers=hdr), timeout=timeout) as r:
        return json.load(r)


def refresh_ha_entities():
    """Pull the full entity list from Home Assistant into HA_ENTITIES_CACHE.
    Call at startup and periodically; failures are logged, never fatal."""
    try:
        states = _ha_request("/api/states", timeout=15)
        ents = []
        for s in states:
            eid = s.get("entity_id", "")
            attrs = s.get("attributes", {}) or {}
            ents.append({
                "entity_id": eid,
                "domain": eid.split(".")[0] if "." in eid else "?",
                "name": attrs.get("friendly_name", eid),
                "state": s.get("state"),
            })
        HA_ENTITIES_CACHE["entities"] = ents
        HA_ENTITIES_CACHE["full"] = states  # complete states with all attributes
        HA_ENTITIES_CACHE["updated"] = datetime.now().isoformat(timespec="seconds")
        print(f"[ha] cached {len(ents)} entities")
    except Exception as e:
        print(f"[ha] entity refresh failed: {e} (will retry on next list call)")


def ha_entities_text():
    """Compact human-readable entity list for the system prompt."""
    ents = HA_ENTITIES_CACHE["entities"]
    if not ents:
        return ""
    lines = []
    for e in sorted(ents, key=lambda x: x["entity_id"]):
        lines.append(f"- {e['entity_id']} ({e['name']}): {e['state']}")
    return "\n".join(lines)


def home_assistant(action, entity_id="", domain="", service="", data=None):
    try:
        if action == "list":
            # Refresh if the cache is empty/stale, then return the compact list
            if not HA_ENTITIES_CACHE["entities"]:
                refresh_ha_entities()
            ents = HA_ENTITIES_CACHE["entities"]
            if not ents:
                return "Could not reach Home Assistant to list entities."
            lines = [f"{e['entity_id']} | {e['name']} | {e['state']}" for e in sorted(ents, key=lambda x: x["entity_id"])]
            return "\n".join(lines)
        elif action == "dump":
            # Full entity dump: every entity with ALL attributes (brightness, color,
            # temperature, etc.). Use when you need details the compact list lacks.
            # Optional entity_id filters to one entity; otherwise returns everything.
            if not HA_ENTITIES_CACHE["full"]:
                refresh_ha_entities()
            full = HA_ENTITIES_CACHE["full"]
            if not full:
                return "Could not reach Home Assistant for entity dump."
            if entity_id:
                match = [s for s in full if s.get("entity_id") == entity_id]
                if not match:
                    return f"Entity {entity_id} not found."
                return json.dumps(match[0], indent=1)[:4000]
            # Return everything, trimmed to a reasonable size
            return json.dumps(full, indent=1)[:12000]
        elif action == "state":
            out = _ha_request(f"/api/states/{entity_id}")
            return json.dumps({"state": out["state"], "attributes": out["attributes"]})[:2000]
        elif action == "call":
            out = _ha_request(f"/api/services/{domain}/{service}", {"entity_id": entity_id, **(data or {})})
            return json.dumps(out)[:2000]
        else:
            return "action must be 'list', 'state', 'dump', or 'call'"
    except Exception as e:
        return f"Home Assistant error: {e}"


# --- registry ---------------------------------------------------------------
def _spec(name, desc, props=None, required=(), risky=False, risky_if=None):
    """risky_if: optional callable(args_dict) -> bool for argument-aware approval
    (used when only some calls of a tool are dangerous)."""
    return {"risky": risky, "risky_if": risky_if, "schema": {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": list(required)}}}}


# HA domains where a `call` action is security-sensitive (physical security /
# access control) and must ask for approval again, even though the general HA
# approval gate was removed. Everything else acts directly.
SECURE_HA_DOMAINS = frozenset({"lock", "alarm_control_panel", "cover"})


def _ha_call_needs_approval(args):
    args = args or {}
    return args.get("action") == "call" and \
        str(args.get("domain", "")).strip().lower() in SECURE_HA_DOMAINS


S = lambda d: {"type": "string", "description": d}
SPECS = {
    "update_plan": _spec("update_plan", "Show the user your plan for a multi-step task. Call it first, then again as steps finish. status: pending|doing|done.",
        {"steps": {"type": "array", "items": {"type": "object", "properties": {"text": S("step"), "status": S("pending|doing|done")}}}}, ["steps"]),
    "web_search": _spec("web_search", "Search the web for current information, news, or facts you don't know.", {"query": S("search query")}, ["query"]),
    "read_webpage": _spec("read_webpage", "Open a URL and return its readable text. Use after web_search to read a promising result.", {"url": S("full URL")}, ["url"]),
    "get_weather": _spec("get_weather", "Current weather and 3-day outlook for a place.", {"location": S("city, e.g. Paris"), "unit": S("Fahrenheit or Celsius")}, ["location"]),
    "maps_distance": _spec("maps_distance", "Driving distance and time between two places (OpenStreetMap, no key needed). A map of the route is shown in the chat.", {"origin": S("starting place/address"), "destination": S("destination place/address")}, ["origin", "destination"]),
    "maps_directions": _spec("maps_directions", "Turn-by-turn driving directions between two places, plus construction/roadworks on the route (OpenStreetMap, no key needed). A map of the route is shown in the chat — mention it in your answer.", {"origin": S("starting place/address"), "destination": S("destination place/address")}, ["origin", "destination"]),
    "maps_traffic": _spec("maps_traffic", "Live traffic along the driving route between two places. Needs a TomTom key (VIKI_TOMTOM_KEY); without one it says so plainly instead of guessing. A map of the route is shown in the chat.", {"origin": S("starting place/address"), "destination": S("destination place/address")}, ["origin", "destination"]),
    "calculate": _spec("calculate", "Evaluate a math expression exactly. Always use this for arithmetic.", {"expression": S("e.g. 18.5*12 + sqrt(49)")}, ["expression"]),
    "current_time": _spec("current_time", "Get the current local date and time."),
    "remember": _spec("remember", "Save a durable fact about the user (preference, project, name) for future sessions. Never save secrets.", {"fact": S("one short sentence")}, ["fact"]),
    "list_files": _spec("list_files", "List files in your workspace folder."),
    "read_file": _spec("read_file", "Read a text file from the workspace.", {"path": S("relative path")}, ["path"]),
    "write_file": _spec("write_file", "Create or overwrite a text file in the workspace (notes, drafts, reports). Needs user approval.",
        {"path": S("relative path"), "content": S("file text")}, ["path", "content"], risky=True),
    "home_assistant": _spec("home_assistant", "Control Home Assistant. Actions: list (compact entity list - call FIRST when unsure what exists), dump (FULL details of all entities with every attribute, or one entity if entity_id given), state (read one entity), call (run a service). Common domains/services: light.turn_on/turn_off/toggle (data: brightness_pct, color_name, rgb_color), switch.turn_on/turn_off/toggle, climate.set_temperature (data: temperature), media_player.play_media/volume_set/media_play_pause, script.turn_on, automation.trigger. No approval needed - act directly, EXCEPT call actions on lock, alarm_control_panel, or cover domains (door locks, alarms, blinds/garage doors), which need user approval.",
        {"action": S("list, state, dump, or call"), "entity_id": S("e.g. light.living_room_lamp (required for state/call; optional filter for dump; omit for list)"), "domain": S("e.g. light (required for call)"), "service": S("e.g. turn_on (required for call)"), "data": S("optional JSON object, e.g. {\"brightness_pct\": 50}")}, ["action"], risky_if=_ha_call_needs_approval),
    "gmail_search": _spec("gmail_search", "Search the user's Gmail. Returns message ids, senders, subjects, and snippets; use gmail_read for the full text.",
        {"query": S("Gmail search query, e.g. 'from:amazon subject:receipt'"), "max_results": S("max results (default 5)")}, ["query"]),
    "gmail_read": _spec("gmail_read", "Read a full Gmail message by id (get ids from gmail_search).",
        {"message_id": S("Gmail message id"), "max_chars": S("max body characters (default 6000)")}, ["message_id"]),
    "gmail_send": _spec("gmail_send", "Send an email from the user's Gmail account. Needs user approval.",
        {"to": S("recipient email address"), "subject": S("subject"), "body": S("message body")}, ["to", "subject", "body"], risky=True),
    "calendar_list": _spec("calendar_list", "List upcoming Google Calendar events.",
        {"days": S("how many days ahead (default 1)"), "max_results": S("max events (default 10)")}, []),
    "calendar_create": _spec("calendar_create", "Create a Google Calendar event. Needs user approval.",
        {"summary": S("event title"), "start": S("'YYYY-MM-DD HH:MM' local time"), "end": S("'YYYY-MM-DD HH:MM' local time"),
         "description": S("optional notes"), "location": S("optional location")}, ["summary", "start", "end"], risky=True),
    "github_search_repos": _spec("github_search_repos", "Search GitHub repositories by keyword.",
        {"query": S("search keywords"), "max_results": S("max results (default 5)")}, ["query"]),
    "github_search_code": _spec("github_search_code", "Search code across GitHub repositories.",
        {"query": S("code search query"), "max_results": S("max results (default 5)")}, ["query"]),
    "github_read_file": _spec("github_read_file", "Read a file from a GitHub repo.",
        {"repo": S("owner/name, e.g. 'octocat/hello-world'"), "path": S("file path in the repo"), "ref": S("optional branch/tag")}, ["repo", "path"]),
    "github_list_issues": _spec("github_list_issues", "List issues in a GitHub repo.",
        {"repo": S("owner/name"), "state": S("open, closed, or all (default open)"), "max_results": S("max results (default 10)")}, ["repo"]),
    "github_create_issue": _spec("github_create_issue", "Open a new issue in a GitHub repo. Needs user approval.",
        {"repo": S("owner/name"), "title": S("issue title"), "body": S("issue body (markdown ok)")}, ["repo", "title"], risky=True),
    "browser_goto": _spec("browser_goto", "Open a URL in VIKI's own browser. Returns the page title and a text preview.",
        {"url": S("full URL including https://")}, ["url"]),
    "browser_snapshot": _spec("browser_snapshot", "List the interactive elements on the current page as an accessibility tree with [ref=eN] handles. Use the refs with browser_click / browser_fill.",
        {}, []),
    "browser_text": _spec("browser_text", "Get the readable text of the current page.",
        {"max_chars": S("max characters (default 8000)")}, []),
    "browser_click": _spec("browser_click", "Click the element with the given ref from browser_snapshot. Needs user approval.",
        {"ref": S("element ref like e12")}, ["ref"], risky=True),
    "browser_fill": _spec("browser_fill", "Type text into the element with the given ref from browser_snapshot. Needs user approval.",
        {"ref": S("element ref like e12"), "text": S("text to type"), "submit": S("press Enter after typing: true/false")}, ["ref", "text"], risky=True),
    "browser_screenshot": _spec("browser_screenshot", "Save a screenshot of the current page under workspace/browser/.",
        {}, []),
    "announce": _spec("announce", "Speak text out loud on the Google Home Mini (plays audio in the home).",
        {"text": S("what to announce (max 500 chars)"), "device": S("optional device name to pick a specific speaker")}, ["text"]),
    "cast_devices": _spec("cast_devices", "List Google Cast devices (Home Mini etc.) visible on the LAN.",
        {}, []),
    "roku_status": _spec("roku_status", "What's on the Roku right now (state, app, what's playing).",
        {}, []),
    "roku_apps": _spec("roku_apps", "List apps installed on the Roku.",
        {}, []),
    "roku_launch": _spec("roku_launch", "Launch an app on the Roku by name.",
        {"app": S("app name, e.g. 'Netflix' — see roku_apps")}, ["app"]),
    "roku_search": _spec("roku_search", "Open Roku universal search and enter a keyword. Results appear on the TV.",
        {"keyword": S("movie, show, or app to search for")}, ["keyword"]),
    "roku_key": _spec("roku_key", "Press one Roku remote key.",
        {"key": S("up/down/left/right/select/back/home/play/volume_up/volume_down/volume_mute, or Lit_<text> to type")}, ["key"]),
    "roku_type": _spec("roku_type", "Type text with the Roku remote (for in-app search boxes).",
        {"text": S("text to type")}, ["text"]),
    "roku_play_movie": _spec("roku_play_movie", "Find a movie with Roku universal search and autoplay the top result. Best-effort: reports what the Roku is doing after; if it's wrong, steer with roku_key. If plex_search finds the title in his Plex library, prefer plex_play — it's exact.",
        {"title": S("movie title")}, ["title"]),
    "plex_libraries": _spec("plex_libraries", "List Plex media libraries (movies, shows, music).",
        {}, []),
    "plex_search": _spec("plex_search", "Search the Plex library for movies and episodes by title.",
        {"query": S("title to search for")}, ["query"]),
    "plex_on_deck": _spec("plex_on_deck", "What's on deck in Plex (continue watching).",
        {}, []),
    "plex_players": _spec("plex_players", "List Plex players currently seen on the network.",
        {}, []),
    "plex_play": _spec("plex_play", "Play the first Plex match for a title on a Plex player (e.g. the Roku).",
        {"title": S("movie or episode title"), "player": S("player name, e.g. 'Roku' (default: PLEX_PLAYER or first player)")}, ["title"]),
    "plex_pause": _spec("plex_pause", "Pause a Plex player.",
        {"player": S("player name (default: PLEX_PLAYER or first player)")}, []),
    "spotify_search": _spec("spotify_search", "Search Spotify for tracks, albums, artists, or playlists.",
        {"query": S("what to search for"), "kind": S("track/album/artist/playlist (default track)")}, ["query"]),
    "spotify_play": _spec("spotify_play", "Play the top Spotify track match.",
        {"query": S("song to play"), "device": S("device name (default: active device)")}, ["query"]),
    "spotify_play_uri": _spec("spotify_play_uri", "Play a Spotify URI (album/playlist/track).",
        {"uri": S("spotify:... URI"), "device": S("device name")}, ["uri"]),
    "spotify_pause": _spec("spotify_pause", "Pause Spotify.", {}, []),
    "spotify_resume": _spec("spotify_resume", "Resume Spotify playback.", {}, []),
    "spotify_next": _spec("spotify_next", "Skip to the next track.", {}, []),
    "spotify_previous": _spec("spotify_previous", "Go back a track.", {}, []),
    "spotify_devices": _spec("spotify_devices", "List devices Spotify can play on.", {}, []),
    "spotify_now_playing": _spec("spotify_now_playing", "What's currently playing on Spotify.", {}, []),
    "spotify_volume": _spec("spotify_volume", "Set Spotify volume (0-100).",
        {"percent": S("0-100")}, ["percent"]),
    "viki_check_update": _spec("viki_check_update", "Check if a newer VIKI release is available on GitHub. Safe, read-only.",
        {}, []),
    "viki_apply_update": _spec("viki_apply_update", "Download and apply the latest VIKI release, then restart. Your .env, tokens, and data are preserved. Needs user approval.",
        {}, [], risky=True),
}


GOOGLE_TOOLS = {"gmail_search", "gmail_read", "gmail_send", "calendar_list", "calendar_create"}
GITHUB_TOOLS = {"github_search_repos", "github_search_code", "github_read_file",
                "github_list_issues", "github_create_issue"}
BROWSER_TOOLS = {"browser_goto", "browser_snapshot", "browser_text",
                 "browser_click", "browser_fill", "browser_screenshot"}
ROKU_TOOLS = {"roku_status", "roku_apps", "roku_launch", "roku_search",
              "roku_key", "roku_type", "roku_play_movie"}
PLEX_TOOLS = {"plex_libraries", "plex_search", "plex_on_deck",
              "plex_players", "plex_play", "plex_pause"}
SPOTIFY_TOOLS = {"spotify_search", "spotify_play", "spotify_play_uri",
                 "spotify_pause", "spotify_resume", "spotify_next",
                 "spotify_previous", "spotify_devices", "spotify_now_playing",
                 "spotify_volume"}
UPDATE_TOOLS = {"viki_check_update", "viki_apply_update"}


def update_configured() -> bool:
    return bool(os.getenv("VIKI_UPDATE_REPO") and os.getenv("GITHUB_TOKEN"))


def viki_check_update() -> str:
    import updater
    st = updater.check()
    if not st["ok"]:
        return f"Update check failed: {st['error']}"
    if st["update_available"]:
        return (f"Update available: {st['local']} -> {st['latest']}.\n"
                f"Release notes: {st['notes'] or '(none)'}")
    return f"VIKI is up to date (v{st['local']})."


def viki_apply_update() -> str:
    import updater
    return updater.apply_update()


class Toolbox:
    def __init__(self, store):
        self.store = store
        self.funcs = {"web_search": web_search, "read_webpage": read_webpage, "get_weather": get_weather,
                      "maps_distance": maps_distance, "maps_directions": maps_directions,
                      "maps_traffic": maps_traffic,
                      "calculate": calculate, "current_time": current_time, "list_files": list_files,
                      "read_file": read_file, "write_file": write_file, "home_assistant": home_assistant,
                      "gmail_search": gmail_search, "gmail_read": gmail_read, "gmail_send": gmail_send,
                      "calendar_list": calendar_list, "calendar_create": calendar_create,
                      "github_search_repos": github_search_repos, "github_search_code": github_search_code,
                      "github_read_file": github_read_file, "github_list_issues": github_list_issues,
                      "github_create_issue": github_create_issue,
                      "browser_goto": browser_goto, "browser_snapshot": browser_snapshot,
                      "browser_text": browser_text, "browser_click": browser_click,
                      "browser_fill": browser_fill, "browser_screenshot": browser_screenshot,
                      "announce": home_announce, "cast_devices": cast_list_devices,
                      "roku_status": roku_status, "roku_apps": roku_apps,
                      "roku_launch": roku_launch, "roku_search": roku_search,
                      "roku_key": roku_key, "roku_type": roku_type,
                      "roku_play_movie": roku_play_movie,
                      "plex_libraries": plex_libraries, "plex_search": plex_search,
                      "plex_on_deck": plex_on_deck, "plex_players": plex_players,
                      "plex_play": plex_play, "plex_pause": plex_pause,
                      "spotify_search": spotify_search, "spotify_play": spotify_play,
                      "spotify_play_uri": spotify_play_uri, "spotify_pause": spotify_pause,
                      "spotify_resume": spotify_resume, "spotify_next": spotify_next,
                      "spotify_previous": spotify_previous, "spotify_devices": spotify_devices,
                      "spotify_now_playing": spotify_now_playing,
                      "spotify_volume": spotify_volume,
                      "remember": lambda fact: "Remembered." if self.store.add_facts([fact], source="agent") else "Not saved (duplicate or sensitive).",
                      "update_plan": lambda steps: "Plan updated.",
                      "viki_check_update": viki_check_update,
                      "viki_apply_update": viki_apply_update}
        self.enabled = [n for n in SPECS
                        if (n != "home_assistant" or (os.getenv("HA_URL") and os.getenv("HA_TOKEN")))
                        and (n not in GOOGLE_TOOLS or google_connected())
                        and (n not in GITHUB_TOOLS or github_connected())
                        and (n not in ROKU_TOOLS or ha_connected())
                        and (n not in PLEX_TOOLS or plex_configured())
                        and (n not in SPOTIFY_TOOLS or spotify_connected())
                        and (n not in UPDATE_TOOLS or update_configured())]

    def schemas(self):
        return [SPECS[n]["schema"] for n in self.enabled]

    def needs_approval(self, name, args):
        spec = SPECS.get(name, {})
        if spec.get("risky"):
            return True
        cond = spec.get("risky_if")
        return bool(cond(args or {}) if cond else False)

    def run(self, name, args):
        fn = self.funcs.get(name) if name in self.enabled else None
        if not fn:
            return f"Unknown tool: {name}"
        # Per-tool timeout: a hung network/browser tool must not stall the
        # agent loop forever. Override with VIKI_TOOL_TIMEOUT (seconds).
        try:
            timeout = float(os.getenv("VIKI_TOOL_TIMEOUT", "90"))
        except (TypeError, ValueError):
            timeout = 90.0
        if timeout <= 0:
            timeout = 90.0
        box = {}

        def _target():
            try:
                box["result"] = str(fn(**args))
            except TypeError as e:
                box["error"] = f"Bad arguments for {name}: {e}"
            except Exception as e:
                box["error"] = f"{name} failed: {e}"

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            # The thread keeps running in the background (daemon) but the
            # agent gets a graceful, actionable answer instead of hanging.
            return (f"{name} timed out after {timeout:g}s — the service may "
                    "be down or slow. Try again, or do something else.")
        if "error" in box:
            return box["error"]
        return box.get("result", "")
