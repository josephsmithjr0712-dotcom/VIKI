"""Roku control for VIKI through Home Assistant.

Setup: HA_URL + HA_TOKEN in .env (same as the home_assistant tool), with the
Roku integration added in Home Assistant (Settings > Devices & services > Roku,
or auto-discovered). Entity IDs are auto-discovered — any media_player/remote
with 'roku' in the name; override with ROKU_ENTITY / ROKU_REMOTE.

Tools:
  roku_status()          - what's on the Roku now (state, app, what's playing)
  roku_apps()            - installed apps
  roku_launch(app)       - launch an app by name
  roku_search(keyword)   - open Roku universal search and enter a keyword
  roku_key(key)          - press one remote key (up/down/left/right/select/...)
  roku_type(text)        - type text with the remote
  roku_play_movie(title) - search for a movie and autoplay the top result
"""
import json
import os
import time
import urllib.request

# Remote keys supported by the HA Roku integration (docs) + Lit_ typing.
VALID_KEYS = {
    "back", "backspace", "channel_down", "channel_up", "down", "enter",
    "find_remote", "forward", "home", "info", "input_av1", "input_hdmi1",
    "input_hdmi2", "input_hdmi3", "input_hdmi4", "input_tuner", "left",
    "play", "power", "replay", "reverse", "right", "search", "select",
    "up", "volume_down", "volume_mute", "volume_up",
}

_entities_cache = None


def ha_connected() -> bool:
    return bool(os.getenv("HA_URL") and os.getenv("HA_TOKEN"))


def _ha():
    base = os.getenv("HA_URL", "").rstrip("/")
    token = os.getenv("HA_TOKEN", "")
    if not base or not token:
        raise RuntimeError(
            "Home Assistant is not connected. Set HA_URL and HA_TOKEN in .env, "
            "and add the Roku integration in Home Assistant."
        )
    return base, token


def _service(domain: str, service: str, data: dict):
    base, token = _ha()
    req = urllib.request.Request(
        f"{base}/api/services/{domain}/{service}",
        data=json.dumps(data).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def _state(entity_id: str) -> dict:
    base, token = _ha()
    req = urllib.request.Request(
        f"{base}/api/states/{entity_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def _entities():
    """(media_player entity, remote entity) — auto-discovered, cached per process."""
    global _entities_cache
    if _entities_cache:
        return _entities_cache
    base, token = _ha()
    req = urllib.request.Request(
        f"{base}/api/states", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        states = json.load(r)
    mp = [s["entity_id"] for s in states
          if s["entity_id"].startswith("media_player.") and "roku" in s["entity_id"]]
    rem = [s["entity_id"] for s in states
           if s["entity_id"].startswith("remote.") and "roku" in s["entity_id"]]
    entity = os.getenv("ROKU_ENTITY") or (mp[0] if mp else None)
    remote = os.getenv("ROKU_REMOTE") or (rem[0] if rem else None)
    if not entity:
        raise RuntimeError(
            "No Roku media_player found in Home Assistant. Add the Roku "
            "integration (Settings > Devices & services) and try again."
        )
    _entities_cache = (entity, remote)
    return _entities_cache


def reset_cache():
    global _entities_cache
    _entities_cache = None


def _run(fn):
    try:
        return fn()
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        return f"Roku action failed: {type(e).__name__}: {e}"


def _status_text(entity):
    st = _state(entity)
    attrs = st.get("attributes", {})
    lines = [f"State: {st.get('state')}",
             f"App: {attrs.get('app_name') or attrs.get('source') or '?'}"]
    if attrs.get("media_title"):
        lines.append(f"Now playing: {attrs['media_title']}")
    return "\n".join(lines)


def roku_status():
    def go():
        entity, _ = _entities()
        return _status_text(entity) + f"\nEntity: {entity}"
    return _run(go)


def roku_apps():
    def go():
        entity, _ = _entities()
        apps = _state(entity).get("attributes", {}).get("source_list") or []
        if not apps:
            return "No app list available from Home Assistant."
        return "Installed apps:\n" + "\n".join(f"- {a}" for a in apps)
    return _run(go)


def roku_launch(app):
    def go(name=app):
        entity, _ = _entities()
        _service("media_player", "select_source",
                 {"entity_id": entity, "source": name})
        return f"Launching '{name}' on the Roku."
    return _run(go)


def roku_search(keyword):
    def go(kw=keyword):
        entity, _ = _entities()
        kw = (kw or "").strip()
        if not kw:
            return "Give me a keyword to search for."
        _service("roku", "search", {"entity_id": entity, "keyword": kw})
        return f"Roku search opened for '{kw}' — results should be on screen now."
    return _run(go)


def roku_key(key):
    def go(k=key):
        _, remote = _entities()
        if not remote:
            return ("No Roku remote entity found in Home Assistant "
                    "(set ROKU_REMOTE or check the integration).")
        k = (k or "").strip()
        if not k or (k not in VALID_KEYS and not k.startswith("Lit_")):
            return ("Unknown key. Valid keys: " + ", ".join(sorted(VALID_KEYS))
                    + ", or Lit_<text> to type.")
        _service("remote", "send_command", {"entity_id": remote, "command": [k]})
        return f"Pressed '{k}'."
    return _run(go)


def roku_type(text):
    """Type text with the Roku remote (for in-app search boxes)."""
    def go(t=text):
        _, remote = _entities()
        if not remote:
            return ("No Roku remote entity found in Home Assistant "
                    "(set ROKU_REMOTE or check the integration).")
        t = (t or "").strip()
        if not t:
            return "Give me text to type."
        _service("remote", "send_command",
                 {"entity_id": remote, "command": [f"Lit_{t}"]})
        return f"Typed '{t}'."
    return _run(go)


def roku_play_movie(title):
    """Find a movie with Roku universal search and autoplay the top result.

    Best-effort: search results can't be read back programmatically, so this
    opens search, navigates to the first result and presses play, then reports
    what the Roku says it's doing. If it's wrong, tell me what you see on
    screen and I'll steer with roku_key.
    """
    def go(t=title):
        entity, _ = _entities()
        t = (t or "").strip()
        if not t:
            return "Give me a movie title."
        log = [f"Searching Roku for '{t}'..."]
        _service("roku", "search", {"entity_id": entity, "keyword": t})
        time.sleep(4)
        _, remote = _entities()
        if not remote:
            return ("\n".join(log) + "\nSearch is on screen, but I have no remote "
                    "entity to navigate with (set ROKU_REMOTE).")
        for key, wait in [("down", 1.0), ("select", 2.5), ("select", 2.0), ("play", 3.0)]:
            _service("remote", "send_command", {"entity_id": remote, "command": [key]})
            time.sleep(wait)
        log.append("Autoplay sequence sent (down → select → select → play).")
        log.append(_status_text(entity))
        log.append("If that's not your movie, tell me what you see and I'll steer.")
        return "\n".join(log)
    return _run(go)
