"""Google Home Mini announcements via Home Assistant.

Speaks out loud on the Mini using HA's media_player.play_media service.
Audio is rendered with Edge TTS (Aria US female by default — set
VIKI_TTS_VOICE in .env to change) and served over HTTP from this machine.

    announce("Dinner's ready")                  # auto-find the Mini
    announce("Time to leave", device="kitchen") # pick by name fragment

Setup: HA_URL + HA_TOKEN in .env (already configured), and the Mini
must be visible in Home Assistant as a media_player (Google Cast
integration). Override the entity with MINI_ENTITY in .env.

Falls back to direct Cast discovery (pychromecast) if HA is unreachable.

No-repeat guard: announce() dedupes through announcements.AnnouncementLog
by default, so the same message is never spoken twice — a failed attempt
is NOT recorded, so it can be retried. Pass dedupe=False to repeat a
message on purpose.
"""
import json
import os
import socket
import threading
import time
import urllib.request
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    # announcements.py lives at the repo root; if home_mini is imported
    # without it on sys.path, dedupe just degrades to off (never a crash).
    from announcements import AnnouncementLog
except Exception:  # pragma: no cover - import-path fallback
    AnnouncementLog = None

HERE = Path(__file__).resolve().parent
ANNOUNCE_DIR = HERE.parent / "workspace" / "announcements"
PORT = int(os.getenv("VIKI_ANNOUNCE_PORT", "7871"))

_server = None
_server_lock = threading.Lock()


def _lan_ip() -> str:
    """The LAN address other devices on the network can reach us at."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no traffic is sent
        return s.getsockname()[0]
    finally:
        s.close()


def _ensure_server() -> int:
    """Serve ANNOUNCE_DIR over HTTP on PORT (started once per process)."""
    global _server
    with _server_lock:
        if _server is None:
            ANNOUNCE_DIR.mkdir(parents=True, exist_ok=True)
            handler = partial(SimpleHTTPRequestHandler, directory=str(ANNOUNCE_DIR))
            _server = ThreadingHTTPServer(("0.0.0.0", PORT), handler)
            t = threading.Thread(target=_server.serve_forever, daemon=True)
            t.start()
        return PORT


# --- Home Assistant --------------------------------------------------------

def _ha():
    base = os.getenv("HA_URL", "").rstrip("/")
    token = os.getenv("HA_TOKEN", "")
    if not base or not token:
        raise RuntimeError("Home Assistant not configured (HA_URL/HA_TOKEN in .env).")
    return base, token


def _ha_service(domain: str, service: str, data: dict):
    base, token = _ha()
    req = urllib.request.Request(
        f"{base}/api/services/{domain}/{service}",
        data=json.dumps(data).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def _ha_states():
    base, token = _ha()
    req = urllib.request.Request(
        f"{base}/api/states",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def find_mini_entity(name_hint: str | None = None) -> str | None:
    """Find the Home Mini's media_player entity in HA."""
    override = os.getenv("MINI_ENTITY", "").strip()
    if override:
        return override
    try:
        states = _ha_states()
    except Exception:
        return None
    players = [s for s in states if s["entity_id"].startswith("media_player.")]
    if name_hint:
        hint = name_hint.lower()
        for s in players:
            label = (s["entity_id"] + " " + str(s["attributes"].get("friendly_name", ""))).lower()
            if hint in label:
                return s["entity_id"]
        return None
    # Prefer something that looks like a Nest/Home/Mini speaker
    for s in players:
        label = (s["entity_id"] + " " + str(s["attributes"].get("friendly_name", ""))).lower()
        if any(k in label for k in ("mini", "nest", "home mini", "google home")):
            return s["entity_id"]
    return players[0]["entity_id"] if players else None


def _wait_until_idle(entity: str, timeout: float = 60) -> None:
    """Block until the HA media_player stops playing (or timeout). So the
    mic doesn't hear the Mini talking. Never raises."""
    try:
        base, token = _ha()
    except Exception:
        return  # HA not configured: nothing to poll
    deadline = time.time() + timeout
    # Give the Mini a moment to start playing before we check
    time.sleep(2)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{base}/api/states/{entity}",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                state = json.load(r).get("state", "")
            if state not in ("playing", "buffering"):
                return
        except Exception:
            return  # if we can't poll, don't block forever
        time.sleep(1)


def _announce_ha(text: str, device: str | None, wait: bool = True) -> str:
    """Announce via Home Assistant. Raises on failure.
    wait=True blocks until the Mini finishes speaking (mic stays paused)."""
    from connectors.edge_tts import speak as edge_speak

    entity = find_mini_entity(device)
    if not entity:
        raise RuntimeError(
            "No speaker found in Home Assistant. Is the Google Cast integration "
            "set up and the Mini online? Set MINI_ENTITY in .env to override."
        )
    # Render with Edge TTS (Aria voice) and serve it
    mp3 = edge_speak(text)
    if not mp3:
        raise RuntimeError("TTS produced no audio.")
    ANNOUNCE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"announce-{datetime.now():%Y%m%d-%H%M%S}.mp3"
    (ANNOUNCE_DIR / fname).write_bytes(mp3)
    port = _ensure_server()
    url = f"http://{_lan_ip()}:{port}/{fname}"
    _ha_service("media_player", "play_media", {
        "entity_id": entity,
        "media_content_id": url,
        "media_content_type": "music",
    })
    if wait:
        _wait_until_idle(entity)
    return f"Announced on '{entity}' via Home Assistant."


# --- Direct Cast fallback --------------------------------------------------

def _pychromecast():
    try:
        import pychromecast
    except ImportError:
        raise RuntimeError("pychromecast is not installed. Run: pip install pychromecast")
    return pychromecast


def list_devices(timeout=5) -> str:
    pcc = _pychromecast()
    chromecasts, browser = pcc.get_chromecasts(timeout=timeout)
    try:
        if not chromecasts:
            return "No Cast devices found on the LAN."
        return "Cast devices:\n" + "\n".join(f"- {cc.name} ({cc.host})" for cc in chromecasts)
    finally:
        browser.stop_discovery()


def _announce_cast(text: str, device: str | None, timeout: float = 30) -> str:
    """Announce via direct Cast discovery (fallback when HA is down)."""
    from connectors.edge_tts import speak as edge_speak

    pcc = _pychromecast()
    mp3 = edge_speak(text)
    ANNOUNCE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"announce-{datetime.now():%Y%m%d-%H%M%S}.mp3"
    (ANNOUNCE_DIR / fname).write_bytes(mp3)
    port = _ensure_server()
    url = f"http://{_lan_ip()}:{port}/{fname}"
    chromecasts, browser = pcc.get_chromecasts(timeout=8)
    try:
        if not chromecasts:
            return "No Cast devices found. Is the Home Mini on and on the same WiFi?"
        if device:
            match = [c for c in chromecasts if device.lower() in c.name.lower()]
            if not match:
                return f"No Cast device matching '{device}'. " + list_devices()
            cast = match[0]
        else:
            cast = chromecasts[0]
        cast.wait(timeout=10)
        mc = cast.media_controller
        mc.play_media(url, "audio/mp3")
        mc.block_until_active(timeout=10)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not mc.status.player_is_playing and mc.status.player_state in ("IDLE", "UNKNOWN"):
                break
            time.sleep(0.5)
        return f"Announced on '{cast.name}' (direct Cast)."
    except Exception as e:
        return f"Announcement failed: {type(e).__name__}: {e}"
    finally:
        browser.stop_discovery()


# --- Public API ------------------------------------------------------------

def announce(text: str, device: str | None = None, timeout: float = 30,
           wait: bool = True, dedupe: bool = True, log=None) -> str:
    """Speak text out loud on the Google Home Mini. Returns a status string.

    Tries Home Assistant first (reliable, on your LAN), falls back to
direct Cast discovery if HA is unreachable.
    wait=True (default) blocks until the Mini finishes speaking — the
voice loop uses this so the mic stays paused during announcements.
    dedupe=True (default) skips a message already spoken (checked via
AnnouncementLog; pass an explicit log in tests, or dedupe=False to
repeat a message on purpose). Only a successful announcement is
recorded — a failed attempt is never marked, so it can be retried.
    """
    text = (text or "").strip()
    if not text:
        return "Nothing to announce."
    if len(text) > 500:
        return "Announcement too long (500 char max) — keep it short."
    log_obj = None
    if dedupe:
        if log is not None:
            log_obj = log
        elif AnnouncementLog is not None:
            try:
                log_obj = AnnouncementLog()
            except Exception:
                log_obj = None
        if log_obj is not None and log_obj.already_announced(text):
            return "Already announced — skipped (no-repeat guard)."
    try:
        result = _announce_ha(text, device, wait=wait)
    except Exception as e:
        ha_err = f"{type(e).__name__}: {e}"
        try:
            result = _announce_cast(text, device, timeout)
        except Exception as e2:
            return f"HA failed ({ha_err}); Cast fallback also failed ({type(e2).__name__}: {e2})"
    if log_obj is not None and result.startswith("Announced on"):
        try:
            log_obj.mark_announced(text)
        except Exception:
            pass
    return result
