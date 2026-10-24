"""Presence detection for VIKI's smart announcements (v0.2.29).

This is the wiring piece the announcement guard (announcements.py)
was waiting on: actually detecting whether Joey is home, so the
welcome-home greeting and "announce only when he's home" rules have
real inputs instead of placeholders.

Two signals, both fail-soft (a failed check reads as "not home",
never as a crash, so VIKI never announces on a guess):

- ``phone_home``: Joey's phone on the home network. Preferred path is
  Home Assistant: a ``device_tracker`` entity for his phone reporting
  ``home`` (pin it with ``VIKI_PHONE_TRACKER`` in .env, e.g.
  ``device_tracker.joeys_phone``). Fallback is a single ping to the
  phone's Tailscale IP (``VIKI_PHONE_IP`` in .env) — the phone answers
  ping while awake on the home network.
- ``lights_on``: any ``light.*`` entity ``on`` in Home Assistant
  (pin one lamp with ``LIGHT_ENTITY`` in .env).

``welcome_home_check`` ties it together with the guard: weekday,
after 3pm, phone home, lights on, and no greeting yet today.

Home Assistant access reuses the ``HA_URL`` + ``HA_TOKEN`` .env pair
from connectors/home_mini.py. All I/O is injectable (``states``,
``ping``) so the logic is fully testable without a network.
"""

import json
import os
import subprocess
import urllib.request
from datetime import datetime

try:
    from announcements import AnnouncementLog
except Exception:  # pragma: no cover - import-path fallback, like home_mini
    AnnouncementLog = None


# --- Home Assistant -------------------------------------------------------

def _ha_states():
    """Fetch HA states, or None if HA is unconfigured/unreachable."""
    base = os.getenv("HA_URL", "").rstrip("/")
    token = os.getenv("HA_TOKEN", "")
    if not base or not token:
        return None
    try:
        req = urllib.request.Request(
            f"{base}/api/states",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            states = json.load(r)
        return states if isinstance(states, list) else None
    except Exception:
        return None


def _states_or_fetch(states):
    """Use caller-supplied states, else try HA once, else None."""
    if states is not None:
        return states
    return _ha_states()


def _friendly(state):
    attrs = state.get("attributes") or {}
    return str(attrs.get("friendly_name", ""))


# --- phone home -----------------------------------------------------------

def _tracker_match(state):
    """True if this device_tracker looks like Joey's phone."""
    override = os.getenv("VIKI_PHONE_TRACKER", "").strip().lower()
    entity = str(state.get("entity_id", "")).lower()
    label = (entity + " " + _friendly(state)).lower()
    if override:
        return override in label
    return "phone" in label or "joey" in label


def phone_home(states=None, ping=None, phone_ip=None):
    """True if Joey's phone looks home.

    Order: HA device_tracker first, then a ping to VIKI_PHONE_IP.
    ``states``/``ping`` are injectable for tests: ``ping`` is a
    callable taking an IP and returning bool. Returns False (never
    raises) when nothing is configured or every check fails.
    """
    states = _states_or_fetch(states)
    if states:
        for s in states:
            if not str(s.get("entity_id", "")).startswith("device_tracker."):
                continue
            if _tracker_match(s):
                return str(s.get("state", "")).lower() == "home"
    ip = phone_ip or os.getenv("VIKI_PHONE_IP", "").strip()
    if not ip:
        return False
    if ping is not None:
        try:
            return bool(ping(ip))
        except Exception:
            return False
    try:
        r = subprocess.run(
            ["ping", "-c", "1", "-W", "2", ip],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


# --- lights on ------------------------------------------------------------

def lights_on(states=None, entity=None):
    """True if the house lights look on.

    ``LIGHT_ENTITY`` (or ``entity``) pins one lamp; otherwise any
    ``light.*`` entity in state ``on`` counts. Never raises.
    """
    states = _states_or_fetch(states)
    if not states:
        return False
    want = (entity or os.getenv("LIGHT_ENTITY", "")).strip().lower()
    for s in states:
        if not str(s.get("entity_id", "")).startswith("light."):
            continue
        if want and want not in str(s.get("entity_id", "")).lower():
            continue
        if str(s.get("state", "")).lower() == "on":
            return True
    return False


# --- combined -------------------------------------------------------------

def home_status(states=None, ping=None, phone_ip=None):
    """{'phone_home': bool, 'lights_on': bool, 'ready': bool}.

    ``ready`` is the announce-when-home gate: both signals true.
    ``states`` is fetched once and shared by both checks.
    """
    states = _states_or_fetch(states)
    ph = phone_home(states=states, ping=ping, phone_ip=phone_ip)
    lo = lights_on(states=states)
    return {"phone_home": ph, "lights_on": lo, "ready": bool(ph and lo)}


def welcome_home_check(now=None, log=None, states=None, ping=None,
                       phone_ip=None):
    """Should VIKI greet Joey with the welcome-home message now?

    Returns (True, "ok") or (False, reason). The once-per-day rule
    lives in announcements.AnnouncementLog; pass a log in tests, or
    one is created (and the greeting is NOT marked here — the caller
    marks it via log.mark_welcome_home() after actually speaking, so
    a failed announcement doesn't burn the day's greeting).
    """
    now = now or datetime.now()
    status = home_status(states=states, ping=ping, phone_ip=phone_ip)
    if not status["phone_home"]:
        return False, "phone not home"
    if not status["lights_on"]:
        return False, "lights off"
    if log is None:
        if AnnouncementLog is None:
            return False, "announcement log unavailable"
        try:
            log = AnnouncementLog()
        except Exception:
            return False, "announcement log unavailable"
    if not log.welcome_home_due(now, True, True):
        if now.weekday() >= 5:
            return False, "weekend: no greeting"
        if now.hour < 15:
            return False, "before 3pm"
        return False, "already greeted today"
    return True, "ok"
