"""Smart-announcement guard for the Google Home Mini (v0.2.24).

Joey's rule: VIKI may announce things out loud (urgent mail, welcome-home
greetings), but she must never nag. This module is the anti-spam brain
behind that rule — pure logic, no network calls, so it is cheap to test
and safe to call from any announcement path:

- ``fingerprint``: a stable ID for a message, so "Urgent mail from X"
  announced once is never announced again, even if reworded slightly
  (case/whitespace differences collapse to the same fingerprint).
- ``AnnouncementLog``: a small JSON record of what was already spoken,
  kept under ``workspace/`` (never committed). Entries older than 30 days
  are pruned on save so the file stays tiny.
- ``welcome_home_due``: the welcome-home greeting fires only when
  Joey's phone is home AND the lights are on AND it is after 3pm on a
  weekday — and only once per day.

Wired into ``connectors/home_mini.py`` ``announce()`` since v0.2.26
(dedupe on by default; only successful announcements are recorded).
Detecting phone-home + lights-on for the welcome-home trigger itself is
still a later step; this module deliberately has no side effects on import.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

# How long an announced message stays "already said" (days).
RETENTION_DAYS = 30

# Welcome-home greeting rules (Joey, 2026-10-23): weekdays, after 3pm.
WELCOME_AFTER_HOUR = 15

_DEFAULT_PATH = Path(__file__).resolve().parent / "workspace" / "announced.json"

_WELCOME_KEY = "__welcome_home__"


def fingerprint(text: str) -> str:
    """Stable short ID for a message.

    Lowercases, collapses whitespace, and strips punctuation so trivial
    rewordings of the same announcement dedupe to one fingerprint.
    """
    cleaned = re.sub(r"[^a-z0-9 ]+", "", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]


class AnnouncementLog:
    """Persistent record of announced messages and welcome greetings."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else _DEFAULT_PATH
        self._data = self._load()

    # -- storage ---------------------------------------------------------
    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            pass
        return {"announced": {}, "welcome_days": []}

    def _save(self) -> None:
        self._prune()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def _prune(self, now: datetime | None = None) -> None:
        """Drop announced entries older than RETENTION_DAYS."""
        now = now or datetime.now()
        kept = {}
        for fp, stamp in self._data.get("announced", {}).items():
            try:
                when = datetime.fromisoformat(stamp)
            except (TypeError, ValueError):
                continue
            if (now - when).days < RETENTION_DAYS:
                kept[fp] = stamp
        self._data["announced"] = kept

    # -- announcements ---------------------------------------------------
    def already_announced(self, text: str) -> bool:
        """True if this message (or a trivial rewording) was spoken before."""
        return fingerprint(text) in self._data.get("announced", {})

    def mark_announced(self, text: str, when: datetime | None = None) -> None:
        """Record a message as spoken and persist the log."""
        when = when or datetime.now()
        self._data.setdefault("announced", {})[fingerprint(text)] = when.isoformat()
        self._save()

    def gate(self, text: str) -> tuple[bool, str]:
        """Allow/deny helper: (True, "ok") or (False, reason)."""
        if self.already_announced(text):
            return False, "already announced"
        return True, "ok"

    # -- welcome-home greeting -------------------------------------------
    def welcome_home_due(
        self,
        now: datetime,
        phone_home: bool,
        lights_on: bool,
    ) -> bool:
        """True only for a weekday after 3pm, phone home, lights on,
        and no welcome greeting recorded yet today."""
        if not phone_home or not lights_on:
            return False
        if now.weekday() >= 5:  # Saturday/Sunday: no greeting
            return False
        if now.hour < WELCOME_AFTER_HOUR:
            return False
        return now.date().isoformat() not in self._data.get("welcome_days", [])

    def mark_welcome_home(self, when: datetime | None = None) -> None:
        """Record today's welcome greeting so it fires at most once a day."""
        when = when or datetime.now()
        days = self._data.setdefault("welcome_days", [])
        day = when.date().isoformat()
        if day not in days:
            days.append(day)
        # Keep only recent days; the list is tiny, but never let it grow
        # without bound if the clock ever jumps around.
        self._data["welcome_days"] = sorted(days)[-90:]
        self._save()
