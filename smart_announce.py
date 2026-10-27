"""Presence-gated smart announcements for VIKI (v0.2.31).

This is the wiring piece that ties the three announcement parts
together into one call a scheduler job (or any proactive path) can
use:

- ``presence.home_status`` — is Joey actually home (phone home AND
  lights on)?
- ``announcements.AnnouncementLog`` — was this already said (never
  repeat), and was today's welcome-home greeting already spoken?
- ``connectors.home_mini.announce`` — the actual speaking.

Joey's rules, enforced here (fail-soft throughout: anything unknown
reads as "don't speak", never as a guess):

- A proactive announcement only fires when phone-home + lights-on.
  An explicit "announce this now" from Joey is NOT routed through
  here — that uses home_mini.announce directly.
- A message is never spoken twice (dedupe by fingerprint).
- The welcome-home greeting fires at most once per day, weekdays
  after 3pm, and only when he's home.
- A failed announcement is never recorded, so it can be retried.
- Quiet hours (v0.3.14): no proactive announcement speaks between
  10pm and 7am — a skipped announcement is never recorded, so it can
  speak later. Override with VIKI_QUIET_HOURS="22-7" (any "start-end"
  hour pair, overnight wrap allowed); "off" disables the guard.

All I/O is injectable (``announce_fn``, ``log``, ``states``,
``ping``), so the whole module is testable with no network, no Home
Assistant, and no real speaker. ``announce_fn`` takes the text and
returns either True or a status string; success is a True return or
a string starting with "Announced" (the home_mini.announce
convention) — anything else counts as a failure and is not recorded.
"""

import os
from datetime import datetime

try:
    from announcements import AnnouncementLog
except Exception:  # pragma: no cover - import-path fallback, like home_mini
    AnnouncementLog = None

try:
    from presence import home_status, welcome_home_check
except Exception:  # pragma: no cover - import-path fallback
    home_status = None
    welcome_home_check = None

WELCOME_HOME_TEXT = "Welcome home, boss."

# Quiet hours (v0.3.14): default 10pm–7am, matching Joey's 7am shift
# start. VIKI_QUIET_HOURS="start-end" overrides (e.g. "21-6");
# "off" disables the guard entirely.
DEFAULT_QUIET_START_HOUR = 22
DEFAULT_QUIET_END_HOUR = 7


def _quiet_hours():
    """(start, end) quiet-hour pair from env, or None if disabled.

    Malformed values fall back to the default window rather than
    silently disabling the guard — a typo should not wake the house.
    A start equal to the end means an empty window (disabled).
    """
    raw = os.getenv("VIKI_QUIET_HOURS", "").strip().lower()
    if raw in ("off", "none", "disabled"):
        return None
    if not raw:
        return (DEFAULT_QUIET_START_HOUR, DEFAULT_QUIET_END_HOUR)
    try:
        start_s, end_s = raw.split("-", 1)
        start, end = int(start_s.strip()), int(end_s.strip())
    except ValueError:
        return (DEFAULT_QUIET_START_HOUR, DEFAULT_QUIET_END_HOUR)
    if not (0 <= start <= 23 and 0 <= end <= 23):
        return (DEFAULT_QUIET_START_HOUR, DEFAULT_QUIET_END_HOUR)
    if start == end:
        return None
    return (start, end)


def is_quiet_hours(now=None):
    """True if ``now`` (default: right now) falls in quiet hours.

    The window wraps midnight when start > end (22–7 means 22:00
    through 06:59); otherwise it is a plain [start, end) range.
    """
    window = _quiet_hours()
    if window is None:
        return False
    start, end = window
    hour = (now or datetime.now()).hour
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end

# Scheduler job actions understood by run_smart_job.
WELCOME_HOME_ACTION = "welcome-home"
ANNOUNCE_ACTION_PREFIX = "announce-if-home:"


def _success(result) -> bool:
    """True if an announce_fn result means 'actually spoken'."""
    if result is True:
        return True
    return isinstance(result, str) and result.startswith("Announced")


def _default_announce(text: str) -> str:
    """Speak via home_mini, with its own dedupe off.

    smart_announce owns the gate + record here, so home_mini must not
    double-gate (and must not mark the message on its own path).
    """
    from connectors.home_mini import announce as home_announce

    return home_announce(text, dedupe=False)


def _get_log(log):
    """Caller-supplied log, else the default on-disk log, else None."""
    if log is not None:
        return log
    if AnnouncementLog is None:
        return None
    try:
        return AnnouncementLog()
    except Exception:
        return None


def announce_if_home(text, announce_fn=None, log=None, states=None,
                     ping=None, phone_ip=None, now=None):
    """Speak ``text`` only if Joey is home and it wasn't said before.

    Returns (True, result) on a spoken announcement, or (False, reason)
    when skipped or failed. Reasons: "nothing to announce",
    "presence unavailable", "phone not home", "lights off",
    "quiet hours", "already announced", "announce failed: ...", or
    the speaker's failure string. A failure — and a quiet-hours skip
    — is never recorded, so it can be retried (and spoken later).
    ``now`` defaults to right now; callers that need a deterministic
    check (tests, the scheduler) pass it explicitly.
    """
    text = (text or "").strip()
    if not text:
        return False, "nothing to announce"
    if home_status is None:
        return False, "presence unavailable"
    try:
        status = home_status(states=states, ping=ping, phone_ip=phone_ip)
    except Exception:
        return False, "presence unavailable"
    if not status.get("phone_home"):
        return False, "phone not home"
    if not status.get("lights_on"):
        return False, "lights off"
    if is_quiet_hours(now=now):
        return False, "quiet hours"
    log_obj = _get_log(log)
    if log_obj is not None and log_obj.already_announced(text):
        return False, "already announced"
    fn = announce_fn or _default_announce
    try:
        result = fn(text)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed silently
        return False, f"announce failed: {type(exc).__name__}: {exc}"
    if not _success(result):
        return False, str(result)
    if log_obj is not None:
        try:
            log_obj.mark_announced(text)
        except Exception:
            pass
    return True, str(result)


def announce_brief(brief, announce_fn=None, log=None, states=None,
                   ping=None, phone_ip=None, now=None):
    """Speak a scheduled connection-check brief, if Joey is home.

    This is the morning-brief path: a scheduled connection-check job
    saves its brief for the Settings screen AND speaks it on the Home
    Mini — but only through the same gate as every other proactive
    announcement (phone home + lights on, never repeated). A manual
    /api/brief/refresh does NOT call this: when Joey refreshes by hand
    he is looking at the screen, so speaking would just be noise.

    The brief's newlines collapse to spaces for speech, behind a short
    header. Returns announce_if_home's (ok, reason) tuple; an empty
    brief never speaks.
    """
    text = " ".join((brief or "").split())
    if not text:
        return False, "nothing to announce"
    spoken = "Here's your brief. " + text
    return announce_if_home(spoken, announce_fn=announce_fn, log=log,
                            states=states, ping=ping, phone_ip=phone_ip,
                            now=now)


def welcome_home(announce_fn=None, log=None, now=None, states=None,
                 ping=None, phone_ip=None):
    """Speak the welcome-home greeting if it is due, at most once a day.

    Returns (True, result) when spoken, or (False, reason) when not
    due ("phone not home", "lights off", "before 3pm", "weekend: no
    greeting", "already greeted today") or when speaking failed. A
    failed greeting does NOT burn the day's greeting.
    """
    if welcome_home_check is None:
        return False, "presence unavailable"
    log_obj = _get_log(log)
    try:
        due, reason = welcome_home_check(now=now, log=log_obj,
                                         states=states, ping=ping,
                                         phone_ip=phone_ip)
    except Exception:
        return False, "presence unavailable"
    if not due:
        return False, reason
    if is_quiet_hours(now=now):
        # Quiet hours do NOT burn the day's greeting: it can still
        # speak later (the due check re-runs on the next tick).
        return False, "quiet hours"
    fn = announce_fn or _default_announce
    try:
        result = fn(WELCOME_HOME_TEXT)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed silently
        return False, f"announce failed: {type(exc).__name__}: {exc}"
    if not _success(result):
        return False, str(result)
    if log_obj is not None:
        try:
            log_obj.mark_welcome_home(when=now)
        except Exception:
            pass
    return True, str(result)


def run_smart_job(job, announce_fn=None, log=None, now=None,
                  states=None, ping=None, phone_ip=None):
    """Scheduler-executor entry point for smart-announcement jobs.

    Job actions:
      - "welcome-home" -> the welcome-home greeting (once per day).
      - "announce-if-home:<text>" -> that text, only when Joey is home.
    Returns a short status string for the job's record. Unknown
    actions return an explanatory string instead of raising, so one
    bad job never blocks the scheduler's other jobs.
    """
    action = (job or {}).get("action", "").strip()
    if action == WELCOME_HOME_ACTION:
        ok, reason = welcome_home(announce_fn=announce_fn, log=log,
                                  now=now, states=states, ping=ping,
                                  phone_ip=phone_ip)
        return f"welcome-home: {'spoken' if ok else reason}"
    if action.startswith(ANNOUNCE_ACTION_PREFIX):
        text = action[len(ANNOUNCE_ACTION_PREFIX):].strip()
        ok, reason = announce_if_home(text, announce_fn=announce_fn,
                                      log=log, states=states, ping=ping,
                                      phone_ip=phone_ip, now=now)
        return f"announce-if-home: {'spoken' if ok else reason}"
    return f"unknown smart-announcement job: {action!r}"


# --- urgent-email announcements (v0.3.10) ---------------------------------
# The last unfinished piece of Joey's 2026-10-23 smart-announcement
# request: a scheduled job that speaks urgent mail on the Home Mini.
# What counts as "urgent" is configurable, not hardcoded to one guess:
# VIKI_URGENT_SENDERS (comma-separated sender substrings) and
# VIKI_URGENT_KEYWORDS (comma-separated, overrides the defaults below).
# Until Joey sets his own list, the keyword defaults stay conservative
# and the job itself is opt-in — nothing speaks until he creates the
# scheduled task, and announce_if_home still gates on phone-home +
# lights-on and never repeats a message.
DEFAULT_URGENT_KEYWORDS = ("urgent", "emergency", "asap",
                            "action required", "time-sensitive",
                            "time sensitive")


def _urgent_config(senders=None, keywords=None):
    """Resolve urgency criteria: explicit args win, then env, then defaults."""
    if senders is None:
        senders = [s.strip() for s in os.getenv("VIKI_URGENT_SENDERS", "").split(",")
                   if s.strip()]
    if keywords is None:
        env = [k.strip() for k in os.getenv("VIKI_URGENT_KEYWORDS", "").split(",")
               if k.strip()]
        keywords = env or list(DEFAULT_URGENT_KEYWORDS)
    return [s.lower() for s in senders], [k.lower() for k in keywords]


def is_urgent(message, senders=None, keywords=None):
    """True if a Gmail message dict looks urgent.

    Sender match is a case-insensitive substring on the From header
    (so "gerresheimer.com" matches any sender at that domain); keyword
    match scans the subject + snippet. A message with neither is not
    urgent, whatever the sender's display name claims.
    """
    senders, keywords = _urgent_config(senders, keywords)
    sender = str((message or {}).get("from", "")).lower()
    if senders and any(s in sender for s in senders):
        return True
    haystack = (f"{(message or {}).get('subject', '')} "
                f"{(message or {}).get('snippet', '')}").lower()
    return any(k in haystack for k in keywords)


def announce_urgent_emails(messages, announce_fn=None, log=None, states=None,
                           ping=None, phone_ip=None, senders=None,
                           keywords=None, now=None):
    """Speak each urgent message in ``messages`` once, if Joey is home.

    Non-urgent messages are silently skipped. Each spoken line goes
    through announce_if_home, so the usual gates apply (phone home +
    lights on, never repeated, failures not recorded). Returns a short
    summary string for the scheduler job's record, e.g.
    "urgent emails: 2 announced" or "no urgent emails".
    """
    urgent = [m for m in (messages or [])
              if is_urgent(m, senders=senders, keywords=keywords)]
    if not urgent:
        return "no urgent emails"
    spoken, skipped = 0, []
    for message in urgent:
        sender = str(message.get("from") or "someone").split("<")[0].strip() or "someone"
        subject = str(message.get("subject") or "(no subject)").strip()
        ok, reason = announce_if_home(
            f"Urgent email from {sender}: {subject}.",
            announce_fn=announce_fn, log=log, states=states, ping=ping,
            phone_ip=phone_ip, now=now)
        if ok:
            spoken += 1
        else:
            skipped.append(reason)
    summary = f"urgent emails: {spoken} announced"
    if skipped:
        summary += f", {len(skipped)} skipped ({', '.join(sorted(set(skipped)))})"
    return summary


def announce_followup_nudges(threads, announce_fn=None, log=None, states=None,
                             ping=None, phone_ip=None, now=None):
    """Speak each unanswered thread once, if Joey is home.

    The follow-up-nudges job (v0.3.11) only listed threads on the job
    result; this second slice speaks them, mirroring
    announce_urgent_emails: one line per thread through announce_if_home,
    so the usual gates apply (phone home + lights on, never repeated,
    failures not recorded). Returns a short summary string for the
    scheduler job's record, e.g. "follow-up nudges: 2 announced" or
    "no follow-up nudges".
    """
    threads = [t for t in (threads or []) if t]
    if not threads:
        return "no follow-up nudges"
    spoken, skipped = 0, []
    for thread in threads:
        sender = str(thread.get("from") or "someone").split("<")[0].strip() or "someone"
        subject = str(thread.get("subject") or "(no subject)").strip()
        ok, reason = announce_if_home(
            f"You still owe a reply to {sender} about {subject}.",
            announce_fn=announce_fn, log=log, states=states, ping=ping,
            phone_ip=phone_ip, now=now)
        if ok:
            spoken += 1
        else:
            skipped.append(reason)
    summary = f"follow-up nudges: {spoken} announced"
    if skipped:
        summary += f", {len(skipped)} skipped ({', '.join(sorted(set(skipped)))})"
    return summary
