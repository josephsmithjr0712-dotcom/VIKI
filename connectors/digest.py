"""Connection checks for VIKI: one brief from Gmail + Calendar.

Used by the `check_connections` agent tool and by scheduled
connection-check jobs. The Google connector already covers both APIs
(no new OAuth scopes), so this module only shapes the results into a
short brief. Callers may inject their own gmail/calendar functions for
testing; by default the real connector functions are used, imported
lazily so the module stays importable without google-api-python-client.
"""


def _gmail_search(query, max_results=10):
    from connectors.google import gmail_search
    return gmail_search(query, max_results=max_results)


def _calendar_list(days=1, max_results=10):
    from connectors.google import calendar_list
    return calendar_list(days=days, max_results=max_results)


def connection_check(gmail_search=None, calendar_list=None):
    """Return a short brief: unread mail from the last day + today's
    calendar. Never raises: a failing service becomes one line in the
    brief, so a scheduled check still reports instead of dying."""
    gmail_search = gmail_search or _gmail_search
    calendar_list = calendar_list or _calendar_list
    parts = []
    try:
        mail = gmail_search("is:unread newer_than:1d", max_results=10)
    except Exception as e:  # noqa: BLE001 - surfaced in the brief, not hidden
        mail = f"Gmail check failed: {e}"
    parts.append("Unread mail (last day):\n" + (mail or "No unread mail."))
    try:
        cal = calendar_list(days=1, max_results=10)
    except Exception as e:  # noqa: BLE001 - surfaced in the brief, not hidden
        cal = f"Calendar check failed: {e}"
    parts.append("Calendar (today):\n" + (cal or "No upcoming events."))
    return "\n\n".join(parts)
