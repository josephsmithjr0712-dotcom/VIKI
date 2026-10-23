"""Google connector for VIKI: Gmail + Calendar tools over OAuth 2.0.

One-time setup:
1. In Google Cloud Console, create an OAuth client ID (Desktop app) and
   download it as client_secrets.json next to this file
   (or set GOOGLE_CLIENT_SECRETS_FILE to its path).
2. From the VIKI folder run:  python -m connectors.google_auth
   A browser window opens for approval; the token is saved to
   GOOGLE_TOKEN_FILE (default ./connectors/google_token.json).
3. The Google tools enable automatically once the token exists.

Google libraries are imported lazily so the rest of VIKI works without them.
"""
import base64
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
]

HERE = Path(__file__).resolve().parent
SECRETS_FILE = Path(os.getenv("GOOGLE_CLIENT_SECRETS_FILE") or HERE / "client_secrets.json")
TOKEN_FILE = Path(os.getenv("GOOGLE_TOKEN_FILE") or HERE / "google_token.json")


def token_exists() -> bool:
    return TOKEN_FILE.exists()


def _credentials():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        raise RuntimeError(
            "Google libraries are not installed. Run: "
            "pip install google-api-python-client google-auth-oauthlib "
            "Then run 'python -m connectors.google_auth' once to approve access."
        )

    if not TOKEN_FILE.exists():
        raise RuntimeError(
            "Google is not connected yet. Run 'python -m connectors.google_auth' once to approve access."
        )
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _service(api: str, version: str):
    from googleapiclient.discovery import build

    return build(api, version, credentials=_credentials(), cache_discovery=False)


# --- Gmail -----------------------------------------------------------------
def gmail_search(query, max_results=5):
    svc = _service("gmail", "v1")
    try:
        resp = (
            svc.users().messages().list(userId="me", q=query, maxResults=int(max_results)).execute()
        )
    except Exception as e:
        return f"Gmail search failed: {e}"
    msgs = resp.get("messages", [])
    if not msgs:
        return "No messages found."
    out = []
    for m in msgs:
        try:
            full = (
                svc.users()
                .messages()
                .get(userId="me", id=m["id"], format="metadata",
                     metadataHeaders=["From", "Subject", "Date"])
                .execute()
            )
            hdr = {h["name"]: h["value"] for h in full["payload"].get("headers", [])}
            out.append(
                f"[{m['id']}] {hdr.get('Date', '')}\n"
                f"From: {hdr.get('From', '')}\n"
                f"Subject: {hdr.get('Subject', '')}\n"
                f"Snippet: {full.get('snippet', '')}"
            )
        except Exception as e:
            out.append(f"[{m['id']}] (could not fetch: {e})")
    return "\n\n".join(out)


def _body_from_payload(payload) -> str:
    """Extract readable text from a Gmail message payload (recursive)."""
    mime = payload.get("mimeType", "")
    data = payload.get("body", {}).get("data")
    if data:
        try:
            text = base64.urlsafe_b64decode(data).decode("utf-8", "ignore")
        except Exception:
            text = ""
        if mime.startswith("text/plain"):
            return text
        if mime.startswith("text/html"):
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    for part in payload.get("parts", []):
        text = _body_from_payload(part)
        if text.strip():
            return text
    return ""


def gmail_read(message_id, max_chars=6000):
    svc = _service("gmail", "v1")
    try:
        full = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception as e:
        return f"Gmail read failed: {e}"
    hdr = {h["name"]: h["value"] for h in full["payload"].get("headers", [])}
    body = _body_from_payload(full["payload"])[: int(max_chars)]
    return (
        f"From: {hdr.get('From', '')}\n"
        f"To: {hdr.get('To', '')}\n"
        f"Date: {hdr.get('Date', '')}\n"
        f"Subject: {hdr.get('Subject', '')}\n\n{body}"
    )


def gmail_send(to, subject, body):
    import email.message

    svc = _service("gmail", "v1")
    msg = email.message.EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    try:
        sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    except Exception as e:
        return f"Gmail send failed: {e}"
    return f"Sent (id {sent.get('id')}) to {to}."


# --- Calendar --------------------------------------------------------------
def calendar_list(days=1, max_results=10):
    svc = _service("calendar", "v3")
    now = datetime.now().astimezone()
    try:
        resp = (
            svc.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=(now + timedelta(days=int(days))).isoformat(),
                maxResults=int(max_results),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
    except Exception as e:
        return f"Calendar lookup failed: {e}"
    items = resp.get("items", [])
    if not items:
        return "No upcoming events."
    out = []
    for e in items:
        start = e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "?"))
        line = f"{start} — {e.get('summary', '(no title)')}"
        if e.get("location"):
            line += f" @ {e['location']}"
        out.append(line)
    return "\n".join(out)


def calendar_create(summary, start, end, description="", location=""):
    """start/end as 'YYYY-MM-DD HH:MM' in local time."""
    svc = _service("calendar", "v3")
    try:
        tz = datetime.now().astimezone().tzinfo
        s = datetime.strptime(start, "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        e = datetime.strptime(end, "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    except ValueError:
        return "Use 'YYYY-MM-DD HH:MM' for start and end, e.g. 2026-10-22 14:00."
    body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": s.isoformat()},
        "end": {"dateTime": e.isoformat()},
    }
    try:
        ev = svc.events().insert(calendarId="primary", body=body).execute()
    except Exception as e:
        return f"Calendar create failed: {e}"
    return f"Created '{summary}' on {start} (id {ev.get('id')})."
