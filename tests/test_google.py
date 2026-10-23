"""Tests for the Google connector (Gmail + Calendar). The Google API is mocked."""
import base64
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.google as g


def _b64(text):
    return base64.urlsafe_b64encode(text.encode()).decode()


class TestGmail(TestCase):
    def test_search_formats_results(self):
        svc = MagicMock()
        svc.users().messages().list().execute.return_value = {"messages": [{"id": "abc123"}]}
        svc.users().messages().get().execute.return_value = {
            "snippet": "hello there",
            "payload": {"headers": [
                {"name": "From", "value": "boss@example.com"},
                {"name": "Subject", "value": "Hi"},
                {"name": "Date", "value": "Tue, 21 Oct 2026"},
            ]},
        }
        with patch.object(g, "_service", return_value=svc):
            out = g.gmail_search("from:boss")
        self.assertIn("[abc123]", out)
        self.assertIn("boss@example.com", out)
        self.assertIn("hello there", out)

    def test_search_no_results(self):
        svc = MagicMock()
        svc.users().messages().list().execute.return_value = {}
        with patch.object(g, "_service", return_value=svc):
            self.assertEqual(g.gmail_search("zzz"), "No messages found.")

    def test_read_extracts_plain_text(self):
        svc = MagicMock()
        svc.users().messages().get().execute.return_value = {
            "payload": {
                "headers": [{"name": "Subject", "value": "Test"}],
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64("plain body here")}},
                    {"mimeType": "text/html", "body": {"data": _b64("<b>html</b>")}},
                ],
            }
        }
        with patch.object(g, "_service", return_value=svc):
            out = g.gmail_read("abc123")
        self.assertIn("plain body here", out)
        self.assertNotIn("<b>", out)

    def test_send_builds_rfc822(self):
        svc = MagicMock()
        svc.users().messages().send().execute.return_value = {"id": "sent1"}
        with patch.object(g, "_service", return_value=svc):
            out = g.gmail_send("a@b.com", "Hi", "body text")
        self.assertIn("sent1", out)
        sent_body = svc.users().messages().send.call_args[1]["body"]
        raw = base64.urlsafe_b64decode(sent_body["raw"]).decode()
        self.assertIn("a@b.com", raw)
        self.assertIn("Hi", raw)


class TestCalendar(TestCase):
    def test_list_formats_events(self):
        svc = MagicMock()
        svc.events().list().execute.return_value = {"items": [
            {"summary": "Dentist", "start": {"dateTime": "2026-10-22T09:00:00-04:00"}, "location": "Main St"},
            {"summary": "Lunch", "start": {"date": "2026-10-23"}},
        ]}
        with patch.object(g, "_service", return_value=svc):
            out = g.calendar_list(days=2)
        self.assertIn("Dentist", out)
        self.assertIn("Main St", out)
        self.assertIn("Lunch", out)

    def test_list_empty(self):
        svc = MagicMock()
        svc.events().list().execute.return_value = {}
        with patch.object(g, "_service", return_value=svc):
            self.assertEqual(g.calendar_list(), "No upcoming events.")

    def test_create_ok(self):
        svc = MagicMock()
        svc.events().insert().execute.return_value = {"id": "ev1"}
        with patch.object(g, "_service", return_value=svc):
            out = g.calendar_create("Gym", "2026-10-22 18:00", "2026-10-22 19:00")
        self.assertIn("ev1", out)
        body = svc.events().insert.call_args[1]["body"]
        self.assertEqual(body["summary"], "Gym")
        self.assertIn("2026-10-22T18:00", body["start"]["dateTime"])

    def test_create_bad_date(self):
        with patch.object(g, "_service", return_value=MagicMock()):
            out = g.calendar_create("Gym", "tomorrow 6pm", "tomorrow 7pm")
        self.assertIn("YYYY-MM-DD HH:MM", out)


class TestGating(TestCase):
    def test_no_token_raises_helpful_error(self):
        with patch.object(g, "TOKEN_FILE", Path(tempfile.mkdtemp()) / "nope.json"):
            with self.assertRaises(RuntimeError) as ctx:
                g._credentials()
        self.assertIn("google_auth", str(ctx.exception))

    def test_toolbox_hides_google_without_token(self):
        import tools as tools_mod
        from memory import SessionStore
        from tools import Toolbox, GOOGLE_TOOLS

        with patch.object(tools_mod, "google_connected", return_value=False):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        for t in GOOGLE_TOOLS:
            self.assertNotIn(t, tb.enabled)

    def test_toolbox_shows_google_with_token(self):
        import tools as tools_mod
        from memory import SessionStore
        from tools import Toolbox, GOOGLE_TOOLS

        with patch.object(tools_mod, "google_connected", return_value=True):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        for t in GOOGLE_TOOLS:
            self.assertIn(t, tb.enabled)

    def test_send_and_create_need_approval(self):
        import tools as tools_mod
        from memory import SessionStore
        from tools import Toolbox

        with patch.object(tools_mod, "google_connected", return_value=True):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        self.assertTrue(tb.needs_approval("gmail_send", {}))
        self.assertTrue(tb.needs_approval("calendar_create", {}))
        self.assertFalse(tb.needs_approval("gmail_search", {}))


if __name__ == "__main__":
    main(verbosity=2)
