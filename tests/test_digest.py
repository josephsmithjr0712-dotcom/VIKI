"""Tests for VIKI's connection checks (v0.3.0)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectors.digest import connection_check


def fake_gmail(query, max_results=10):
    assert "is:unread" in query
    return "[1] From: boss@work.com\nSubject: Shift swap\nSnippet: can you cover Friday?"


def fake_calendar(days=1, max_results=10):
    assert days == 1
    return "2026-10-26T07:00 — Gerresheimer first day @ plant"


def failing_gmail(query, max_results=10):
    raise RuntimeError("Google is not connected yet.")


class TestConnectionCheck(unittest.TestCase):
    def test_brief_combines_mail_and_calendar(self):
        brief = connection_check(gmail_search=fake_gmail, calendar_list=fake_calendar)
        self.assertIn("Shift swap", brief)
        self.assertIn("Gerresheimer first day", brief)

    def test_failing_service_becomes_a_line_not_a_crash(self):
        brief = connection_check(gmail_search=failing_gmail, calendar_list=fake_calendar)
        self.assertIn("Gmail check failed", brief)
        self.assertIn("Gerresheimer first day", brief)

    def test_empty_results_still_readable(self):
        brief = connection_check(gmail_search=lambda q, max_results=10: "No messages found.",
                                 calendar_list=lambda days=1, max_results=10: "No upcoming events.")
        self.assertIn("No messages found.", brief)
        self.assertIn("No upcoming events.", brief)


if __name__ == "__main__":
    unittest.main()
