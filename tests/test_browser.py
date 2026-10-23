"""Tests for VIKI's browser tools. The Playwright session is mocked."""
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.browser as b


def mock_session():
    sess = MagicMock()
    sess.goto.return_value = "Title: Example\nURL: https://example.com\n\nhello"
    sess.snapshot.return_value = "- button [ref=e1]: OK"
    sess.text.return_value = "page text here"
    sess.click.return_value = "Clicked e1. New URL: https://example.com/next"
    sess.fill.return_value = "Filled e2."
    sess.screenshot.return_value = "Saved screenshot to workspace/browser/shot.png"
    return sess


class TestBrowserTools(TestCase):
    def test_goto(self):
        with patch.object(b, "_session", mock_session()):
            out = b.browser_goto("https://example.com")
        self.assertIn("Example", out)

    def test_snapshot(self):
        with patch.object(b, "_session", mock_session()):
            out = b.browser_snapshot()
        self.assertIn("[ref=e1]", out)

    def test_text(self):
        with patch.object(b, "_session", mock_session()):
            self.assertEqual(b.browser_text(), "page text here")

    def test_click(self):
        sess = mock_session()
        with patch.object(b, "_session", sess):
            out = b.browser_click("e1")
        sess.click.assert_called_once_with("e1")
        self.assertIn("Clicked e1", out)

    def test_fill_submit_parsing(self):
        sess = mock_session()
        with patch.object(b, "_session", sess):
            b.browser_fill("e2", "hello", "true")
        sess.fill.assert_called_once_with("e2", "hello", True)
        with patch.object(b, "_session", sess):
            b.browser_fill("e2", "hello", "false")
        self.assertEqual(sess.fill.call_args[0][2], False)

    def test_screenshot(self):
        with patch.object(b, "_session", mock_session()):
            out = b.browser_screenshot()
        self.assertIn("workspace/browser/", out)

    def test_errors_become_strings(self):
        sess = mock_session()
        sess.goto.side_effect = Exception("boom")
        with patch.object(b, "_session", sess):
            out = b.browser_goto("https://example.com")
        self.assertIn("Browser action failed", out)
        self.assertIn("boom", out)

    def test_toolbox_wiring(self):
        import tools as tools_mod
        from tools import Toolbox, BROWSER_TOOLS
        from memory import SessionStore

        tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        for t in BROWSER_TOOLS:  # always enabled, no credentials needed
            self.assertIn(t, tb.enabled)
        self.assertTrue(tb.needs_approval("browser_click", {"ref": "e1"}))
        self.assertTrue(tb.needs_approval("browser_fill", {"ref": "e2", "text": "x"}))
        self.assertFalse(tb.needs_approval("browser_goto", {"url": "https://example.com"}))


class TestGotoUrlValidation(TestCase):
    """browser_goto only allows http(s) URLs with a real host."""

    def goto_with(self, url):
        sess = mock_session()
        with patch.object(b, "_session", sess):
            out = b.browser_goto(url)
        return out, sess

    def test_valid_https_passes_through(self):
        out, sess = self.goto_with("https://example.com/page")
        sess.goto.assert_called_once_with("https://example.com/page")
        self.assertIn("Example", out)

    def test_valid_http_passes_through(self):
        _, sess = self.goto_with("http://192.168.1.10:8123")
        sess.goto.assert_called_once_with("http://192.168.1.10:8123")

    def test_uppercase_scheme_allowed(self):
        _, sess = self.goto_with("HTTPS://example.com")
        sess.goto.assert_called_once_with("HTTPS://example.com")

    def test_whitespace_stripped(self):
        _, sess = self.goto_with("  https://example.com  ")
        sess.goto.assert_called_once_with("https://example.com")

    def test_javascript_blocked(self):
        out, sess = self.goto_with("javascript:alert(1)")
        sess.goto.assert_not_called()
        self.assertIn("Invalid URL", out)

    def test_data_blocked(self):
        out, sess = self.goto_with("data:text/html,<h1>x</h1>")
        sess.goto.assert_not_called()
        self.assertIn("Invalid URL", out)

    def test_file_blocked(self):
        out, sess = self.goto_with("file:///etc/passwd")
        sess.goto.assert_not_called()
        self.assertIn("Invalid URL", out)

    def test_no_host_rejected(self):
        out, sess = self.goto_with("https:///no-host")
        sess.goto.assert_not_called()
        self.assertIn("Invalid URL", out)

    def test_empty_rejected(self):
        out, sess = self.goto_with("   ")
        sess.goto.assert_not_called()
        self.assertIn("Invalid URL", out)

    def test_non_string_rejected(self):
        out, sess = self.goto_with(None)
        sess.goto.assert_not_called()
        self.assertIn("Invalid URL", out)

    def test_too_long_rejected(self):
        out, sess = self.goto_with("https://example.com/" + "a" * 3000)
        sess.goto.assert_not_called()
        self.assertIn("Invalid URL", out)

    def test_validate_helper_direct(self):
        with self.assertRaises(ValueError):
            b._validate_url("ftp://example.com")
        self.assertEqual(b._validate_url("https://Example.COM/x"), "https://Example.COM/x")


if __name__ == "__main__":
    main(verbosity=2)
