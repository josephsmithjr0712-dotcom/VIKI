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


if __name__ == "__main__":
    main(verbosity=2)
