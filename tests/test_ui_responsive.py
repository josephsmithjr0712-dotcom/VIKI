"""Static regression tests for VIKI's responsive web UI (v0.3.2).

The UI is a single static/index.html; these tests pin the responsive
markers so a later edit can't silently drop mobile/desktop support.
They read the file, they don't render it.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


class TestResponsiveUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = HTML.read_text(encoding="utf-8")
        style = re.search(r"<style>(.*?)</style>", cls.html, re.S)
        cls.css = style.group(1) if style else ""
        script = re.search(r"<script>(.*?)</script>", cls.html, re.S)
        cls.js = script.group(1) if script else ""

    def test_viewport_fit_cover(self):
        self.assertIn("viewport-fit=cover", self.html)

    def test_safe_area_insets_used(self):
        self.assertIn("safe-area-inset-bottom", self.css)
        self.assertIn("safe-area-inset-top", self.css)

    def test_dynamic_viewport_height(self):
        self.assertIn("100dvh", self.css)

    def test_desktop_breakpoint_widens_column(self):
        m = re.search(r"@media\(min-width:900px\)\{(.*?)\n\}", self.css, re.S)
        self.assertIsNotNone(m, "desktop breakpoint missing")
        self.assertIn("max-width:780px", m.group(1))

    def test_mobile_breakpoint_no_zoom_inputs(self):
        m = re.search(r"@media\(max-width:560px\)\{(.*?)\n\}", self.css, re.S)
        self.assertIsNotNone(m, "mobile breakpoint missing")
        self.assertIn("font-size:16px", m.group(1))

    def test_chat_switcher_does_not_overflow_narrow_screens(self):
        self.assertIn("min-width:0", self.css)

    def test_rename_delete_buttons_wired(self):
        self.assertIn('id="renamechat"', self.html)
        self.assertIn('id="delchat"', self.html)
        self.assertIn("/rename", self.js)
        self.assertIn("method:'DELETE'", self.js)
        self.assertIn("can't be deleted", self.js)

    def test_handsfree_quick_reply_loop(self):
        self.assertIn("handsfreeOn", self.js)
        self.assertIn("function startListening", self.js)
        self.assertIn("if(handsfreeOn&&input.value.trim())send()", self.js)
        self.assertIn("handsfreeOn||$('#speak').checked", self.js)
        self.assertIn("if(handsfreeOn)startListening()", self.js)

    def test_jump_to_latest_wired(self):
        self.assertIn('id="totop"', self.html)
        self.assertIn("totop", self.css)
        self.assertIn("scrollHeight - feed.scrollTop - feed.clientHeight", self.js)
        self.assertIn("$('#totop').onclick", self.js)


if __name__ == "__main__":
    unittest.main()
