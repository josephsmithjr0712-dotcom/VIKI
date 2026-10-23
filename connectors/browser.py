"""VIKI's own browser: a real headless Chromium driven through Playwright.

Tools:
  browser_goto(url)      - open a page, returns title + text preview
  browser_snapshot()     - accessibility tree with [ref=eN] handles for clicking/typing
  browser_text()         - full readable text of the current page
  browser_click(ref)     - click the element with the given ref (needs approval)
  browser_fill(ref,text) - type into the element with the given ref (needs approval)
  browser_screenshot()   - save a PNG under workspace/browser/, returns the path

The browser keeps one persistent context, so logins and cookies survive
across tool calls within a server run. Playwright is imported lazily so
VIKI works without it until these tools are used.
"""
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent / "workspace" / "browser"
_session = None


class BrowserSession:
    def __init__(self):
        self._pw = None
        self._page = None

    def _ensure(self):
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && python -m playwright install chromium"
            )
        self._pw = sync_playwright().start()
        try:
            browser = self._pw.chromium.launch(headless=True)
        except Exception as e:
            if "Executable doesn't exist" in str(e):
                raise RuntimeError(
                    "Browser binaries not downloaded yet. Run: python -m playwright install chromium"
                )
            raise
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        self._page = ctx.new_page()

    def close(self):
        if self._pw:
            self._pw.stop()
        self._pw = None
        self._page = None

    def goto(self, url):
        self._ensure()
        self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = self._page.title()
        text = self._page.locator("body").inner_text()[:1500]
        return f"Title: {title}\nURL: {self._page.url}\n\n{text}"

    def snapshot(self):
        self._ensure()
        snap = self._page.locator("body").aria_snapshot(timeout=10000)
        return snap[:8000] if len(snap) > 8000 else snap

    def text(self, max_chars=8000):
        self._ensure()
        return self._page.locator("body").inner_text()[: int(max_chars)]

    def click(self, ref):
        self._ensure()
        self._page.get_by_ref(ref).click(timeout=8000)
        self._page.wait_for_timeout(800)
        return f"Clicked {ref}. New URL: {self._page.url}"

    def fill(self, ref, text, submit=False):
        self._ensure()
        loc = self._page.get_by_ref(ref)
        loc.fill(text, timeout=8000)
        if submit:
            loc.press("Enter")
            self._page.wait_for_timeout(800)
        return f"Filled {ref}." + (f" New URL: {self._page.url}" if submit else "")

    def screenshot(self):
        self._ensure()
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        path = WORK_DIR / f"shot-{datetime.now():%Y%m%d-%H%M%S}.png"
        self._page.screenshot(path=str(path))
        return f"Saved screenshot to workspace/browser/{path.name}"


def _browser() -> BrowserSession:
    global _session
    if _session is None:
        _session = BrowserSession()
    return _session


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        return f"Browser action failed: {type(e).__name__}: {e}"


def browser_goto(url):
    return _run(_browser().goto, url)


def browser_snapshot():
    return _run(_browser().snapshot)


def browser_text(max_chars=8000):
    return _run(_browser().text, max_chars)


def browser_click(ref):
    return _run(_browser().click, ref)


def browser_fill(ref, text, submit=False):
    submit = str(submit).lower() in ("1", "true", "yes") if isinstance(submit, str) else bool(submit)
    return _run(_browser().fill, ref, text, submit)


def browser_screenshot():
    return _run(_browser().screenshot)
