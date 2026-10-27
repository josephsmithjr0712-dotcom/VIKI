"""Frontend/backend contract tests (v0.3.13).

Guards the `/api/ha` vs `/api/ha_check` class of bug: every `/api/*`
endpoint the web UI calls must exist as a route in server.py, and every
DOM id the UI looks up with getElementById must exist in the same page.

Pure static analysis over the repo tree: no server, no network, no node.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "server.py"
STATIC = ROOT / "static"

ROUTE_RE = re.compile(r"@app\.(?:get|post|put|patch|delete|route)\(\s*['\"]([^'\"]+)['\"]")
API_LITERAL_RE = re.compile(r"['\"]/api[^'\"]*['\"]")
ID_REF_RE = re.compile(r"getElementById\(\s*['\"]([^'\"]+)['\"]\)")
ID_DEF_RE = re.compile(r'id="([^"]+)"')


def load_routes():
    routes = set()
    for m in ROUTE_RE.finditer(SERVER.read_text(errors="ignore")):
        routes.add(m.group(1))
    return routes


def route_matches(route, path):
    """True if a Flask route pattern (with <param> segments) serves path."""
    pattern = re.sub(r"<[^>]+>", r"[^/]+", route)
    return re.fullmatch(pattern, path) is not None


def frontend_endpoints():
    """(literal, html_file) pairs for /api/* string literals in static HTML.

    Query strings are stripped. A literal ending in '/' comes from a
    concatenation like '/api/chats/' + encodeURIComponent(id) and denotes
    both the collection route and a parameterized child route.
    """
    found = []
    for html in sorted(STATIC.glob("*.html")):
        text = html.read_text(errors="ignore")
        for m in API_LITERAL_RE.finditer(text):
            literal = m.group(0)[1:-1].split("?")[0]
            found.append((literal, html.name))
    return found


class TestFrontendContract(unittest.TestCase):
    def test_every_frontend_endpoint_has_a_backend_route(self):
        routes = load_routes()
        self.assertTrue(routes, "no routes parsed from server.py")
        problems = []
        for literal, page in frontend_endpoints():
            if literal.endswith("/"):
                base = literal.rstrip("/")
                if not any(route_matches(r, base) for r in routes):
                    problems.append(f"{page}: {literal} (no collection route)")
                if not any(
                    r.startswith(base + "/<") for r in routes
                ):
                    problems.append(f"{page}: {literal} (no parameterized child route)")
            else:
                if not any(route_matches(r, literal) for r in routes):
                    problems.append(f"{page}: {literal} (no matching route)")
        self.assertEqual(problems, [], "frontend calls backend routes that do not exist")

    def test_every_get_element_by_id_exists(self):
        problems = []
        for html in sorted(STATIC.glob("*.html")):
            text = html.read_text(errors="ignore")
            defined = set(ID_DEF_RE.findall(text))
            for ref in ID_REF_RE.findall(text):
                if ref not in defined:
                    problems.append(f"{html.name}: getElementById('{ref}') has no id=")
        self.assertEqual(problems, [], "UI looks up DOM ids that do not exist")


if __name__ == "__main__":
    unittest.main()
