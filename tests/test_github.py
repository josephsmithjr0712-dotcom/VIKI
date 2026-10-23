"""Tests for the GitHub connector. The HTTP layer is mocked."""
import base64
import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.github as gh


class TestGitHub(TestCase):
    def test_search_repos(self):
        with patch.object(gh, "_api", return_value=(True, {"items": [
            {"full_name": "octocat/hello-world", "stargazers_count": 42,
             "description": "demo", "html_url": "https://github.com/octocat/hello-world"},
        ]})):
            out = gh.github_search_repos("demo")
        self.assertIn("octocat/hello-world", out)
        self.assertIn("42", out)

    def test_search_repos_empty(self):
        with patch.object(gh, "_api", return_value=(True, {"items": []})):
            self.assertEqual(gh.github_search_repos("zzz"), "No repositories found.")

    def test_search_code(self):
        with patch.object(gh, "_api", return_value=(True, {"items": [
            {"repository": {"full_name": "a/b"}, "path": "x.py",
             "html_url": "https://github.com/a/b/blob/main/x.py"},
        ]})):
            out = gh.github_search_code("foo")
        self.assertIn("a/b / x.py", out)

    def test_read_file_decodes_base64(self):
        payload = {"encoding": "base64",
                   "content": base64.b64encode(b"print('hi')").decode()}
        with patch.object(gh, "_api", return_value=(True, payload)):
            self.assertIn("print('hi')", gh.github_read_file("a/b", "x.py"))

    def test_read_file_api_error(self):
        with patch.object(gh, "_api", return_value=(False, "GitHub API error 404: Not Found")):
            out = gh.github_read_file("a/b", "nope.py")
        self.assertIn("404", out)

    def test_list_issues(self):
        with patch.object(gh, "_api", return_value=(True, [
            {"number": 7, "title": "Bug here", "labels": [{"name": "bug"}]},
            {"number": 8, "title": "Docs", "labels": []},
        ])):
            out = gh.github_list_issues("a/b")
        self.assertIn("#7 Bug here [bug]", out)
        self.assertIn("#8 Docs", out)

    def test_create_issue(self):
        with patch.object(gh, "_api", return_value=(True, {"number": 9, "html_url": "https://x/9"})):
            out = gh.github_create_issue("a/b", "Title", "Body")
        self.assertIn("#9", out)

    def test_no_token_message(self):
        with patch.object(gh, "_api", return_value=(False, "GitHub is not connected. Set GITHUB_TOKEN in the environment first.")):
            self.assertIn("GITHUB_TOKEN", gh.github_search_repos("x"))

    def test_toolbox_gating(self):
        import tools as tools_mod
        from tools import Toolbox, GITHUB_TOOLS
        from memory import SessionStore
        import tempfile

        with patch.object(tools_mod, "github_connected", return_value=False):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        for t in GITHUB_TOOLS:
            self.assertNotIn(t, tb.enabled)
        with patch.object(tools_mod, "github_connected", return_value=True):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s2"))
        for t in GITHUB_TOOLS:
            self.assertIn(t, tb.enabled)
        self.assertTrue(tb.needs_approval("github_create_issue", {}))
        self.assertFalse(tb.needs_approval("github_search_repos", {}))


if __name__ == "__main__":
    main(verbosity=2)
