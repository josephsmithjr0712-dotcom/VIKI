"""Tests for updater.py: version parsing, preserve-list safety, and check()."""
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import updater as updater_mod


class TestNorm(unittest.TestCase):
    def test_messy_tags(self):
        cases = {
            "v0.2.0": (0, 2, 0),
            "V0.1.2": (0, 1, 2),
            "Version02": (2, 0, 0),
            "version0.3": (0, 3, 0),
            "release-1.2": (1, 2, 0),
            "r2.0": (2, 0, 0),
            "ver 3.1.4": (3, 1, 4),
            "0.2.3": (0, 2, 3),
            "  v0.2.3  ": (0, 2, 3),
            "2": (2, 0, 0),
            "0.2": (0, 2, 0),
            "nope": (0, 0, 0),
            "": (0, 0, 0),
        }
        for tag, want in cases.items():
            with self.subTest(tag=tag):
                self.assertEqual(updater_mod._norm(tag), want)

    def test_ordering(self):
        self.assertTrue(updater_mod._norm("v0.2.3") > updater_mod._norm("0.2.2"))
        self.assertTrue(updater_mod._norm("0.10.0") > updater_mod._norm("0.9.9"))
        self.assertFalse(updater_mod._norm("0.2.3") > updater_mod._norm("0.2.3"))
        self.assertFalse(updater_mod._norm("0.1.9") > updater_mod._norm("0.2.0"))


class TestShouldPreserve(unittest.TestCase):
    def test_never_touch_secrets_and_data(self):
        preserved = [
            ".env",
            ".venv",
            ".git",
            "workspace",
            "backups",
            "__pycache__",
            "memory",  # runtime memory store — an update must never wipe it
            "connectors/my_token.json",
            "connectors/client_secrets.json",
        ]
        for p in preserved:
            with self.subTest(path=p):
                self.assertTrue(updater_mod._should_preserve(Path(p)), p)

    def test_code_files_not_preserved(self):
        not_preserved = ["agent.py", "server.py", "README.md", "updater.py"]
        for p in not_preserved:
            with self.subTest(path=p):
                self.assertFalse(updater_mod._should_preserve(Path(p)), p)


class TestLocalVersion(unittest.TestCase):
    def test_reads_version_file(self):
        with tempfile.TemporaryDirectory() as d:
            vf = Path(d) / "VERSION"
            vf.write_text("0.2.4\n", encoding="utf-8")
            with patch.object(updater_mod, "VERSION_FILE", vf):
                self.assertEqual(updater_mod.local_version(), "0.2.4")

    def test_missing_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(updater_mod, "VERSION_FILE", Path(d) / "NOPE"):
                self.assertEqual(updater_mod.local_version(), "0.0.0")


class TestCheck(unittest.TestCase):
    def test_update_available(self):
        rel = {
            "version": "0.2.4",
            "tag": "v0.2.4",
            "zip_url": "http://x/v.zip",
            "notes": "n",
        }
        with patch.object(updater_mod, "local_version", return_value="0.2.3"), patch.object(
            updater_mod, "latest_release", return_value=rel
        ):
            out = updater_mod.check()
        self.assertTrue(out["ok"])
        self.assertTrue(out["update_available"])
        self.assertEqual(out["latest"], "0.2.4")
        self.assertEqual(out["zip_url"], "http://x/v.zip")

    def test_no_update_when_same(self):
        rel = {"version": "0.2.3", "tag": "v0.2.3", "zip_url": "http://x/v.zip", "notes": ""}
        with patch.object(updater_mod, "local_version", return_value="0.2.3"), patch.object(
            updater_mod, "latest_release", return_value=rel
        ):
            out = updater_mod.check()
        self.assertTrue(out["ok"])
        self.assertFalse(out["update_available"])
        self.assertEqual(out["zip_url"], "")

    def test_api_error_is_reported(self):
        with patch.object(updater_mod, "local_version", return_value="0.2.3"), patch.object(
            updater_mod, "latest_release", side_effect=RuntimeError("boom")
        ):
            out = updater_mod.check()
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "boom")
        self.assertEqual(out["local"], "0.2.3")


class TestVerifyArchive(unittest.TestCase):
    def _make_zip(self, d, names):
        zf = Path(d) / "r.zip"
        with zipfile.ZipFile(zf, "w") as z:
            for n in names:
                z.writestr(n, "data")
        return zf

    def test_good_archive_passes(self):
        with tempfile.TemporaryDirectory() as d:
            zf = self._make_zip(d, ["agent.py", "server.py"])
            self.assertEqual(updater_mod._verify_archive(zf), "")

    def test_empty_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            zf = Path(d) / "r.zip"
            zf.write_bytes(b"")
            self.assertIn("empty", updater_mod._verify_archive(zf))

    def test_non_zip_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            zf = Path(d) / "r.zip"
            zf.write_text("this is not a zip file at all")
            self.assertIn("valid zip", updater_mod._verify_archive(zf))

    def test_zip_slip_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            zf = self._make_zip(d, ["ok.py", "../../evil.py"])
            self.assertIn("unsafe", updater_mod._verify_archive(zf))

    def test_absolute_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            zf = self._make_zip(d, ["/etc/evil"])
            self.assertIn("unsafe", updater_mod._verify_archive(zf))

    def test_oversize_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            zf = self._make_zip(d, ["ok.py"])
            with patch.object(updater_mod, "MAX_RELEASE_ZIP", 1):
                self.assertIn("over", updater_mod._verify_archive(zf))

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            zf = Path(d) / "nope.zip"
            self.assertIn("unreadable", updater_mod._verify_archive(zf))


class TestVerifyReleaseDir(unittest.TestCase):
    def _release_dir(self, d, version):
        src = Path(d) / "src"
        src.mkdir()
        for name in ("agent.py", "server.py", "tools.py"):
            (src / name).write_text("# code", encoding="utf-8")
        (src / "VERSION").write_text(version + "\n", encoding="utf-8")
        return src

    def test_good_release_passes(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._release_dir(d, "0.2.14")
            self.assertEqual(updater_mod._verify_release_dir(src, "v0.2.14"), "")

    def test_missing_core_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._release_dir(d, "0.2.14")
            (src / "server.py").unlink()
            self.assertIn("server.py", updater_mod._verify_release_dir(src, "v0.2.14"))

    def test_version_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._release_dir(d, "0.2.13")
            self.assertIn("VERSION", updater_mod._verify_release_dir(src, "v0.2.14"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
