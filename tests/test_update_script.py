"""Tests for update-viki.sh: the one-command HomeLab server updater."""
import shutil
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "update-viki.sh"


class TestUpdateScript(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.is_file(), "update-viki.sh missing from repo root")

    def test_runs_via_bash(self):
        # The push path strips the executable bit, so the supported
        # invocation is `bash update-viki.sh`; assert that works.
        bash = shutil.which("bash")
        self.assertIsNotNone(bash, "bash not found")
        result = subprocess.run(
            [bash, str(SCRIPT), "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, f"bash invocation failed:\n{result.stderr}")
        self.assertIn("--dry-run", result.stdout)

    def test_syntax_valid(self):
        bash = shutil.which("bash")
        self.assertIsNotNone(bash, "bash not found")
        result = subprocess.run(
            [bash, "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, f"bash -n failed:\n{result.stderr}")

    def test_stashes_dirty_changes(self):
        text = SCRIPT.read_text()
        self.assertIn("git stash push -u", text,
                      "script must stash local changes (the thing blocking git pull)")

    def test_hard_resets_to_origin_main(self):
        text = SCRIPT.read_text()
        self.assertIn("git fetch origin", text)
        self.assertIn('git reset --hard "origin/$BRANCH"', text)

    def test_restarts_viki_server_service(self):
        text = SCRIPT.read_text()
        self.assertIn('systemctl restart "$SERVICE"', text)
        self.assertIn('SERVICE="viki-server"', text,
                      "must restart viki-server, not some other service")

    def test_prints_service_status(self):
        text = SCRIPT.read_text()
        self.assertIn("systemctl is-active", text)
        self.assertIn("systemctl status", text)

    def test_refuses_non_systemd_machines(self):
        text = SCRIPT.read_text()
        self.assertIn("/run/systemd/system", text,
                      "must bail on machines without systemd (e.g. the Intel Mac)")

    def test_verifies_repo_remote(self):
        text = SCRIPT.read_text()
        self.assertIn("josephsmithjr0712-dotcom/VIKI", text,
                      "must verify origin is the VIKI repo before touching anything")


if __name__ == "__main__":
    unittest.main()
