"""Tests for clearing finished one-shot checks (v0.3.31).

One-shot checks turn themselves off after firing and can never run
again, but they stayed in the list forever — a spent one-shot cluttered
list_tasks and the web panel alongside checks that still run. Now
Scheduler.prune_finished_once, the clear_finished_tasks chat tool,
POST /api/tasks/prune-finished, and a panel "Clear finished" button all
drop exactly the fired one-shots: pending one-shots (never run) and
every recurring check are kept, even paused ones, because those may
still run.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import SessionStore
from scheduler import Scheduler
from tools import Toolbox


def toolbox():
    tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
    tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
    return tb


def _seed(s):
    """A fired once, a pending once, a paused once, and a daily job."""
    fired = s.add("Fired once", "once", action="connection-check",
                  at="2026-01-01 09:00")
    s.run_one(fired["id"], lambda j: "done")
    pending = s.add("Pending once", "once", action="connection-check",
                    at="2099-01-01 09:00")
    paused = s.add("Paused once", "once", action="connection-check",
                   at="2099-06-01 09:00")
    s.set_enabled(paused["id"], False)
    daily = s.add("Daily", "daily", action="connection-check", at="09:00")
    return fired, pending, paused, daily


class TestPruneFinishedOnce(unittest.TestCase):
    def test_removes_only_fired_once_jobs(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        fired, pending, paused, daily = _seed(s)
        self.assertEqual(s.prune_finished_once(), 1)
        names = [t["name"] for t in s.list_tasks()]
        self.assertNotIn("Fired once", names)
        self.assertEqual(names, ["Pending once", "Paused once", "Daily"])

    def test_prune_is_idempotent_and_persists(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        _seed(s)
        self.assertEqual(s.prune_finished_once(), 1)
        self.assertEqual(s.prune_finished_once(), 0)
        # A fresh Scheduler over the same folder sees the pruned list.
        again = Scheduler(folder=s.dir)
        self.assertEqual([t["name"] for t in again.list_tasks()],
                         ["Pending once", "Paused once", "Daily"])

    def test_nothing_finished_removes_nothing(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        s.add("Daily", "daily", action="connection-check", at="09:00")
        self.assertEqual(s.prune_finished_once(), 0)
        self.assertEqual(len(s.list_tasks()), 1)

    def test_tool_reports_count_and_empty_case(self):
        tb = toolbox()
        _seed(tb.scheduler)
        self.assertEqual(tb._clear_finished_tasks(),
                         "Cleared 1 finished one-shot check.")
        self.assertEqual(tb._clear_finished_tasks(),
                         "No finished one-shot checks to clear.")

    def test_tool_is_registered(self):
        tb = toolbox()
        self.assertIn("clear_finished_tasks", tb.enabled)

    def test_panel_contract_has_prune_button(self):
        html = (Path(__file__).resolve().parent.parent / "static"
                / "index.html").read_text()
        self.assertIn('data-action="prune-finished"', html)
        self.assertIn("Clear finished", html)
        self.assertIn("/api/tasks/prune-finished", html)


# Server endpoint tests need the repo's own import pattern (see
# tests/test_server.py): pin the web credentials to temp values so
# importing server.py never writes a .env into the repo, and stand in
# a FakeVIKI for the Ollama-backed agent, which is unavailable here.
import os as _os

_TMP = tempfile.mkdtemp(prefix="viki-prune-finished-")
for _k, _v in {"VIKI_SESSION_DIR": _TMP,
               "VIKI_SECRET_KEY": "test-secret-key",
               "VIKI_WEB_PASSWORD": "test-password",
               "VIKI_AUTO_MEMORY": "off"}.items():
    _os.environ.setdefault(_k, _v)

try:
    import agent as _agent_mod

    class _FakeVIKI:
        def __init__(self, toolbox):
            self.toolbox = toolbox
            self.model = "test-model"

        def warmup(self):
            pass

    _real_VIKI = _agent_mod.VIKI
    _agent_mod.VIKI = _FakeVIKI
    try:
        import server as server_mod
    finally:
        _agent_mod.VIKI = _real_VIKI
    _HAS_SERVER = server_mod is not None
except Exception:
    server_mod = None
    _HAS_SERVER = False


@unittest.skipUnless(_HAS_SERVER, "server.py not importable here")
class TestPruneFinishedEndpoint(unittest.TestCase):
    def _client(self):
        client = server_mod.app.test_client()
        with client.session_transaction() as sess:
            sess["viki_auth"] = True
        return client

    def test_prune_endpoint_removes_only_fired_once(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        _seed(sched)
        old_sched = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            client = self._client()
            resp = client.post("/api/tasks/prune-finished")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json()["removed"], 1)
            self.assertEqual([t["name"] for t in sched.list_tasks()],
                             ["Pending once", "Paused once", "Daily"])
            resp = client.post("/api/tasks/prune-finished")
            self.assertEqual(resp.get_json()["removed"], 0)
        finally:
            server_mod.task_scheduler = old_sched


if __name__ == "__main__":
    unittest.main()
