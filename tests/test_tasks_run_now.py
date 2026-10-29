"""Tests for running a scheduled task right now (v0.3.24).

Next-run display, last outcome, and pause/resume all landed, but a
check could still only run when its slot came round — there was no
way to fire one on demand to test it. Now Scheduler.run_one, the
run_task_now chat tool, POST /api/tasks/<id>/run, and a panel button
all fire the same job immediately, through the normal outcome
recording, so the run shows up in list_tasks and on the panel.
"""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import SessionStore
from scheduler import Scheduler
from tools import Toolbox


def toolbox():
    tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
    tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
    return tb


class TestRunOne(unittest.TestCase):
    def test_run_one_runs_undue_job_and_records_outcome(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", at="09:00")
        outcome = s.run_one(job["id"], lambda j: "checked ok",
                            now=datetime(2026, 10, 28, 17, 20))
        self.assertEqual(outcome["name"], "Morning")
        self.assertEqual(outcome["result"], "checked ok")
        self.assertIsNone(outcome["error"])
        stored = s.list_tasks()[0]
        self.assertEqual(stored["last_run"], "2026-10-28T17:20:00")
        self.assertEqual(stored["run_count"], 1)

    def test_run_one_paused_job_stays_paused(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", at="09:00")
        s.set_enabled(job["id"], False)
        s.run_one(job["id"], lambda j: "checked ok")
        self.assertFalse(s.list_tasks()[0]["enabled"])

    def test_run_one_fires_once_job(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("One-off", "once", at="2026-10-29 09:00")
        s.run_one(job["id"], lambda j: "checked ok")
        stored = s.list_tasks()[0]
        self.assertFalse(stored["enabled"])
        # Fired one-shots never run again, scheduled or manual.
        self.assertEqual(s.run_due(lambda j: "again",
                                   now=datetime(2026, 10, 29, 9, 0)), [])

    def test_run_one_unknown_id_returns_none(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        self.assertIsNone(s.run_one("nope", lambda j: "ok"))

    def test_run_one_executor_failure_is_recorded(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", at="09:00")

        def boom(job):
            raise RuntimeError("kaboom")

        outcome = s.run_one(job["id"], boom)
        self.assertEqual(outcome["error"], "RuntimeError: kaboom")
        self.assertEqual(s.list_tasks()[0]["last_error"],
                         "RuntimeError: kaboom")


class TestRunTaskNowTool(unittest.TestCase):
    def test_tool_runs_and_reports(self):
        import tools as tools_mod
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily", at="09:00")
        old = tools_mod.connection_check
        tools_mod.connection_check = lambda: {"gmail": "ok"}
        try:
            text = tb._run_task_now(job["id"])
        finally:
            tools_mod.connection_check = old
        self.assertIn("Ran 'Morning' now", text)
        self.assertIn("ok", text)
        self.assertIsNotNone(tb.scheduler.list_tasks()[0]["last_run"])

    def test_tool_unknown_id(self):
        tb = toolbox()
        self.assertEqual(tb._run_task_now("nope"), "No task with that id.")

    def test_tool_is_registered_and_risky(self):
        tb = toolbox()
        self.assertIn("run_task_now", tb.funcs)
        self.assertTrue(tb.needs_approval("run_task_now", {"job_id": "x"}))

    def test_panel_contract_has_run_button_and_post(self):
        html = (Path(__file__).resolve().parent.parent / "static"
                / "index.html").read_text()
        self.assertIn('data-action="run"', html)
        self.assertIn("'/run'", html)
        self.assertIn("method:'POST'", html)


# Server endpoint tests need the repo's own import pattern (see
# tests/test_server.py): pin the web credentials to temp values so
# importing server.py never writes a .env into the repo, and stand in
# a FakeVIKI for the Ollama-backed agent, which is unavailable here.
import os as _os

_TMP = tempfile.mkdtemp(prefix="viki-run-now-")
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
class TestTasksRunNowEndpoint(unittest.TestCase):
    def test_post_run_fires_job_and_404_unknown(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        job = sched.add("Morning", "daily", at="09:00")
        old_sched, old_exec = server_mod.task_scheduler, server_mod._run_connection_job
        server_mod.task_scheduler = sched
        server_mod._run_connection_job = lambda j: "panel run ok"
        try:
            client = server_mod.app.test_client()
            with client.session_transaction() as sess:
                sess["viki_auth"] = True
            resp = client.post(f"/api/tasks/{job['id']}/run")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json()["outcome"]["result"],
                             "panel run ok")
            self.assertIsNotNone(sched.list_tasks()[0]["last_run"])
            self.assertEqual(client.post("/api/tasks/nope/run").status_code,
                             404)
        finally:
            server_mod.task_scheduler = old_sched
            server_mod._run_connection_job = old_exec


if __name__ == "__main__":
    unittest.main()
