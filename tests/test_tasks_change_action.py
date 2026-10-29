"""Tests for changing what a scheduled task does (v0.3.30).

Rename, reschedule, and duplicate all landed, but switching a check
from one job to another — e.g. connection check to urgent-email
check — still meant deleting it and recreating it, losing its run
history. Now Scheduler.change_action, the change_action_task chat
tool, PATCH /api/tasks/<id>, and a panel button all change the action
in place: the id, name, schedule, and history stay exactly as they
were, and an unknown action is rejected before anything changes.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import SessionStore
from scheduler import Scheduler, TASK_ACTIONS
from tools import Toolbox


def toolbox():
    tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
    tb.scheduler = Scheduler(folder=tempfile.mkdtemp())
    return tb


class TestChangeAction(unittest.TestCase):
    def test_change_keeps_id_name_schedule_and_history(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", action="connection-check",
                    at="09:00")
        s.run_one(job["id"], lambda j: "checked ok")
        self.assertTrue(s.change_action(job["id"], "urgent-email-check"))
        stored = s.list_tasks()[0]
        self.assertEqual(stored["id"], job["id"])
        self.assertEqual(stored["name"], "Morning")
        self.assertEqual(stored["at"], "09:00")
        self.assertEqual(stored["action"], "urgent-email-check")
        self.assertEqual(stored["run_count"], 1)
        self.assertEqual(stored["last_result"], "checked ok")

    def test_change_unknown_id_returns_false(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        self.assertFalse(s.change_action("nope", "urgent-email-check"))

    def test_change_unknown_action_raises_and_changes_nothing(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        job = s.add("Morning", "daily", action="connection-check",
                    at="09:00")
        with self.assertRaises(ValueError):
            s.change_action(job["id"], "dance-party")
        self.assertEqual(s.list_tasks()[0]["action"], "connection-check")

    def test_builtin_actions_are_the_executor_actions(self):
        self.assertEqual(TASK_ACTIONS,
                         ("connection-check", "urgent-email-check",
                          "follow-up-nudges"))


class TestChangeActionTaskTool(unittest.TestCase):
    def test_tool_changes_and_reports(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily",
                               action="connection-check", at="09:00")
        text = tb._change_action_task(job["id"], "follow-up-nudges")
        self.assertEqual(
            text,
            "Changed 'Morning' from 'connection-check' "
            "to 'follow-up-nudges'.")
        self.assertEqual(tb.scheduler.list_tasks()[0]["action"],
                         "follow-up-nudges")

    def test_tool_unknown_id(self):
        tb = toolbox()
        self.assertEqual(tb._change_action_task("nope", "urgent-email-check"),
                         "No task with that id.")

    def test_tool_unknown_action(self):
        tb = toolbox()
        job = tb.scheduler.add("Morning", "daily",
                               action="connection-check", at="09:00")
        self.assertIn("Could not change",
                      tb._change_action_task(job["id"], "dance-party"))
        self.assertEqual(tb.scheduler.list_tasks()[0]["action"],
                         "connection-check")

    def test_tool_is_registered_and_risky(self):
        tb = toolbox()
        self.assertIn("change_action_task", tb.funcs)
        self.assertTrue(tb.needs_approval("change_action_task",
                                          {"job_id": "x",
                                           "action": "urgent-email-check"}))

    def test_panel_contract_has_change_action_button(self):
        html = (Path(__file__).resolve().parent.parent / "static"
                / "index.html").read_text()
        self.assertIn('data-action="change-action"', html)
        self.assertIn("prompt('Change what this check does", html)
        self.assertIn("${esc(t.action||'')}", html)
        self.assertIn("method:'PATCH'", html)


# Server endpoint tests need the repo's own import pattern (see
# tests/test_server.py): pin the web credentials to temp values so
# importing server.py never writes a .env into the repo, and stand in
# a FakeVIKI for the Ollama-backed agent, which is unavailable here.
import os as _os

_TMP = tempfile.mkdtemp(prefix="viki-change-action-")
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
class TestTasksChangeActionEndpoint(unittest.TestCase):
    def _client(self):
        client = server_mod.app.test_client()
        with client.session_transaction() as sess:
            sess["viki_auth"] = True
        return client

    def test_patch_change_action_and_errors(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        job = sched.add("Morning", "daily", action="connection-check",
                        at="09:00")
        sched.run_one(job["id"], lambda j: "checked ok")
        old_sched = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        try:
            client = self._client()
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"action": "urgent-email-check"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json()["action"],
                             "urgent-email-check")
            stored = sched.list_tasks()[0]
            self.assertEqual(stored["action"], "urgent-email-check")
            self.assertEqual(stored["at"], "09:00")
            self.assertEqual(stored["run_count"], 1)
            self.assertEqual(client.patch("/api/tasks/nope",
                                          json={"action": "urgent-email-check"}).status_code,
                             404)
            self.assertEqual(client.patch(f"/api/tasks/{job['id']}",
                                          json={"action": "dance-party"}).status_code,
                             400)
            # A bad action combined with a rename changes nothing.
            self.assertEqual(client.patch(f"/api/tasks/{job['id']}",
                                          json={"name": "Renamed",
                                                "action": "dance-party"}).status_code,
                             400)
            self.assertEqual(sched.list_tasks()[0]["name"], "Morning")
            # Rename + action change can combine in one call.
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"name": "Evening",
                                      "action": "follow-up-nudges"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(sched.list_tasks()[0]["name"], "Evening")
            self.assertEqual(sched.list_tasks()[0]["action"],
                             "follow-up-nudges")
            # Toggle still works through the same PATCH route.
            resp = client.patch(f"/api/tasks/{job['id']}",
                                json={"enabled": False})
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(resp.get_json()["enabled"])
        finally:
            server_mod.task_scheduler = old_sched


if __name__ == "__main__":
    unittest.main()
