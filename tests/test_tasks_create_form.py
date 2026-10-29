"""Tests for creating every kind of scheduled check from the web panel (v0.3.25).

The panel form used to hardcode kind='daily', so interval and once
checks could only be created from chat — and the form offered no way
to pick which check runs. Now the form has a kind selector (daily /
every N min / once), a minutes field, a date field, and an action
selector, and the submit handler sends the matching payload to the
POST /api/tasks endpoint, which already accepted all kinds.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler import Scheduler


def _html():
    return (Path(__file__).resolve().parent.parent / "static"
            / "index.html").read_text()


class TestCreateFormContract(unittest.TestCase):
    def test_form_has_kind_minutes_date_and_action_fields(self):
        html = _html()
        self.assertIn('id="taskkind"', html)
        for value in ('value="daily"', 'value="interval"', 'value="once"'):
            self.assertIn(value, html)
        self.assertIn('id="taskevery"', html)
        self.assertIn('id="taskdate"', html)
        self.assertIn('id="taskaction"', html)
        for action in ('value="connection-check"',
                       'value="urgent-email-check"',
                       'value="follow-up-nudges"'):
            self.assertIn(action, html)

    def test_submit_sends_matching_payload_per_kind(self):
        html = _html()
        # Interval branch: minutes, no time string.
        self.assertIn("payload.every_minutes", html)
        # Once branch: date + time joined into the server's format.
        self.assertIn("$('#taskdate').value+' '+$('#taskat').value", html)
        # Action travels with every kind.
        self.assertIn("action=$('#taskaction').value", html)
        self.assertIn("kind=$('#taskkind').value", html)

    def test_kind_selector_toggles_the_right_fields(self):
        html = _html()
        self.assertIn("$('#taskkind').onchange", html)
        self.assertIn("$('#taskevery').style.display", html)
        self.assertIn("$('#taskdate').style.display", html)

    def test_scheduler_accepts_the_payloads_the_form_sends(self):
        s = Scheduler(folder=tempfile.mkdtemp())
        self.assertEqual(s.add("Daily", "daily", at="07:00")["kind"], "daily")
        job = s.add("Frequent", "interval", every_minutes=15)
        self.assertEqual(job["every_minutes"], 15)
        job = s.add("One-off", "once", at="2026-10-29 09:00")
        self.assertEqual(job["at"], "2026-10-29 09:00")


# Server endpoint tests need the repo's own import pattern (see
# tests/test_server.py): pin the web credentials to temp values so
# importing server.py never writes a .env into the repo, and stand in
# a FakeVIKI for the Ollama-backed agent, which is unavailable here.
import os as _os

_TMP = tempfile.mkdtemp(prefix="viki-create-form-")
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
class TestTasksCreateEndpoint(unittest.TestCase):
    def _client(self, sched):
        old = server_mod.task_scheduler
        server_mod.task_scheduler = sched
        client = server_mod.app.test_client()
        with client.session_transaction() as sess:
            sess["viki_auth"] = True
        return client, old

    def test_post_creates_interval_and_once_jobs(self):
        sched = Scheduler(folder=tempfile.mkdtemp())
        client, old = self._client(sched)
        try:
            r = client.post("/api/tasks", json={
                "name": "Frequent", "kind": "interval",
                "every_minutes": 15, "action": "urgent-email-check"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.get_json()["task"]["every_minutes"], 15)
            r = client.post("/api/tasks", json={
                "name": "One-off", "kind": "once",
                "at": "2026-10-29 09:00", "action": "follow-up-nudges"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.get_json()["task"]["at"], "2026-10-29 09:00")
            self.assertEqual(len(sched.list_tasks()), 2)
            # Bad payloads still rejected, not silently created.
            self.assertEqual(client.post("/api/tasks", json={
                "name": "Bad", "kind": "interval",
                "action": "no-such-action"}).status_code, 400)
        finally:
            server_mod.task_scheduler = old


if __name__ == "__main__":
    unittest.main()
