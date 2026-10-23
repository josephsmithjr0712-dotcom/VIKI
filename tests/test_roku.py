"""Tests for Roku control via Home Assistant. HA's REST API is mocked."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.roku as rk

HA_ENV = {"HA_URL": "http://ha:8123", "HA_TOKEN": "tok"}
NO_HA_ENV = {"HA_URL": "", "HA_TOKEN": ""}
ENT = ("media_player.roku", "remote.roku")


class FakeResp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def states_payload():
    return [
        {"entity_id": "media_player.roku", "state": "idle",
         "attributes": {"app_name": "Roku", "source_list": ["Netflix", "Plex"]}},
        {"entity_id": "remote.roku", "state": "on", "attributes": {}},
        {"entity_id": "light.kitchen", "state": "on", "attributes": {}},
    ]


class TestRoku(TestCase):
    def setUp(self):
        rk.reset_cache()
        self.env = patch.dict(os.environ, HA_ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def tearDown(self):
        rk.reset_cache()

    def test_entities_discovery(self):
        with patch("urllib.request.urlopen", return_value=FakeResp(states_payload())):
            self.assertEqual(rk._entities(), ENT)

    def test_entities_env_override(self):
        with patch.dict(os.environ, {"ROKU_ENTITY": "media_player.bedroom_roku",
                                     "ROKU_REMOTE": "remote.bedroom_roku"}):
            with patch("urllib.request.urlopen", return_value=FakeResp(states_payload())):
                self.assertEqual(rk._entities(),
                                 ("media_player.bedroom_roku", "remote.bedroom_roku"))

    def test_entities_none_found(self):
        with patch("urllib.request.urlopen", return_value=FakeResp(
                [{"entity_id": "light.kitchen", "state": "on", "attributes": {}}])):
            with self.assertRaises(RuntimeError) as ctx:
                rk._entities()
        self.assertIn("No Roku media_player", str(ctx.exception))

    def test_status(self):
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_state", return_value={
                 "state": "playing",
                 "attributes": {"app_name": "Plex", "media_title": "Dune"}}):
            out = rk.roku_status()
        self.assertIn("State: playing", out)
        self.assertIn("App: Plex", out)
        self.assertIn("Now playing: Dune", out)

    def test_apps(self):
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_state", return_value={
                 "state": "idle", "attributes": {"source_list": ["Netflix", "Plex"]}}):
            out = rk.roku_apps()
        self.assertIn("- Netflix", out)
        self.assertIn("- Plex", out)

    def test_launch(self):
        svc = MagicMock()
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_service", svc):
            out = rk.roku_launch("Netflix")
        svc.assert_called_once_with("media_player", "select_source",
                                    {"entity_id": "media_player.roku", "source": "Netflix"})
        self.assertIn("Netflix", out)

    def test_search_uses_ha_schema(self):
        svc = MagicMock()
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_service", svc):
            out = rk.roku_search("Dune")
        # exact schema from HA docs: roku.search + entity_id + keyword
        svc.assert_called_once_with("roku", "search",
                                    {"entity_id": "media_player.roku", "keyword": "Dune"})
        self.assertIn("Dune", out)

    def test_key(self):
        svc = MagicMock()
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_service", svc):
            self.assertEqual(rk.roku_key("select"), "Pressed 'select'.")
        svc.assert_called_once_with("remote", "send_command",
                                    {"entity_id": "remote.roku", "command": ["select"]})

    def test_key_invalid(self):
        svc = MagicMock()
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_service", svc):
            out = rk.roku_key("diagonal")
        self.assertIn("Unknown key", out)
        svc.assert_not_called()

    def test_key_lit_passthrough(self):
        svc = MagicMock()
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_service", svc):
            rk.roku_key("Lit_hello")
        svc.assert_called_once_with("remote", "send_command",
                                    {"entity_id": "remote.roku", "command": ["Lit_hello"]})

    def test_type(self):
        svc = MagicMock()
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_service", svc):
            out = rk.roku_type("dune part two")
        svc.assert_called_once_with("remote", "send_command",
                                    {"entity_id": "remote.roku", "command": ["Lit_dune part two"]})
        self.assertIn("dune part two", out)

    def test_play_movie_sequence(self):
        svc = MagicMock()
        with patch.object(rk, "_entities", return_value=ENT), \
             patch.object(rk, "_service", svc), \
             patch.object(rk, "_state", return_value={
                 "state": "playing",
                 "attributes": {"app_name": "Netflix", "media_title": "Dune: Part Two"}}), \
             patch("connectors.roku.time.sleep"):
            out = rk.roku_play_movie("Dune Part Two")
        calls = svc.call_args_list
        # first: roku universal search with the title
        self.assertEqual(calls[0][0][:2], ("roku", "search"))
        self.assertEqual(calls[0][0][2]["keyword"], "Dune Part Two")
        # then: remote navigation down -> select -> select -> play
        keys = [c[0][2]["command"][0] for c in calls[1:]]
        self.assertEqual(keys, ["down", "select", "select", "play"])
        for c in calls[1:]:
            self.assertEqual(c[0][:2], ("remote", "send_command"))
        self.assertIn("Dune: Part Two", out)

    def test_no_ha_creds(self):
        with patch.dict(os.environ, NO_HA_ENV):
            out = rk.roku_status()
        self.assertIn("HA_URL and HA_TOKEN", out)

    def test_toolbox_gating(self):
        from tools import Toolbox, ROKU_TOOLS
        from memory import SessionStore
        import tools as tools_mod

        with patch.object(tools_mod, "ha_connected", return_value=False):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        for t in ROKU_TOOLS:
            self.assertNotIn(t, tb.enabled)
        with patch.object(tools_mod, "ha_connected", return_value=True):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s2"))
        for t in ROKU_TOOLS:
            self.assertIn(t, tb.enabled)
        # playing a movie on his own TV is on-demand and reversible: no approval
        self.assertFalse(tb.needs_approval("roku_play_movie", {"title": "Dune"}))
        self.assertFalse(tb.needs_approval("roku_key", {"key": "select"}))


if __name__ == "__main__":
    main(verbosity=2)
