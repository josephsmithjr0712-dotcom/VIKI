"""Tests for the Plex connector. plexapi is mocked."""
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.plex as px

PLEX_ENV = {"PLEX_URL": "http://plex:32400", "PLEX_TOKEN": "tok"}


def fake_movie(title="Dune", year=2021):
    m = MagicMock()
    m.title = title
    m.year = year
    m.TYPE = "movie"
    m.seasonNumber = None
    m.episodeNumber = None
    return m


def fake_episode():
    e = MagicMock()
    e.title = "Winter Is Coming"
    e.year = 2011
    e.TYPE = "episode"
    e.seasonNumber = 1
    e.episodeNumber = 1
    return e


def fake_server():
    return MagicMock()


class TestPlex(TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, PLEX_ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def _server(self, server):
        return patch.object(px, "_server", return_value=server)

    def test_libraries(self):
        server = fake_server()
        s1, s2 = MagicMock(), MagicMock()
        s1.title, s1.type = "Movies", "movie"
        s2.title, s2.type = "TV Shows", "show"
        server.library.sections.return_value = [s1, s2]
        with self._server(server):
            out = px.plex_libraries()
        self.assertIn("Movies (movie)", out)
        self.assertIn("TV Shows (show)", out)

    def test_search(self):
        server = fake_server()
        server.library.search.return_value = [fake_movie(), fake_episode()]
        with self._server(server):
            out = px.plex_search("dune")
        server.library.search.assert_called_once_with("dune")
        self.assertIn("Dune (2021) [movie]", out)
        self.assertIn("Winter Is Coming (2011) S01E01 [episode]", out)

    def test_search_empty(self):
        server = fake_server()
        server.library.search.return_value = []
        with self._server(server):
            self.assertIn("No Plex matches", px.plex_search("zzz"))

    def test_on_deck(self):
        server = fake_server()
        server.library.onDeck.return_value = [fake_episode()]
        with self._server(server):
            out = px.plex_on_deck()
        self.assertIn("Winter Is Coming", out)

    def test_players(self):
        server = fake_server()
        c = MagicMock()
        c.title, c.product, c.address = "Living Room Roku", "Plex for Roku", "192.168.1.20"
        server.clients.return_value = [c]
        with self._server(server):
            out = px.plex_players()
        self.assertIn("Living Room Roku", out)

    def test_players_none(self):
        server = fake_server()
        server.clients.return_value = []
        with self._server(server):
            out = px.plex_players()
        self.assertIn("No Plex players seen", out)

    def test_play_picks_player_by_name(self):
        server = fake_server()
        movie = fake_movie()
        server.library.search.return_value = [movie]
        roku = MagicMock()
        roku.title = "Living Room Roku"
        phone = MagicMock()
        phone.title = "Pixel"
        server.clients.return_value = [phone, roku]
        with self._server(server), \
             patch.object(px, "_maybe_launch_plex_on_roku") as launch:
            out = px.plex_play("Dune", player="roku")
        movie.play.assert_called_once_with(roku)
        launch.assert_called_once_with("Living Room Roku")
        self.assertIn("Living Room Roku", out)

    def test_play_no_match(self):
        server = fake_server()
        server.library.search.return_value = []
        server.clients.return_value = [MagicMock(title="Roku")]
        with self._server(server):
            out = px.plex_play("zzz")
        self.assertIn("No Plex matches", out)

    def test_play_unknown_player(self):
        server = fake_server()
        server.library.search.return_value = [fake_movie()]
        c = MagicMock()
        c.title = "Pixel"
        server.clients.return_value = [c]
        with self._server(server):
            out = px.plex_play("Dune", player="roku")
        self.assertIn("No Plex player matching 'roku'", out)

    def test_pause(self):
        server = fake_server()
        c = MagicMock()
        c.title = "Living Room Roku"
        server.clients.return_value = [c]
        with self._server(server):
            out = px.plex_pause()
        c.pause.assert_called_once()
        self.assertIn("Paused", out)

    def test_not_configured(self):
        with patch.dict(os.environ, {"PLEX_URL": "", "PLEX_TOKEN": ""}):
            self.assertIn("PLEX_URL", px.plex_libraries())

    def test_toolbox_gating(self):
        from tools import Toolbox, PLEX_TOOLS
        from memory import SessionStore
        import tools as tools_mod

        with patch.object(tools_mod, "plex_configured", return_value=False):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        for t in PLEX_TOOLS:
            self.assertNotIn(t, tb.enabled)
        with patch.object(tools_mod, "plex_configured", return_value=True):
            tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s2"))
        for t in PLEX_TOOLS:
            self.assertIn(t, tb.enabled)
        self.assertFalse(tb.needs_approval("plex_play", {"title": "Dune"}))


if __name__ == "__main__":
    main(verbosity=2)
