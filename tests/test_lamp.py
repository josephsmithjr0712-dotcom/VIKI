"""Tests for the lamp status indicator. Home Assistant is mocked."""
import os
import sys
import tempfile
import wave
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import connectors.lamp as lamp
import voice_loop as vl

HA_ENV = {"HA_URL": "http://ha:8123", "HA_TOKEN": "tok", "LAMP_ENTITY": "light.desk"}


def wav_bytes():
    import array
    return array.array("h", [100] * vl.FRAME_SAMPLES).tobytes()


class TestLampIndicator(TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, HA_ENV)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.svc = MagicMock()
        self.svc_patch = patch.object(lamp, "_service", self.svc)
        self.svc_patch.start()
        self.addCleanup(self.svc_patch.stop)

    def make(self, snapshot):
        ind = lamp.LampIndicator("light.desk")
        ind._snapshot = None
        with patch.object(lamp.LampIndicator, "_read_state", return_value=snapshot):
            ind.set("wake")
        return ind

    def test_wake_sets_purple(self):
        self.make({"on": False})
        self.svc.assert_called_once_with("light", "turn_on",
                                         {"entity_id": "light.desk", "rgb_color": [128, 0, 255]})

    def test_thinking_blue_speaking_pink(self):
        ind = self.make({"on": False})
        ind.set("thinking")
        ind.set("speaking")
        rgbs = [c[0][2]["rgb_color"] for c in self.svc.call_args_list]
        self.assertEqual(rgbs, [[128, 0, 255], [30, 144, 255], [255, 105, 180]])

    def test_snapshot_taken_once(self):
        ind = lamp.LampIndicator("light.desk")
        with patch.object(lamp.LampIndicator, "_read_state",
                          return_value={"on": False}) as rs:
            ind.set("wake")
            ind.set("thinking")
            ind.set("speaking")
        rs.assert_called_once()

    def test_restore_when_was_off(self):
        ind = self.make({"on": False})
        ind.set("restore")
        self.svc.assert_called_with("light", "turn_off", {"entity_id": "light.desk"})

    def test_restore_when_was_on(self):
        ind = self.make({"on": True, "rgb_color": [255, 255, 255], "brightness": 200})
        ind.set("restore")
        self.svc.assert_called_with("light", "turn_on",
                                    {"entity_id": "light.desk",
                                     "rgb_color": [255, 255, 255], "brightness": 200})

    def test_unknown_state(self):
        ind = lamp.LampIndicator("light.desk")
        with self.assertRaises(ValueError):
            ind.set("disco")

    def test_no_entity_configured(self):
        with patch.dict(os.environ, {"LAMP_ENTITY": ""}), \
             patch.object(lamp, "_get_states", return_value=[]):
            with self.assertRaises(RuntimeError) as ctx:
                lamp.LampIndicator()
        self.assertIn("LAMP_ENTITY", str(ctx.exception))


class TestLampDiscovery(TestCase):
    def test_env_wins(self):
        with patch.dict(os.environ, HA_ENV):
            self.assertEqual(lamp.resolve_lamp_entity(), "light.desk")

    def test_single_light_auto(self):
        with patch.dict(os.environ, {"LAMP_ENTITY": ""}), \
             patch.object(lamp, "_get_states", return_value=[
                 {"entity_id": "light.only_one", "state": "off", "attributes": {}}]):
            self.assertEqual(lamp.resolve_lamp_entity(), "light.only_one")

    def test_ambiguous_needs_env(self):
        with patch.dict(os.environ, {"LAMP_ENTITY": ""}), \
             patch.object(lamp, "_get_states", return_value=[
                 {"entity_id": "light.a", "state": "off", "attributes": {}},
                 {"entity_id": "light.b", "state": "off", "attributes": {}}]):
            self.assertIsNone(lamp.resolve_lamp_entity())

    def test_available(self):
        with patch.dict(os.environ, HA_ENV):
            self.assertTrue(lamp.available())
        with patch.dict(os.environ, {"HA_URL": "", "HA_TOKEN": "", "LAMP_ENTITY": "light.desk"}):
            self.assertFalse(lamp.available())


class TestVoiceLoopLamp(TestCase):
    def _loop(self):
        mock_lamp = MagicMock()
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "hello"
        speaker = MagicMock()
        loop = vl.VoiceLoop(transcriber=transcriber, speaker=speaker,
                            agent_runner=lambda t: "hi boss", lamp=mock_lamp)
        return loop, mock_lamp

    def _wav(self):
        p = Path(tempfile.mkdtemp()) / "u.wav"
        vl.write_wav(p, wav_bytes())
        return p

    def test_states_in_order(self):
        loop, mock_lamp = self._loop()
        loop.handle_utterance(self._wav())
        states = [c[0][0] for c in mock_lamp.set.call_args_list]
        self.assertEqual(states, ["thinking", "speaking", "restore"])

    def test_empty_transcript_restores(self):
        loop, mock_lamp = self._loop()
        loop.transcriber.transcribe.return_value = "   "
        self.assertEqual(loop.handle_utterance(self._wav()), "")
        states = [c[0][0] for c in mock_lamp.set.call_args_list]
        self.assertEqual(states, ["restore"])

    def test_lamp_failures_dont_break_voice(self):
        loop, mock_lamp = self._loop()
        mock_lamp.set.side_effect = Exception("HA down")
        # should not raise
        self.assertEqual(loop.handle_utterance(self._wav()), "hi boss")

    def test_lamp_disabled(self):
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "hello"
        loop = vl.VoiceLoop(transcriber=transcriber, speaker=MagicMock(),
                            agent_runner=lambda t: "hi", lamp=False)
        self.assertIsNone(loop._lamp_stage())
        self.assertEqual(loop.handle_utterance(self._wav()), "hi")

    def test_auto_detect_off_without_ha(self):
        # no HA_URL/HA_TOKEN here -> indicator resolves to None, voice still works
        with patch.dict(os.environ, {"HA_URL": "", "HA_TOKEN": "", "LAMP_ENTITY": ""}):
            loop = vl.VoiceLoop(transcriber=MagicMock(), speaker=MagicMock(),
                                agent_runner=lambda t: "hi")
            loop.transcriber.transcribe.return_value = "hello"
            self.assertIsNone(loop._lamp_stage())
            self.assertEqual(loop.handle_utterance(self._wav()), "hi")


if __name__ == "__main__":
    main(verbosity=2)
