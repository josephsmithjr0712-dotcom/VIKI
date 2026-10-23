"""Tests for the voice loop. Audio hardware and ML models are mocked;
only the pure logic and the orchestration are exercised."""
import array
import sys
import tempfile
import wave
from dataclasses import replace
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import voice_loop as vl


def frame(value: int, n: int = vl.FRAME_SAMPLES) -> bytes:
    return array.array("h", [value] * n).tobytes()


class TestAudioLogic(TestCase):
    def test_mean_abs(self):
        self.assertEqual(vl._mean_abs(frame(0)), 0)
        self.assertAlmostEqual(vl._mean_abs(frame(1000)), 1000, places=0)

    def test_record_stops_after_silence(self):
        cfg = replace(vl.VoiceConfig(), silence_seconds=0.24, max_seconds=30)  # 3 silent frames
        frames = [frame(8000)] * 5 + [frame(0)] * 20
        audio = vl.record_until_silence(iter(frames), cfg)
        # 5 loud + 3 silent frames, then stop
        self.assertEqual(len(audio), (5 + 3) * vl.FRAME_SAMPLES * 2)

    def test_record_ignores_leading_silence(self):
        cfg = replace(vl.VoiceConfig(), silence_seconds=0.24, max_seconds=30)
        frames = [frame(0)] * 50 + [frame(8000)] * 4 + [frame(0)] * 20
        audio = vl.record_until_silence(iter(frames), cfg)
        # leading silence doesn't count; stops 3 frames after speech ends
        self.assertEqual(len(audio), (50 + 4 + 3) * vl.FRAME_SAMPLES * 2)

    def test_record_respects_max_seconds(self):
        cfg = replace(vl.VoiceConfig(), silence_seconds=99, max_seconds=0.5)
        frames = [frame(8000)] * 1000
        audio = vl.record_until_silence(iter(frames), cfg)
        import math
        expected = math.ceil(0.5 * vl.SAMPLE_RATE / vl.FRAME_SAMPLES)
        self.assertEqual(len(audio), expected * vl.FRAME_SAMPLES * 2)

    def test_write_wav_roundtrip(self):
        p = Path(tempfile.mkdtemp()) / "t.wav"
        vl.write_wav(p, frame(1234) * 2)
        with wave.open(str(p), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getframerate(), vl.SAMPLE_RATE)
            self.assertEqual(w.getnframes(), vl.FRAME_SAMPLES * 2)


class TestVoicePolicy(TestCase):
    def test_voice_mode_never_approves(self):
        self.assertFalse(vl.voice_approve("c1", "gmail_send", {}))
        self.assertFalse(vl.voice_approve("c2", "write_file", {}))

    def test_check_environment_keys(self):
        env = vl.check_environment()
        self.assertIn("faster-whisper (speech to text)", env)
        self.assertIn("piper-tts (text to speech)", env)
        self.assertIn("lamp indicator", env)
        # sandbox has none of the audio stack: all should be MISSING-ish, not crash
        for k, v in env.items():
            if k == "lamp indicator":
                self.assertTrue(v == "not configured (set LAMP_ENTITY)" or v.startswith("ok ("))
            else:
                self.assertIn(v, ("ok", "MISSING", "MISSING — run --setup-voice"))


class TestVoiceLoopOrchestration(TestCase):
    def test_handle_utterance_pipeline(self):
        calls = []
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "what time is it"
        speaker = MagicMock()
        speaker.speak.side_effect = lambda t: calls.append(("speak", t))

        def agent_runner(text):
            calls.append(("agent", text))
            return "It is noon."

        loop = vl.VoiceLoop(transcriber=transcriber, speaker=speaker,
                            agent_runner=agent_runner)
        wav = Path(tempfile.mkdtemp()) / "u.wav"
        vl.write_wav(wav, frame(100))
        reply = loop.handle_utterance(wav)
        self.assertEqual(reply, "It is noon.")
        # order: agent ran on the transcript, then the reply was spoken
        self.assertEqual(calls, [("agent", "what time is it"), ("speak", "It is noon.")])
        # conversation history kept for follow-ups
        self.assertEqual(len(loop._history), 2)

    def test_empty_transcript_says_nothing(self):
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "   "
        speaker = MagicMock()
        agent = MagicMock()
        loop = vl.VoiceLoop(transcriber=transcriber, speaker=speaker, agent_runner=agent)
        wav = Path(tempfile.mkdtemp()) / "u.wav"
        vl.write_wav(wav, frame(100))
        self.assertEqual(loop.handle_utterance(wav), "")
        agent.assert_not_called()
        speaker.speak.assert_not_called()


if __name__ == "__main__":
    main(verbosity=2)
