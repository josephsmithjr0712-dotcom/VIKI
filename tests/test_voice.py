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


class TestListenForFollowup(TestCase):
    def _cfg(self):
        return replace(vl.VoiceConfig(), silence_seconds=0.24,
                       followup_seconds=0.24, max_seconds=30)  # 3-frame window

    def test_speech_in_window_is_recorded(self):
        frames = [frame(8000)] * 2 + [frame(0)] * 20
        audio = vl.listen_for_followup(iter(frames), self._cfg())
        # 2 loud + 3 silent tail frames, then stop
        self.assertEqual(len(audio), (2 + 3) * vl.FRAME_SAMPLES * 2)

    def test_silence_through_window_returns_none(self):
        frames = [frame(0)] * 50
        self.assertIsNone(vl.listen_for_followup(iter(frames), self._cfg()))

    def test_stream_ending_in_window_returns_none(self):
        self.assertIsNone(vl.listen_for_followup(iter([frame(0)]), self._cfg()))

    def test_disabled_window_consumes_nothing(self):
        class Exploding:
            def __iter__(self):
                return self

            def __next__(self):
                raise AssertionError("disabled window must not read frames")

        cfg = replace(vl.VoiceConfig(), followup_seconds=0)
        self.assertIsNone(vl.listen_for_followup(Exploding(), cfg))


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

    def _run_loop(self, frames, texts):
        """Drive VoiceLoop.run() with fakes; the frame stream ends by raising
        KeyboardInterrupt, which run() catches as a clean stop."""
        cfg = replace(vl.VoiceConfig(), silence_seconds=0.24,
                      followup_seconds=0.24, max_seconds=30)

        def mic_frames():
            yield from frames
            raise KeyboardInterrupt

        mic = MagicMock()
        mic.frames.return_value = mic_frames()
        heard = {"n": 0}

        def heard_fn(frame):
            heard["n"] += 1
            return heard["n"] == 1  # wake word only on the first frame

        wake = MagicMock()
        wake.heard.side_effect = heard_fn
        transcriber = MagicMock()
        transcriber.transcribe.side_effect = list(texts)
        spoken = []
        speaker = MagicMock()
        speaker.speak.side_effect = lambda t: spoken.append(t)
        agent_texts = []

        def agent_runner(text):
            agent_texts.append(text)
            return "reply to: " + text

        loop = vl.VoiceLoop(cfg=cfg, mic=mic, wake=wake, transcriber=transcriber,
                            speaker=speaker, agent_runner=agent_runner, lamp=False)
        loop.run()
        return agent_texts, spoken

    def test_run_followup_reply_skips_wake_word(self):
        loud, silent = frame(8000), frame(0)
        frames = ([loud]  # wake frame
                  + [loud] * 3 + [silent] * 3  # first utterance
                  + [loud] * 2 + [silent] * 3  # follow-up reply
                  + [silent] * 3)  # silence through the next window
        agent_texts, spoken = self._run_loop(frames, ["first", "second"])
        self.assertEqual(agent_texts, ["first", "second"])
        # "Yes, boss?" once only — the follow-up did not re-prompt it
        self.assertEqual(spoken, ["Yes, boss?", "reply to: first", "reply to: second"])

    def test_run_silence_through_window_returns_to_wake(self):
        loud, silent = frame(8000), frame(0)
        frames = ([loud]  # wake frame
                  + [loud] * 3 + [silent] * 3  # first utterance
                  + [silent] * 3)  # silence through the follow-up window
        agent_texts, spoken = self._run_loop(frames, ["first"])
        self.assertEqual(agent_texts, ["first"])
        self.assertEqual(spoken, ["Yes, boss?", "reply to: first"])

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
