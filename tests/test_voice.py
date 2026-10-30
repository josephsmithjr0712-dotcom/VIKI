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

class TestDefaultAgentRunner(TestCase):
    """Regression tests for v0.3.17: the headless voice path used to call
    VIKI(model=..., tools=...) — which __init__ rejects — and read a
    "text" key that run() never returns, so it could never answer."""

    def _runner_with(self, run_result):
        from unittest.mock import patch
        loop = vl.VoiceLoop(transcriber=MagicMock(), speaker=MagicMock(),
                            agent_runner=None, lamp=False)
        # _default_agent_runner builds its own stages; stub them out.
        fake_viki = MagicMock()
        fake_viki.run.return_value = run_result
        viki_cls = patch("agent.VIKI", return_value=fake_viki).start()
        toolbox_cls = patch("tools.Toolbox").start()
        patch("memory.SessionStore").start()
        self.addCleanup(patch.stopall)
        return loop, fake_viki, viki_cls, toolbox_cls

    def test_default_runner_returns_agent_content(self):
        loop, fake_viki, viki_cls, toolbox_cls = self._runner_with(
            {"type": "text", "content": "It is noon."})
        self.assertEqual(loop._default_agent_runner("what time is it"), "It is noon.")
        # VIKI is constructed with exactly one positional arg: the toolbox.
        viki_cls.assert_called_once_with(toolbox_cls.return_value)
        fake_viki.run.assert_called_once()

    def test_default_runner_falls_back_only_on_empty_content(self):
        loop, _, _, _ = self._runner_with({"type": "text", "content": ""})
        self.assertEqual(loop._default_agent_runner("hello"),
                         "Hmm, I came up empty on that one.")

    def test_default_runner_approval_denied_message(self):
        loop, _, _, _ = self._runner_with(
            {"type": "approval_denied", "content": ""})
        self.assertEqual(loop._default_agent_runner("send an email"),
                         "That needs your approval, boss — "
                         "tap Allow in the web UI and ask me again.")


class TestVoiceLoopOrchestrationEmpty(TestCase):
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


class TestVoiceConfigValidation(TestCase):
    """Finding 52: env knobs fail at construction with a clear message."""

    def test_mic_non_integer_env(self):
        from unittest.mock import patch
        import os
        with patch.dict(os.environ, {"VIKI_MIC": "abc"}):
            with self.assertRaises(ValueError) as cm:
                vl.VoiceConfig()
        self.assertIn("VIKI_MIC", str(cm.exception))

    def test_mic_integer_env_ok(self):
        from unittest.mock import patch
        import os
        with patch.dict(os.environ, {"VIKI_MIC": "2"}):
            self.assertEqual(vl.VoiceConfig().mic_device, 2)

    def test_followup_non_numeric_env(self):
        from unittest.mock import patch
        import os
        with patch.dict(os.environ, {"VIKI_FOLLOWUP_SECONDS": "soon"}):
            with self.assertRaises(ValueError) as cm:
                vl.VoiceConfig()
        self.assertIn("VIKI_FOLLOWUP_SECONDS", str(cm.exception))

    def test_followup_negative_rejected(self):
        with self.assertRaises(ValueError) as cm:
            vl.VoiceConfig(followup_seconds=-1)
        self.assertIn("VIKI_FOLLOWUP_SECONDS", str(cm.exception))

    def test_followup_zero_ok(self):
        self.assertEqual(vl.VoiceConfig(followup_seconds=0).followup_seconds, 0)

    def test_threshold_ranges(self):
        for kwargs in ({"wake_threshold": 1.5}, {"wake_threshold": -0.1},
                       {"silence_threshold": -1}, {"silence_threshold": 40000},
                       {"silence_seconds": 0}, {"max_seconds": 0},
                       {"initial_silence_seconds": 0},
                       {"silence_threshold": "loud"}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                vl.VoiceConfig(**kwargs)

    def test_defaults_still_valid(self):
        cfg = vl.VoiceConfig()
        self.assertEqual(cfg.wake_threshold, 0.5)
        self.assertEqual(cfg.followup_seconds, 10)
        self.assertEqual(cfg.initial_silence_seconds, 5.0)


class TestInitialSilenceTimeout(TestCase):
    """Finding 39a: silence before any speech stops recording early."""

    def test_initial_silence_stops_before_max(self):
        cfg = replace(vl.VoiceConfig(), silence_seconds=0.24,
                      initial_silence_seconds=0.24, max_seconds=30)
        audio = vl.record_until_silence(iter([frame(0)] * 50), cfg)
        # 3 silent frames (0.24s), not all 50 and not the 30s max cap
        self.assertEqual(len(audio), 3 * vl.FRAME_SAMPLES * 2)

    def test_speech_after_initial_timeout_is_not_captured(self):
        cfg = replace(vl.VoiceConfig(), silence_seconds=0.24,
                      initial_silence_seconds=0.24, max_seconds=30)
        frames = [frame(0)] * 20 + [frame(8000)] * 4 + [frame(0)] * 20
        audio = vl.record_until_silence(iter(frames), cfg)
        self.assertEqual(len(audio), 3 * vl.FRAME_SAMPLES * 2)


class TestHistoryCap(TestCase):
    """Finding 39b: _history keeps only the recent window."""

    def test_history_capped_at_20(self):
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "hi"
        speaker = MagicMock()
        loop = vl.VoiceLoop(transcriber=transcriber, speaker=speaker,
                            agent_runner=lambda t: "reply", lamp=False)
        wav = Path(tempfile.mkdtemp()) / "u.wav"
        vl.write_wav(wav, frame(100))
        for _ in range(15):  # 30 messages without a cap
            loop.handle_utterance(wav)
        self.assertEqual(len(loop._history), vl.HISTORY_MAX_MESSAGES)
        self.assertEqual(len(loop._history), 20)


class _FakeLamp:
    def __init__(self):
        self.states = []

    def set(self, state):
        self.states.append(state)


def _loop_with_transcript(transcript=None, agent=None, speaker=None,
                          exc=None):
    transcriber = MagicMock()
    if exc == "transcribe":
        transcriber.transcribe.side_effect = RuntimeError("mic boom")
    else:
        transcriber.transcribe.return_value = transcript
    speaker = speaker or MagicMock()
    if exc == "speak":
        speaker.speak.side_effect = RuntimeError("speaker boom")
    agent_runner = agent
    if exc == "agent":
        def agent_runner(text):
            raise RuntimeError("agent boom")
    loop = vl.VoiceLoop(transcriber=transcriber, speaker=speaker,
                        agent_runner=agent_runner, lamp=False)
    wav = Path(tempfile.mkdtemp()) / "u.wav"
    vl.write_wav(wav, frame(100))
    return loop, loop.handle_utterance(wav)


class TestStageGuards(TestCase):
    """Finding 36: a stage failure restores the lamp and is not a crash."""

    def test_transcribe_failure_is_caught_lamp_restored(self):
        lamp = _FakeLamp()
        transcriber = MagicMock()
        transcriber.transcribe.side_effect = RuntimeError("mic boom")
        loop = vl.VoiceLoop(transcriber=transcriber, speaker=MagicMock(),
                            agent_runner=MagicMock(), lamp=lamp)
        wav = Path(tempfile.mkdtemp()) / "u.wav"
        vl.write_wav(wav, frame(100))
        self.assertEqual(loop.handle_utterance(wav), "")
        self.assertTrue(loop._last_stage_failed)
        self.assertEqual(lamp.states[-1], "restore")
        loop.agent_runner.assert_not_called()

    def test_agent_failure_is_caught_no_history_no_speak(self):
        speaker = MagicMock()
        loop, reply = _loop_with_transcript("hello", speaker=speaker, exc="agent")
        self.assertEqual(reply, "")
        self.assertTrue(loop._last_stage_failed)
        self.assertEqual(loop._history, [])
        speaker.speak.assert_not_called()

    def test_agent_failure_restores_lamp(self):
        lamp = _FakeLamp()
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "hello"

        def boom(text):
            raise RuntimeError("agent boom")
        loop = vl.VoiceLoop(transcriber=transcriber, speaker=MagicMock(),
                            agent_runner=boom, lamp=lamp)
        wav = Path(tempfile.mkdtemp()) / "u.wav"
        vl.write_wav(wav, frame(100))
        self.assertEqual(loop.handle_utterance(wav), "")
        self.assertEqual(lamp.states[-1], "restore")

    def test_speak_failure_keeps_history_lamp_restored(self):
        lamp = _FakeLamp()
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "hello"
        speaker = MagicMock()
        speaker.speak.side_effect = RuntimeError("speaker boom")
        loop = vl.VoiceLoop(transcriber=transcriber, speaker=speaker,
                            agent_runner=lambda t: "hi there", lamp=lamp)
        wav = Path(tempfile.mkdtemp()) / "u.wav"
        vl.write_wav(wav, frame(100))
        self.assertEqual(loop.handle_utterance(wav), "")
        self.assertTrue(loop._last_stage_failed)
        self.assertEqual(len(loop._history), 2)  # agent work is not lost
        self.assertEqual(lamp.states[-1], "restore")

    def test_skips_are_success_not_failures(self):
        for transcript in ("   ", "stop"):
            loop, reply = _loop_with_transcript(transcript)
            self.assertEqual(reply, "")
            self.assertFalse(loop._last_stage_failed)

    def _run_loop_with_failures(self, transcribe_side_effect, wake_hits):
        """Drive run() with fakes; stream ends with KeyboardInterrupt."""
        cfg = replace(vl.VoiceConfig(), silence_seconds=0.24,
                      followup_seconds=0.24, max_seconds=30)
        loud, silent = frame(8000), frame(0)
        # Two wake turns, each: wake frame + loud + 3 silent. Second turn
        # is followed by a silent follow-up window, then the stream ends.
        frames = ([loud] + [loud] * 3 + [silent] * 3
                  + [loud] + [loud] * 3 + [silent] * 3
                  + [silent] * 3)

        def mic_frames():
            yield from frames
            raise KeyboardInterrupt

        mic = MagicMock()
        mic.frames.return_value = mic_frames()
        heard = {"n": 0}

        def heard_fn(frame):
            heard["n"] += 1
            return heard["n"] in wake_hits
        wake = MagicMock()
        wake.heard.side_effect = heard_fn
        transcriber = MagicMock()
        transcriber.transcribe.side_effect = transcribe_side_effect
        spoken = []
        speaker = MagicMock()
        speaker.speak.side_effect = lambda t: spoken.append(t)
        agent_texts = []

        def agent_runner(text):
            agent_texts.append(text)
            return "reply to: " + text

        sleeps = []
        loop = vl.VoiceLoop(cfg=cfg, mic=mic, wake=wake, transcriber=transcriber,
                            speaker=speaker, agent_runner=agent_runner,
                            lamp=False, sleep_fn=sleeps.append)
        loop.run()
        return agent_texts, spoken, sleeps, loop

    def test_run_continues_after_failed_utterance(self):
        # First turn's transcribe fails; the loop must still answer the
        # second wake turn (and say "Yes, boss?" for it) instead of dying.
        agent_texts, spoken, sleeps, loop = self._run_loop_with_failures(
            [RuntimeError("transcribe boom"), "second"], wake_hits={1, 2})
        self.assertEqual(agent_texts, ["second"])
        self.assertEqual(spoken.count("Yes, boss?"), 2)
        self.assertIn("reply to: second", spoken)
        self.assertEqual(sleeps, [])  # first consecutive failure: no backoff
        self.assertEqual(loop._consecutive_failures, 0)  # success reset it

    def test_run_backs_off_after_repeated_failures(self):
        agent_texts, spoken, sleeps, loop = self._run_loop_with_failures(
            [RuntimeError("boom"), RuntimeError("boom")], wake_hits={1, 2})
        self.assertEqual(agent_texts, [])
        self.assertEqual(spoken.count("Yes, boss?"), 2)
        # No sleep after the 1st failure, 0.25s after the 2nd — no spin.
        self.assertEqual(sleeps, [0.25])
        self.assertEqual(loop._consecutive_failures, 2)

    def test_yes_boss_failure_still_listens(self):
        cfg = replace(vl.VoiceConfig(), silence_seconds=0.24,
                      followup_seconds=0.24, max_seconds=30)
        loud, silent = frame(8000), frame(0)
        frames = [loud] + [loud] * 3 + [silent] * 3 + [silent] * 3

        def mic_frames():
            yield from frames
            raise KeyboardInterrupt

        mic = MagicMock()
        mic.frames.return_value = mic_frames()
        wake = MagicMock()
        wake.heard.return_value = True
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "hello"
        calls = []

        def speak(text):
            calls.append(text)
            if text == "Yes, boss?":
                raise RuntimeError("mini down")
        speaker = MagicMock()
        speaker.speak.side_effect = speak
        loop = vl.VoiceLoop(cfg=cfg, mic=mic, wake=wake, transcriber=transcriber,
                            speaker=speaker, agent_runner=lambda t: "hi",
                            lamp=False)
        loop.run()  # must not raise; the utterance is still answered
        self.assertIn("hi", calls)


class TestWakeWordScoreKey(TestCase):
    """Finding 37: score key from the loaded model, fail fast if unknown."""

    def _fake_model_cls(self, scores, names=None):
        class FakeModel:
            def __init__(self, wakeword_models, inference_framework):
                self.wakeword_models = wakeword_models
                self.inference_framework = inference_framework
                if names is not None:
                    self.wakeword_model_names = names
                self._scores = scores

            def predict(self, audio):
                return dict(self._scores)
        return FakeModel

    def test_custom_model_stem_is_score_key(self):
        from unittest.mock import patch
        import os
        cfg = vl.VoiceConfig()  # default wakeword hey_jarvis
        fake = self._fake_model_cls({"hey_viki": 0.9})
        with patch.dict(os.environ, {"VIKI_WAKEWORD_MODEL": "/tmp/hey_viki.onnx"}):
            det = vl.WakeWordDetector(cfg, model_cls=fake)
        self.assertEqual(det.score_key, "hey_viki")
        self.assertTrue(det.heard(frame(100)))
        # The same score below threshold does not trigger.
        fake_low = self._fake_model_cls({"hey_viki": 0.1})
        with patch.dict(os.environ, {"VIKI_WAKEWORD_MODEL": "/tmp/hey_viki.onnx"}):
            det_low = vl.WakeWordDetector(cfg, model_cls=fake_low)
        self.assertFalse(det_low.heard(frame(100)))

    def test_model_known_name_beats_file_stem(self):
        from unittest.mock import patch
        import os
        cfg = vl.VoiceConfig()
        fake = self._fake_model_cls({"hey_viki_custom": 0.9},
                                    names=["hey_viki_custom"])
        with patch.dict(os.environ, {"VIKI_WAKEWORD_MODEL": "/tmp/other.onnx"}):
            det = vl.WakeWordDetector(cfg, model_cls=fake)
        self.assertEqual(det.score_key, "hey_viki_custom")
        self.assertTrue(det.heard(frame(100)))

    def test_default_hey_jarvis_path_unchanged(self):
        from unittest.mock import patch
        import os
        cfg = vl.VoiceConfig()
        seen = {}

        class FakeModel:
            def __init__(self, wakeword_models, inference_framework):
                seen["models"] = wakeword_models

            def predict(self, audio):
                return {"hey_jarvis": 0.9}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VIKI_WAKEWORD_MODEL", None)
            det = vl.WakeWordDetector(cfg, model_cls=FakeModel)
        self.assertEqual(seen["models"], ["hey_jarvis"])
        self.assertEqual(det.score_key, "hey_jarvis")
        self.assertTrue(det.heard(frame(100)))

    def test_unknown_wakeword_fails_fast(self):
        cfg = vl.VoiceConfig(wakeword="hey_alexa")
        fake = self._fake_model_cls({"hey_jarvis": 0.9}, names=["hey_jarvis"])
        with self.assertRaises(ValueError) as cm:
            vl.WakeWordDetector(cfg, model_cls=fake)
        self.assertIn("hey_alexa", str(cm.exception))
        self.assertIn("hey_jarvis", str(cm.exception))


if __name__ == "__main__":
    main(verbosity=2)
