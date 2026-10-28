"""Tests for the hands-free upgrade (v0.3.16): web loop + voice-loop stop words.

Web assertions read static/index.html (the contract-test pattern): the loop
must recover from empty recognition ends and mic errors instead of dying
silently, let the user interrupt speech with the mic, stream sentences to
TTS instead of waiting for the whole reply, and show listening/speaking
state. Voice-loop stop words are pure logic tested with fakes — no mic,
no speaker, no network.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import voice_loop as vl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "static" / "index.html").read_text(errors="ignore")


class TestWebHandsfree(unittest.TestCase):
    def test_speech_queue_present(self):
        for snippet in ("speakQueue", "playNextSpeech", "stopSpeech",
                        "splitSentences", "maybeResumeListening"):
            self.assertIn(snippet, HTML)

    def test_sentence_streaming_in_send(self):
        # Tokens are spoken sentence-by-sentence as they stream in.
        self.assertIn("splitSentences(speakBuf)", HTML)
        self.assertIn("streamedSpeech=true", HTML)

    def test_turn_state_resets_and_old_speech_stops(self):
        self.assertIn("turnDone=false; speechInterrupted=false; stopSpeech();",
                      HTML)

    def test_auto_recovery_on_empty_end(self):
        # An empty recognition end restarts listening instead of killing
        # the hands-free loop.
        self.assertIn("setTimeout(()=>{ if(handsfreeOn) startListening(); },150)",
                      HTML)
        self.assertIn("restartSuppressed", HTML)

    def test_mic_error_handling(self):
        self.assertIn("rec.onerror", HTML)
        self.assertIn("not-allowed", HTML)
        self.assertIn("Mic blocked", HTML)
        self.assertIn("Mic trouble", HTML)

    def test_interim_results_and_state_labels(self):
        self.assertIn("interimResults=true", HTML)
        self.assertIn("Listening…", HTML)
        self.assertIn("Speaking… (tap mic to interrupt)", HTML)

    def test_mic_tap_interrupts_speech(self):
        # Tapping the mic while she speaks stops her and starts listening.
        self.assertIn(
            "speechInterrupted=true; stopSpeech(); startListening(); return;",
            HTML)


class TestStopWords(unittest.TestCase):
    def test_is_stop_command(self):
        for phrase in ("stop", "Stop.", "STOP", "cancel", "quiet",
                       "shut up", "never mind", "stop talking"):
            self.assertTrue(vl.is_stop_command(phrase), phrase)
        for phrase in ("stop my timer", "what time is it", "cancel my meeting",
                       "stopwatch", ""):
            self.assertFalse(vl.is_stop_command(phrase), phrase)

    def _loop_with_transcript(self, transcript):
        transcriber = MagicMock()
        transcriber.transcribe.return_value = transcript
        speaker = MagicMock()
        agent = MagicMock(return_value="agent reply")
        loop = vl.VoiceLoop(transcriber=transcriber, speaker=speaker,
                            agent_runner=agent, lamp=False)
        wav = Path(tempfile.mkdtemp()) / "u.wav"
        vl.write_wav(wav, b"\x00\x00" * 100)
        return loop, agent, speaker, loop.handle_utterance(wav)

    def test_stop_halts_without_agent_or_speech(self):
        loop, agent, speaker, reply = self._loop_with_transcript("stop")
        self.assertEqual(reply, "")
        agent.assert_not_called()
        speaker.speak.assert_not_called()
        self.assertEqual(loop._history, [])

    def test_stop_phrase_with_punctuation_halts(self):
        _, agent, speaker, reply = self._loop_with_transcript("Stop talking.")
        self.assertEqual(reply, "")
        agent.assert_not_called()
        speaker.speak.assert_not_called()

    def test_stop_prefix_still_goes_to_agent(self):
        _, agent, speaker, reply = self._loop_with_transcript("stop my timer")
        agent.assert_called_once_with("stop my timer")
        speaker.speak.assert_called_once_with("agent reply")
        self.assertEqual(reply, "agent reply")


if __name__ == "__main__":
    unittest.main()
