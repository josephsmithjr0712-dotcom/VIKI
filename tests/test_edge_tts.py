"""Tests for connectors/edge_tts.py (Edge TTS via Microsoft neural voices).

The real `edge_tts` package needs network + cloud access, so these tests
inject a fake `edge_tts` module into sys.modules. The fake records what
text/voice it was given and replays scripted stream chunks.
"""
import asyncio
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

fake_edge_tts = types.ModuleType("edge_tts")


class FakeCommunicate:
    """Stand-in for edge_tts.Communicate. Test controls `chunks` / `error`."""
    calls = []       # (text, voice) per instantiation
    chunks = [{"type": "audio", "data": b"CHUNK1"},
              {"type": "words", "data": "ignored"},
              {"type": "audio", "data": b"CHUNK2"}]
    error = None

    def __init__(self, text, voice):
        FakeCommunicate.calls.append((text, voice))

    async def stream(self):
        if FakeCommunicate.error:
            raise FakeCommunicate.error
        for c in FakeCommunicate.chunks:
            yield c


async def _fake_list_voices():
    return [
        {"ShortName": "en-US-AriaNeural", "Gender": "Female", "Locale": "en-US"},
        {"ShortName": "en-US-GuyNeural", "Gender": "Male", "Locale": "en-US"},
        {"ShortName": "de-DE-KatjaNeural", "Gender": "Female", "Locale": "de-DE"},
    ]


fake_edge_tts.Communicate = FakeCommunicate
fake_edge_tts.list_voices = _fake_list_voices


class EdgeTtsTest(unittest.TestCase):
    def setUp(self):
        sys.modules["edge_tts"] = fake_edge_tts
        FakeCommunicate.calls = []
        FakeCommunicate.error = None
        FakeCommunicate.chunks = [{"type": "audio", "data": b"CHUNK1"},
                                  {"type": "words", "data": "ignored"},
                                  {"type": "audio", "data": b"CHUNK2"}]
        os.environ.pop("VIKI_TTS_VOICE", None)
        import connectors.edge_tts as m
        # reload so get_voice() picks up env changes cleanly
        import importlib
        self.mod = importlib.reload(m)

    def tearDown(self):
        sys.modules.pop("edge_tts", None)
        os.environ.pop("VIKI_TTS_VOICE", None)

    def test_get_voice_default(self):
        self.assertEqual(self.mod.get_voice(), "en-US-AriaNeural")

    def test_get_voice_env_override(self):
        os.environ["VIKI_TTS_VOICE"] = "en-US-GuyNeural"
        self.assertEqual(self.mod.get_voice(), "en-US-GuyNeural")

    def test_speak_returns_audio_bytes(self):
        out = self.mod.speak("hello boss")
        self.assertEqual(out, b"CHUNK1CHUNK2")

    def test_speak_uses_default_voice(self):
        self.mod.speak("hi")
        self.assertEqual(FakeCommunicate.calls[0][1], "en-US-AriaNeural")

    def test_speak_voice_arg_overrides(self):
        self.mod.speak("hi", voice="en-US-EmmaNeural")
        self.assertEqual(FakeCommunicate.calls[0][1], "en-US-EmmaNeural")

    def test_speak_empty_text_returns_empty_no_network(self):
        for t in ("", "   ", None):
            self.assertEqual(self.mod.speak(t), b"")
        self.assertEqual(FakeCommunicate.calls, [])

    def test_speak_strips_markdown(self):
        self.mod.speak("**bold** and `code` and #hash and ~strike~")
        text = FakeCommunicate.calls[0][0]
        self.assertNotIn("*", text)
        self.assertNotIn("`", text)
        self.assertNotIn("#", text)
        self.assertNotIn("~", text)
        self.assertIn("bold", text)

    def test_speak_resolves_markdown_links(self):
        self.mod.speak("see [the docs](https://example.com/x) now")
        self.assertEqual(FakeCommunicate.calls[0][0], "see the docs now")

    def test_speak_truncates_long_text(self):
        self.mod.speak("x" * 6000)
        self.assertEqual(len(FakeCommunicate.calls[0][0]), 5000)

    def test_speak_from_running_event_loop(self):
        """Regression: used to raise 'This event loop is already running'."""
        async def main():
            return self.mod.speak("nested call")
        out = asyncio.run(main())
        self.assertEqual(out, b"CHUNK1CHUNK2")

    def test_speak_propagates_stream_error(self):
        FakeCommunicate.error = ConnectionError("cloud down")
        with self.assertRaises(ConnectionError):
            self.mod.speak("boom")

    def test_speak_still_works_after_error(self):
        """A failed call must not poison the loop for later calls."""
        FakeCommunicate.error = ConnectionError("cloud down")
        with self.assertRaises(ConnectionError):
            self.mod.speak("boom")
        FakeCommunicate.error = None  # cloud recovers
        self.assertEqual(self.mod.speak("again"), b"CHUNK1CHUNK2")

    def test_list_voices_filters_locale(self):
        voices = self.mod.list_voices()
        names = [v["name"] for v in voices]
        self.assertEqual(names, ["en-US-AriaNeural", "en-US-GuyNeural"])
        self.assertEqual(voices[0]["gender"], "Female")
        self.assertEqual(voices[0]["locale"], "en-US")

    def test_list_voices_no_filter(self):
        voices = self.mod.list_voices(filter_lang=None)
        self.assertEqual(len(voices), 3)


if __name__ == "__main__":
    unittest.main(verbosity=1)
