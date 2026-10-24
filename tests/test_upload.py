"""Tests for secure attachment staging (uploads.py) and the agent's image path."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
import agent as agent_mod
from memory import SessionStore
from tools import Toolbox
from uploads import sniff, stage_files, discard, video_frames, read_text, UploadError, MAX_BYTES

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 64
SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
HTML = b"<!DOCTYPE html><html><body>hi</body></html>"
XML = b'<?xml version="1.0"?><a/>'
TEXT = b"hello world\nthis is a plain text file\n"
BINARY = bytes(range(256))


class TestSniff(unittest.TestCase):
    def test_images(self):
        self.assertEqual(sniff(PNG), ("image", "image/png"))
        self.assertEqual(sniff(JPG), ("image", "image/jpeg"))
        self.assertEqual(sniff(GIF), ("image", "image/gif"))
        self.assertEqual(sniff(WEBP), ("image", "image/webp"))

    def test_video(self):
        self.assertEqual(sniff(MP4), ("video", "video/mp4"))
        self.assertEqual(sniff(WEBM), ("video", "video/webm"))

    def test_text(self):
        self.assertEqual(sniff(TEXT), ("text", "text/plain"))

    def test_markup_rejected(self):
        # SVG is the dangerous one: scriptable, often served inline.
        self.assertEqual(sniff(SVG), (None, None))
        self.assertEqual(sniff(HTML), (None, None))
        self.assertEqual(sniff(XML), (None, None))

    def test_binary_rejected(self):
        self.assertEqual(sniff(BINARY), (None, None))
        self.assertEqual(sniff(b""), (None, None))
        # Valid UTF-8 but with null bytes = binary in disguise.
        self.assertEqual(sniff(b"abc\x00def"), (None, None))

    def test_spoofed_extension_ignored(self):
        # An executable renamed to .png is still rejected: magic bytes rule.
        self.assertEqual(sniff(b"MZ" + b"\x00" * 100), (None, None))
        # Claimed image that is actually HTML.
        self.assertEqual(sniff(b"<html>" + b"\x00" * 100), (None, None))


class TestStageFiles(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())

    def test_happy_path(self):
        staged = stage_files([("photo.png", PNG), ("notes.txt", TEXT)], base=self.base)
        self.assertEqual(len(staged), 2)
        self.assertEqual(staged[0]["kind"], "image")
        self.assertEqual(staged[1]["kind"], "text")
        for s in staged:
            p = Path(s["path"])
            self.assertTrue(p.exists())
            # UUID file names: the user-supplied name never touches disk.
            self.assertNotIn("photo", p.name)
            self.assertNotIn("notes", p.name)
        discard(staged)
        self.assertFalse(any(self.base.glob("viki-up-*")))

    def test_traversal_filename_neutralized(self):
        staged = stage_files([("../../etc/passwd", TEXT)], base=self.base)
        self.assertEqual(staged[0]["name"], "passwd")
        self.assertTrue(Path(staged[0]["path"]).parent.parent == self.base)
        discard(staged)

    def test_oversize_rejected(self):
        with self.assertRaises(UploadError):
            stage_files([("big.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_BYTES + 1))], base=self.base)
        self.assertFalse(list(self.base.glob("viki-up-*")))

    def test_empty_rejected(self):
        with self.assertRaises(UploadError):
            stage_files([("empty.png", b"")], base=self.base)

    def test_unsafe_type_rejected(self):
        with self.assertRaises(UploadError):
            stage_files([("evil.svg", SVG)], base=self.base)
        self.assertFalse(list(self.base.glob("viki-up-*")))

    def test_max_files_cap(self):
        staged = stage_files([("a.txt", TEXT)] * 10, base=self.base)
        self.assertLessEqual(len(staged), 3)
        discard(staged)

    def test_nothing_left_on_failure(self):
        # One good file + one bad file: the whole batch fails atomically.
        with self.assertRaises(UploadError):
            stage_files([("ok.png", PNG), ("evil.svg", SVG)], base=self.base)
        self.assertFalse(list(self.base.glob("viki-up-*")))


class TestVideoFrames(unittest.TestCase):
    def test_never_raises(self):
        # Returns a list (possibly empty when ffmpeg is missing); never raises.
        self.assertIsInstance(video_frames("/nonexistent/file.mp4"), list)


class TestReadText(unittest.TestCase):
    def test_cap(self):
        p = Path(tempfile.mkdtemp()) / "big.txt"
        p.write_text("x" * 200_000)
        self.assertLessEqual(len(read_text(str(p))), 100_000)


# --- agent image path ------------------------------------------------------

class FakeMsg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChunk:
    def __init__(self, message):
        self.message = message


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.kwargs = []

    def chat(self, **kw):
        self.kwargs.append(kw)
        kind, payload = self.script.pop(0)
        if kind == "raise":
            raise payload
        return [FakeChunk(m) for m in payload]


def make_viki(script, tmp):
    fake = FakeClient(script)
    store = SessionStore(folder=tmp / "session")
    with patch.object(agent_mod.ollama, "Client", return_value=fake):
        v = agent_mod.VIKI(Toolbox(store))
    return v, fake


class TestAgentImages(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def run_turn(self, viki, **kw):
        events = []
        return viki.run("look at this", [], [],
                        lambda k, p: events.append((k, p)),
                        lambda cid, n, a: True, think=False,
                        cancel=threading.Event(), **kw), events

    def test_images_attached_to_user_message(self):
        viki, fake = make_viki([("chunks", [FakeMsg("nice pic")])], self.tmp)
        out, _ = self.run_turn(viki, images=[b"imgbytes"])
        user_msgs = [m for m in fake.kwargs[0]["messages"] if m["role"] == "user"]
        self.assertEqual(user_msgs[-1]["images"], [b"imgbytes"])
        self.assertEqual(out["content"], "nice pic")

    def test_no_images_no_field(self):
        viki, fake = make_viki([("chunks", [FakeMsg("hi")])], self.tmp)
        self.run_turn(viki)
        user_msgs = [m for m in fake.kwargs[0]["messages"] if m["role"] == "user"]
        self.assertNotIn("images", user_msgs[-1])

    def test_vision_unsupported_degrades_gracefully(self):
        viki, fake = make_viki([
            ("raise", ollama.ResponseError("model 'gemma4:2b' does not support images")),
            ("chunks", [FakeMsg("working from text")]),
        ], self.tmp)
        out, events = self.run_turn(viki, images=[b"imgbytes"])
        # Second attempt went out without images...
        user_msgs = [m for m in fake.kwargs[1]["messages"] if m["role"] == "user"]
        self.assertNotIn("images", user_msgs[-1])
        # ...the user was told plainly...
        statuses = [p for k, p in events if k == "status"]
        self.assertTrue(any("can't process images" in s for s in statuses))
        # ...and the turn still completed.
        self.assertEqual(out["content"], "working from text")


if __name__ == "__main__":
    unittest.main()
