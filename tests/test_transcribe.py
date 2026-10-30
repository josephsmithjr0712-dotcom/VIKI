"""POST /api/transcribe contract tests (v0.3.27).

The Lenovo Smart Clock posts a 16 kHz mono WAV here (companion README,
"Server contract: POST /api/transcribe") and expects {"text": "..."}.
faster-whisper is stubbed: the real model is never loaded in tests.
Follows the import scaffolding of tests/test_server.py.
"""
import io
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = tempfile.mkdtemp(prefix="viki-transcribe-test-")
_ENV_KEYS = ("VIKI_SESSION_DIR", "VIKI_SECRET_KEY", "VIKI_WEB_PASSWORD",
             "VIKI_AUTO_MEMORY", "VIKI_RETRIES")
_old_env = {k: os.environ.get(k) for k in _ENV_KEYS}
os.environ.update({"VIKI_SESSION_DIR": _TMP,
                   "VIKI_SECRET_KEY": "test-secret-key",
                   "VIKI_WEB_PASSWORD": "test-password",
                   "VIKI_AUTO_MEMORY": "off",
                   "VIKI_RETRIES": "0"})

try:
    import ollama  # noqa: F401
    _ollama_stubbed = False
except ImportError:
    ollama = ModuleType("ollama")

    class ResponseError(Exception):
        pass

    ollama.ResponseError = ResponseError
    ollama.Client = None
    sys.modules["ollama"] = ollama
    _ollama_stubbed = True

_dotenv_had = "dotenv" in sys.modules
_dotenv_old = sys.modules.get("dotenv")
_dotenv_stub = ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **k: False  # noqa: E731
_dotenv_stub.set_key = lambda *a, **k: False  # noqa: E731
sys.modules["dotenv"] = _dotenv_stub

import agent as agent_mod

_real_VIKI = agent_mod.VIKI


class FakeVIKI:
    def __init__(self, toolbox):
        self.toolbox = toolbox
        self.model = "test-model"

    def warmup(self):
        pass


_SERVER_IMPORT_ERROR = None
server_mod = None
try:
    agent_mod.VIKI = FakeVIKI
    import tools as tools_mod  # noqa: E402

    _real_refresh = tools_mod.refresh_ha_entities
    tools_mod.refresh_ha_entities = lambda: None  # noqa: E731
    import server as _srv  # noqa: E402

    server_mod = _srv
except Exception as e:  # flask/ddgs/etc. missing: skip the server tests
    _SERVER_IMPORT_ERROR = e
finally:
    agent_mod.VIKI = _real_VIKI
    try:
        tools_mod.refresh_ha_entities = _real_refresh  # noqa: F821
    except NameError:
        pass
    for k, v in _old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if _ollama_stubbed:
        del sys.modules["ollama"]
    if _dotenv_had:
        sys.modules["dotenv"] = _dotenv_old
    else:
        sys.modules.pop("dotenv", None)

try:
    import faster_whisper  # noqa: F401
    _FW_INSTALLED = True
except ImportError:
    _FW_INSTALLED = False


def _wav_bytes(seconds=0.1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


class _Segment:
    def __init__(self, text):
        self.text = text


class _FakeWhisper:
    """Stands in for faster_whisper.WhisperModel; records constructor args."""
    last_kwargs = None
    segments = []
    fail = None

    def __init__(self, *args, **kwargs):
        _FakeWhisper.last_kwargs = (args, kwargs)

    def transcribe(self, _path, **_kwargs):
        if _FakeWhisper.fail is not None:
            raise _FakeWhisper.fail
        return list(_FakeWhisper.segments), None


class _FakeFasterWhisper:
    """Context manager: inject the fake faster_whisper module for one test."""

    def __enter__(self):
        self._had = "faster_whisper" in sys.modules
        self._old = sys.modules.get("faster_whisper")
        mod = ModuleType("faster_whisper")
        mod.WhisperModel = _FakeWhisper
        sys.modules["faster_whisper"] = mod
        if server_mod is not None:
            server_mod._transcribe_model = None
        return self

    def __exit__(self, *exc):
        if self._had:
            sys.modules["faster_whisper"] = self._old
        else:
            sys.modules.pop("faster_whisper", None)
        if server_mod is not None:
            server_mod._transcribe_model = None
        return False


@unittest.skipUnless(server_mod is not None,
                     f"server.py not importable here: {_SERVER_IMPORT_ERROR}")
class TranscribeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = server_mod.app
        cls.app.testing = True

    def _login(self):
        client = self.app.test_client()
        r = client.post("/login", json={"password": "test-password"})
        self.assertEqual(r.status_code, 200)
        return client

    def test_requires_login(self):
        r = self.app.test_client().post(
            "/api/transcribe", data=_wav_bytes(), content_type="audio/wav")
        self.assertEqual(r.status_code, 401)

    def test_empty_body_rejected(self):
        client = self._login()
        r = client.post("/api/transcribe", data=b"", content_type="audio/wav")
        self.assertEqual(r.status_code, 400)

    def test_invalid_wav_rejected(self):
        client = self._login()
        r = client.post("/api/transcribe", data=b"not-a-wav",
                        content_type="audio/wav")
        self.assertEqual(r.status_code, 400)

    def test_transcript_returned(self):
        client = self._login()
        _FakeWhisper.segments = [_Segment("hello viki")]
        with _FakeFasterWhisper():
            r = client.post("/api/transcribe", data=_wav_bytes(),
                            content_type="audio/wav")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"text": "hello viki"})
        args, kwargs = _FakeWhisper.last_kwargs
        self.assertEqual(args[0], "base.en")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["compute_type"], "int8")

    def test_no_speech_returns_empty_text(self):
        client = self._login()
        _FakeWhisper.segments = []
        with _FakeFasterWhisper():
            r = client.post("/api/transcribe", data=_wav_bytes(),
                            content_type="audio/wav")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"text": ""})

    def test_transcribe_failure_is_502_with_cause(self):
        # The clock only shows a generic failure, so the 502 body must
        # carry the cause (this is how the PyAV 19 incompatibility
        # surfaced: av.open() rejecting faster-whisper's metadata_errors).
        client = self._login()
        _FakeWhisper.fail = TypeError(
            "av.open() got an unexpected keyword argument 'metadata_errors'")
        try:
            with _FakeFasterWhisper():
                r = client.post("/api/transcribe", data=_wav_bytes(),
                                content_type="audio/wav")
        finally:
            _FakeWhisper.fail = None
        self.assertEqual(r.status_code, 502)
        self.assertIn("metadata_errors", r.get_json()["error"])

    def test_voice_requirements_pin_av_below_19(self):
        # Regression guard for the 2026-10-29 clock outage: faster-whisper
        # 1.2.1 declares av>=11 with no upper bound, but PyAV 19 removed
        # the metadata_errors argument it passes to av.open().
        reqs = (Path(__file__).resolve().parent.parent
                / "requirements-voice.txt").read_text()
        av_lines = [ln for ln in reqs.splitlines()
                    if ln.startswith("av>") or ln.startswith("av==")]
        self.assertEqual(len(av_lines), 1)
        self.assertIn("<19", av_lines[0])

    @unittest.skipIf(_FW_INSTALLED,
                     "faster-whisper is installed here, so no 501 path")
    def test_missing_faster_whisper_is_501(self):
        client = self._login()
        server_mod._transcribe_model = None
        sys.modules.pop("faster_whisper", None)
        try:
            r = client.post("/api/transcribe", data=_wav_bytes(),
                            content_type="audio/wav")
        finally:
            server_mod._transcribe_model = None
        self.assertEqual(r.status_code, 501)


if __name__ == "__main__":
    unittest.main()
