"""Shared local text-to-speech via Piper. Used by the voice loop and
Home Mini announcements. Piper is imported lazily so nothing else breaks
when the voice stack isn't installed."""
import os
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent


def default_voice_path() -> Path:
    return Path(os.getenv("VIKI_VOICE", str(HERE.parent / "voice" / "en_US-lessac-medium.onnx")))


def synthesize_wav(text: str, out_path: Path, voice_path: Path | None = None):
    """Render text to a WAV file with Piper. Raises RuntimeError with setup
    instructions when Piper or the voice model is missing."""
    voice_path = voice_path or default_voice_path()
    if not voice_path.exists():
        raise RuntimeError(
            f"Voice model not found at {voice_path}. "
            "Run: python voice_loop.py --setup-voice"
        )
    try:
        from piper import PiperVoice
    except ImportError:
        raise RuntimeError(
            "piper-tts is not installed. Run: pip install -r requirements-voice.txt"
        )
    voice = PiperVoice.load(str(voice_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        voice.synthesize_wav(text, w)
