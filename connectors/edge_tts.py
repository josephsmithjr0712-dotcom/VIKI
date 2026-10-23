"""Edge TTS (Microsoft neural voices) for VIKI.

Free, cloud-based, much more natural than local Piper or browser voices.
Default: en-US-AriaNeural (natural US female, conversational).

Override with VIKI_TTS_VOICE env var, e.g.:
  en-US-JennyNeural  - warm US female
  en-US-EmmaNeural   - young US female
  en-US-AvaNeural    - US female
  en-US-GuyNeural    - US male

Usage:
    from connectors.edge_tts import speak
    mp3_bytes = speak("Hello boss")  # returns MP3 bytes

Requires: pip install edge-tts
"""
import asyncio
import io
import os

DEFAULT_VOICE = "en-US-AriaNeural"


def get_voice():
    return os.getenv("VIKI_TTS_VOICE", DEFAULT_VOICE)


def speak(text, voice=None):
    """Synthesize text to MP3 bytes using Edge TTS. Returns bytes."""
    import edge_tts
    voice = voice or get_voice()
    text = (text or "").strip()
    if not text:
        return b""
    # Strip markdown-ish formatting for cleaner speech
    import re
    text = re.sub(r"[`*#_~]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # [text](url) -> text
    text = text[:5000]  # Edge TTS limit safety

    async def _run():
        buf = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(_run())


def list_voices(filter_lang="en-US"):
    """List available voices, optionally filtered by language. Returns list of dicts."""
    import edge_tts

    async def _run():
        voices = await edge_tts.list_voices()
        if filter_lang:
            voices = [v for v in voices if v.get("Locale", "").startswith(filter_lang)]
        return [{"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]} for v in voices]

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(_run())
