"""VIKI hands-free voice loop: wake word -> listen -> think -> speak. All local.

Pipeline: openwakeword (wake word) -> faster-whisper (speech to text)
        -> VIKI agent loop -> Piper (text to speech) -> speakers.

Install the audio stack once (see requirements-voice.txt):
    pip install -r requirements-voice.txt
    python -m playwright install chromium   # not needed here; browser is separate

Then:
    python voice_loop.py --check        # report what's installed / missing
    python voice_loop.py --setup-voice   # download the default Piper voice
    python voice_loop.py                # start listening for the wake word

Environment knobs:
    VIKI_WAKEWORD   wake word model (default hey_jarvis; train "hey viki"
                    with openwakeword's Colab notebook for a custom one, then set
                    VIKI_WAKEWORD=hey_viki and VIKI_WAKEWORD_MODEL=/path/to/model.onnx)
    VIKI_WAKEWORD_MODEL  path to a custom openwakeword .onnx model (optional)
    LAMP_ENTITY     Home Assistant light used as a status indicator in voice
                    mode: purple on wake, blue while thinking, pink while speaking
                    (e.g. light.desk_lamp). Unset = indicator disabled.
    VIKI_WHISPER    faster-whisper model (default tiny.en; base.en is better, slower)
    VIKI_VOICE      path to Piper .onnx voice (default voice/en_US-lessac-medium.onnx)
    VIKI_MIC        sounddevice input device index (default: system default)
    VIKI_FOLLOWUP_SECONDS  seconds after VIKI speaks during which a quick
                    reply is heard WITHOUT repeating the wake word
                    (default 10; 0 disables the follow-up window)

Safety: in voice mode, risky tools (sending mail, creating events, clicking
in the browser...) are auto-denied — VIKI says so out loud instead of acting.
Use the web UI when you want to approve something.
"""
import os
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms frames, what openwakeword expects


@dataclass
class VoiceConfig:
    wakeword: str = field(default_factory=lambda: os.getenv("VIKI_WAKEWORD", "hey_jarvis"))
    whisper_model: str = field(default_factory=lambda: os.getenv("VIKI_WHISPER", "tiny.en"))
    voice_path: Path = field(default_factory=lambda: Path(
        os.getenv("VIKI_VOICE", str(HERE / "voice" / "en_US-lessac-medium.onnx"))))
    mic_device: int | None = field(default_factory=lambda: (
        int(os.environ["VIKI_MIC"]) if os.getenv("VIKI_MIC") else None))
    silence_threshold: float = 400.0   # mean abs amplitude below this = silence
    silence_seconds: float = 1.2       # stop recording after this much silence
    max_seconds: float = 20.0          # hard cap per utterance
    wake_threshold: float = 0.5        # openwakeword score to trigger
    followup_seconds: float = field(default_factory=lambda: float(
        os.getenv("VIKI_FOLLOWUP_SECONDS", "10")))  # quick-reply window after
    # VIKI speaks: speech here skips the wake word. 0 = disabled.


# Words that stop VIKI mid-flow instead of running the agent (v0.3.16).
# Exact-phrase match on the normalized transcript: "stop" halts her, but
# "stop my timer" still goes to the agent — it may be a real command.
STOP_WORDS = {"stop", "cancel", "quiet", "shut up", "never mind", "stop talking"}


def is_stop_command(text: str) -> bool:
    """True when the transcript is a bare stop phrase (case/punctuation free)."""
    return text.strip().lower().strip(" .!?") in STOP_WORDS


# --- audio primitives (numpy-free where easy; sounddevice only at the edges) --
def _mean_abs(frame: bytes) -> float:
    import array
    a = array.array("h", frame)
    return sum(abs(x) for x in a) / max(1, len(a))


def record_until_silence(frames, cfg: VoiceConfig) -> bytes:
    """Consume an iterable of int16 mono 16kHz frames; stop at end of speech.
    Returns the raw audio bytes. Pure logic — testable without a microphone."""
    import math
    silence_needed = math.ceil(cfg.silence_seconds * SAMPLE_RATE / FRAME_SAMPLES)
    max_frames = math.ceil(cfg.max_seconds * SAMPLE_RATE / FRAME_SAMPLES)
    out, silent, heard = [], 0, False
    for f in frames:
        out.append(f)
        if _mean_abs(f) < cfg.silence_threshold:
            silent += 1
        else:
            silent, heard = 0, True
        if heard and silent >= silence_needed:
            break
        if len(out) >= max_frames:
            break
    return b"".join(out)


def listen_for_followup(frames, cfg: VoiceConfig) -> bytes | None:
    """Wait up to cfg.followup_seconds for a quick reply, then record it.

    Pure logic — testable without a microphone. Consumes frames until the
    first non-silent frame, then hands the rest to record_until_silence
    (so the reply is captured the same way as a wake-word utterance).
    Returns the recorded audio bytes, or None if only silence arrived (or
    the stream ended) within the window. A followup_seconds of 0 disables
    the wait entirely and consumes nothing.
    """
    import itertools
    import math
    if cfg.followup_seconds <= 0:
        return None
    window = math.ceil(cfg.followup_seconds * SAMPLE_RATE / FRAME_SAMPLES)
    it = iter(frames)
    for _ in range(window):
        try:
            f = next(it)
        except StopIteration:
            return None
        if _mean_abs(f) >= cfg.silence_threshold:
            return record_until_silence(itertools.chain([f], it), cfg)
    return None


def write_wav(path: Path, audio: bytes, sample_rate: int = SAMPLE_RATE):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio)


class Microphone:
    """16kHz mono mic stream yielding 80ms frames. Lazy sounddevice import."""

    def __init__(self, cfg: VoiceConfig):
        self.cfg = cfg

    def frames(self):
        import sounddevice as sd
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            device=self.cfg.mic_device,
                            blocksize=FRAME_SAMPLES) as stream:
            while True:
                data, _ = stream.read(FRAME_SAMPLES)
                yield data.tobytes()


# --- pipeline stages ---------------------------------------------------------
class WakeWordDetector:
    def __init__(self, cfg: VoiceConfig):
        from openwakeword.model import Model
        self.cfg = cfg
        custom = os.getenv("VIKI_WAKEWORD_MODEL")
        models = [custom] if custom else [cfg.wakeword]
        self.model = Model(wakeword_models=models, inference_framework="onnx")

    def heard(self, frame: bytes) -> bool:
        import numpy as np
        audio = np.frombuffer(frame, dtype=np.int16)
        scores = self.model.predict(audio)
        return scores.get(self.cfg.wakeword, 0.0) >= self.cfg.wake_threshold


class Transcriber:
    def __init__(self, cfg: VoiceConfig):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")

    def transcribe(self, wav_path: Path) -> str:
        segments, _ = self.model.transcribe(str(wav_path), beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segments).strip()


class Speaker:
    """Speaks via the Google Home Mini through Home Assistant (Edge TTS Aria voice).
    Falls back to local speakers (Edge TTS -> Piper) if HA is unreachable.
    Set VIKI_SPEAKER=local to force local speakers."""
    def __init__(self, cfg: VoiceConfig):
        self.cfg = cfg
        self._mode = os.getenv("VIKI_SPEAKER", "mini").lower()  # mini | local

    def speak(self, text: str):
        if not text.strip():
            return
        if self._mode == "mini":
            try:
                from connectors.home_mini import announce
                result = announce(text)
                if "Announced" in result:
                    return
                print(f"mini announce issue: {result}, falling back to local")
            except Exception as e:
                print(f"mini announce failed ({e}), falling back to local")
        self._speak_local(text)

    def _speak_local(self, text: str):
        """Play through the server's own speakers."""
        audio_bytes = None
        suffix = ".mp3"
        try:
            from connectors.edge_tts import speak as edge_speak
            audio_bytes = edge_speak(text)
        except Exception:
            pass
        if not audio_bytes:
            # Piper fallback
            if not self.cfg.voice_path.exists():
                raise RuntimeError(
                    f"Edge TTS unavailable and no Piper voice at {self.cfg.voice_path}. "
                    "pip install edge-tts, or run: python voice_loop.py --setup-voice"
                )
            from connectors.tts import synthesize_wav
            suffix = ".wav"
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                out = Path(f.name)
            synthesize_wav(text, out, self.cfg.voice_path)
            audio_bytes = out.read_bytes()
            out.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            out = Path(f.name)
            out.write_bytes(audio_bytes)
        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(str(out))
            sd.play(data, sr)
            sd.wait()
        except ImportError:
            import subprocess  # fallback: mpv/aplay on Linux
            player = "mpv" if suffix == ".mp3" else "aplay"
            subprocess.run([player, str(out)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            out.unlink(missing_ok=True)


def download_default_voice(cfg: VoiceConfig):
    """Fetch the default Piper voice (~60MB) into ./voice/."""
    import urllib.request
    base = "https://github.com/rhasspy/piper/releases/download/v1.2.0"
    name = "en_US-lessac-medium"
    cfg.voice_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".onnx", ".onnx.json"):
        url = f"{base}/{name}{ext}"
        dest = cfg.voice_path.parent / f"{name}{ext}"
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    print("done.")


def check_environment() -> dict:
    """Report which voice-loop dependencies are installed."""
    results = {}
    for label, stmt in [
        ("sounddevice (mic + speakers)", "import sounddevice"),
        ("openwakeword (wake word)", "import openwakeword"),
        ("faster-whisper (speech to text)", "import faster_whisper"),
        ("piper-tts (text to speech)", "import piper"),
        ("soundfile (wav playback)", "import soundfile"),
    ]:
        try:
            exec(stmt)
            results[label] = "ok"
        except ImportError:
            results[label] = "MISSING"
    cfg = VoiceConfig()
    results[f"piper voice file ({cfg.voice_path.name})"] = (
        "ok" if cfg.voice_path.exists() else "MISSING — run --setup-voice")
    try:
        from connectors import lamp as lamp_mod
        ent = lamp_mod.resolve_lamp_entity()
        results["lamp indicator"] = f"ok ({ent})" if ent else "not configured (set LAMP_ENTITY)"
    except Exception as e:
        results["lamp indicator"] = f"unavailable: {e}"
    return results


# --- orchestration ------------------------------------------------------------
def voice_approve(call_id, name, args) -> bool:
    """Voice mode never auto-approves risky tools — the user can't see a card."""
    return False


class VoiceLoop:
    """Wires the audio pipeline to the VIKI agent. Stages are injectable for tests."""

    def __init__(self, cfg=None, mic=None, wake=None, transcriber=None, speaker=None,
                 agent_runner=None, lamp="auto"):
        self.cfg = cfg or VoiceConfig()
        self.mic = mic or Microphone(self.cfg)
        self.wake = wake
        self.transcriber = transcriber
        self.speaker = speaker
        self.agent_runner = agent_runner or self._default_agent_runner
        # lamp: "auto" = use the HA lamp indicator if configured (voice mode
        # only — the web UI agent never touches it); False/None = disabled.
        self._lamp_setting = lamp
        self.lamp = None
        self._lamp_resolved = False
        self._history = []

    def _lamp_stage(self):
        """Resolve the lamp indicator once. Returns it or None. Never raises."""
        if not self._lamp_resolved:
            self._lamp_resolved = True
            setting = self._lamp_setting
            if setting == "auto":
                try:
                    from connectors import lamp as lamp_mod
                    self.lamp = lamp_mod.LampIndicator() if lamp_mod.available() else None
                except Exception:
                    self.lamp = None
            elif setting:
                self.lamp = setting
        return self.lamp

    def _lamp_set(self, state: str):
        lamp = self._lamp_stage()
        if lamp is not None:
            try:
                lamp.set(state)
            except Exception:
                pass

    def _ensure_stages(self, with_wake=True):
        if with_wake and self.wake is None:
            self.wake = WakeWordDetector(self.cfg)
        if self.transcriber is None:
            self.transcriber = Transcriber(self.cfg)
        if self.speaker is None:
            self.speaker = Speaker(self.cfg)

    def _default_agent_runner(self, text: str) -> str:
        """Run one headless agent turn. Risky tools are auto-denied in voice mode."""
        from agent import VIKI
        from memory import SessionStore
        from tools import Toolbox

        store = SessionStore()
        tb = Toolbox(store)
        viki = VIKI(model=os.getenv("VIKI_MODEL", "llama3.2:latest"), tools=tb)
        events = []

        out = viki.run(text, self._history, store.search,
                       emit=lambda k, p: events.append((k, p)),
                       approve=voice_approve, think=False)
        reply = out.get("text") or ""
        if out.get("type") == "approval_denied":
            reply = ("That needs your approval, boss — "
                     "tap Allow in the web UI and ask me again.")
        return reply or "Hmm, I came up empty on that one."

    def handle_utterance(self, wav_path: Path) -> str:
        """Transcribe one utterance, run the agent, speak the reply. Returns reply."""
        self._ensure_stages(with_wake=False)
        text = self.transcriber.transcribe(wav_path).strip()
        if not text:
            self._lamp_set("restore")
            return ""
        if is_stop_command(text):
            # Bare "stop" / "cancel" halts her without running the agent.
            self._lamp_set("restore")
            return ""
        print(f"heard: {text}")
        self._lamp_set("thinking")  # blue while the agent works
        try:
            reply = self.agent_runner(text)
            self._history += [{"role": "user", "content": text},
                              {"role": "assistant", "content": reply}]
            print(f"reply: {reply}")
            self._lamp_set("speaking")  # pink while talking back
            self.speaker.speak(reply)
            return reply
        finally:
            self._lamp_set("restore")  # back to however the lamp was

    def _handle_audio(self, audio: bytes):
        """Write one recorded utterance to a temp wav and handle it."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)
        try:
            write_wav(wav_path, audio)
            self.handle_utterance(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

    def run(self):
        self._ensure_stages()
        print(f"Listening for '{self.cfg.wakeword}' — Ctrl+C to stop.")
        frames = self.mic.frames()
        try:
            while True:
                frame = next(frames)
                if self.wake.heard(frame):
                    print("wake word!")
                    self._lamp_set("wake")  # purple
                    self._lamp_set("speaking")
                    self.speaker.speak("Yes, boss?")
                    self._handle_audio(record_until_silence(frames, self.cfg))
                    # Follow-up window: quick replies right after VIKI
                    # speaks skip the wake word. Silence through the whole
                    # window drops back to wake-word listening. No
                    # "Yes, boss?" here — the conversation is already open.
                    while self.cfg.followup_seconds > 0:
                        followup = listen_for_followup(frames, self.cfg)
                        if followup is None:
                            break
                        self._handle_audio(followup)
        except KeyboardInterrupt:
            print("\nStopped.")


def main(argv):
    if "--check" in argv:
        for label, status in check_environment().items():
            print(f"[{'OK' if status == 'ok' else '!!'}] {label}" + ("" if status == "ok" else f": {status}"))
        return
    if "--setup-voice" in argv:
        download_default_voice(VoiceConfig())
        return
    VoiceLoop().run()


if __name__ == "__main__":
    main(sys.argv[1:])
