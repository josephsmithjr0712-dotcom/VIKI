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
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms frames, what openwakeword expects
# VoiceLoop._history is capped to this many recent messages (finding 39b):
# a long session must not grow the history list without bound.
HISTORY_MAX_MESSAGES = 20


def _env_int(name: str):
    """Parse an integer env knob, or None when unset.

    Raises a clear ValueError naming the variable instead of the raw
    ``int()`` error, so a bad VIKI_MIC is diagnosed at construction.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"{name} must be an integer, got {raw!r} "
            "(unset it for the system default)") from None


def _env_float(name: str, default: float) -> float:
    """Parse a float env knob, falling back to ``default`` when unset."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"{name} must be a number, got {raw!r} "
            f"(default {default})") from None


@dataclass
class VoiceConfig:
    wakeword: str = field(default_factory=lambda: os.getenv("VIKI_WAKEWORD", "hey_jarvis"))
    whisper_model: str = field(default_factory=lambda: os.getenv("VIKI_WHISPER", "tiny.en"))
    voice_path: Path = field(default_factory=lambda: Path(
        os.getenv("VIKI_VOICE", str(HERE / "voice" / "en_US-lessac-medium.onnx"))))
    mic_device: int | None = field(default_factory=lambda: _env_int("VIKI_MIC"))
    silence_threshold: float = 400.0   # mean abs amplitude below this = silence
    silence_seconds: float = 1.2       # stop recording after this much silence
    # Initial-silence timeout for the wake -> utterance path (finding 39a):
    # if no speech follows a wake trigger within this long, stop recording
    # and return to wake-word listening instead of holding out to max_seconds.
    initial_silence_seconds: float = 5.0
    max_seconds: float = 20.0          # hard cap per utterance
    wake_threshold: float = 0.5        # openwakeword score to trigger
    followup_seconds: float = field(default_factory=lambda: _env_float(
        "VIKI_FOLLOWUP_SECONDS", 10.0))  # quick-reply window after
    # VIKI speaks: speech here skips the wake word. 0 = disabled.

    def __post_init__(self):
        """Validate env-derived and direct knobs at construction (finding 52).

        Unparseable env values already raise a clear ValueError in the
        _env_* helpers above; this checks the parsed values (and any
        directly passed ones) so a bad knob fails here with a message
        naming the knob, instead of crashing the loop mid-run. Ranges are
        deliberately generous (e.g. silence_seconds may exceed 60 — tests
        and long pauses rely on that); only values that cannot work are
        rejected. Nothing here silently clamps: the file documents no
        fallback for these knobs other than the env defaults above.
        """
        import math

        def _number(field_name, value):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"VoiceConfig.{field_name} must be a number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"VoiceConfig.{field_name} must be finite, got {value!r}")
            return float(value)

        if self.mic_device is not None and (
                isinstance(self.mic_device, bool)
                or not isinstance(self.mic_device, int)):
            raise ValueError(
                f"VIKI_MIC must be an integer device index, "
                f"got {self.mic_device!r} (unset it for the system default)")
        if not isinstance(self.wakeword, str) or not self.wakeword.strip():
            raise ValueError(
                f"VIKI_WAKEWORD must be a non-empty wake word, got {self.wakeword!r}")
        silence_threshold = _number("silence_threshold", self.silence_threshold)
        if not 0 <= silence_threshold <= 32767:
            raise ValueError(
                "VoiceConfig.silence_threshold must be in 0..32767 "
                f"(int16 mean-abs range), got {self.silence_threshold!r}")
        for field_name in ("silence_seconds", "initial_silence_seconds",
                           "max_seconds"):
            if _number(field_name, getattr(self, field_name)) <= 0:
                raise ValueError(
                    f"VoiceConfig.{field_name} must be > 0, "
                    f"got {getattr(self, field_name)!r}")
        wake_threshold = _number("wake_threshold", self.wake_threshold)
        if not 0 <= wake_threshold <= 1:
            raise ValueError(
                f"VoiceConfig.wake_threshold must be in 0..1, "
                f"got {self.wake_threshold!r}")
        if _number("followup_seconds", self.followup_seconds) < 0:
            raise ValueError(
                "VIKI_FOLLOWUP_SECONDS must be a non-negative number, "
                f"got {self.followup_seconds!r} (0 disables the follow-up window)")


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
    Returns the raw audio bytes. Pure logic — testable without a microphone.

    Two bounds: after speech starts, stop after cfg.silence_seconds of
    silence; before any speech, stop after cfg.initial_silence_seconds of
    silence (finding 39a), so a wake trigger with no following speech
    returns to listening instead of holding out to max_seconds. The
    max_seconds hard cap still applies to the whole recording.
    """
    import math
    silence_needed = math.ceil(cfg.silence_seconds * SAMPLE_RATE / FRAME_SAMPLES)
    initial_needed = math.ceil(
        cfg.initial_silence_seconds * SAMPLE_RATE / FRAME_SAMPLES)
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
        if not heard and silent >= initial_needed:
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
def _normalize_wakeword_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _known_wakeword_names(model) -> list:
    """Wake-word names a loaded openwakeword Model knows about, in order.

    The real Model does not expose one canonical attribute across versions,
    so probe the plausible ones; a path-like entry (a custom .onnx file)
    contributes its filename stem, which is the key openwakeword's
    predict() returns for a custom model. Returns [] when nothing is
    exposed — the caller then falls back to the config / file stem.
    """
    names = []
    for attr in ("wakeword_model_names", "model_names", "wakeword_names"):
        value = getattr(model, attr, None)
        if value:
            names.extend(value if isinstance(value, dict) else list(value))
    if names:
        # An explicit name list is authoritative; do NOT also merge in
        # the raw wakeword_models list (which for a custom model is the
        # file path, and would otherwise turn a single model into two
        # "known" names and defeat the single-custom-model rule below).
        pass
    else:
        for attr in ("models", "wakeword_models"):
            value = getattr(model, attr, None)
            if isinstance(value, dict):
                names.extend(value.keys())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        names.append(item)
                    else:
                        for name_attr in ("name", "model_name"):
                            found = getattr(item, name_attr, None)
                            if found:
                                names.append(found)
                                break
    seen, out = set(), []
    for name in names:
        name = str(name)
        if "/" in name or "\\" in name or name.endswith((".onnx", ".tflite")):
            name = Path(name).stem
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


class WakeWordDetector:
    """Wake-word detection with a score key derived from the loaded model.

    Finding 37: the old code always looked scores up by cfg.wakeword, so
    a custom VIKI_WAKEWORD_MODEL whose predict() key is its filename stem
    (e.g. a "hey_viki" model loaded while cfg.wakeword is still the
    default) could never trigger — scores.get(cfg.wakeword, 0.0) silently
    returned 0. The score key is now resolved at construction:
      * if the loaded model exposes known names and one matches the
        configured wake word, that name is the key (default hey_jarvis
        path unchanged);
      * else, for a single custom model, its one known name — or, when
        the model exposes no names, the custom file's stem — is the key;
      * else (a configured wake word with no loaded model at all), fail
        fast with a clear ValueError instead of never triggering.
    ``model`` / ``model_cls`` are injection points for tests (no
    openwakeword install needed); production passes neither.
    """

    def __init__(self, cfg: VoiceConfig, model=None, model_cls=None):
        self.cfg = cfg
        custom = os.getenv("VIKI_WAKEWORD_MODEL")
        models = [custom] if custom else [cfg.wakeword]
        if model is not None:
            self.model = model
        elif model_cls is not None:
            self.model = model_cls(wakeword_models=models,
                                   inference_framework="onnx")
        else:
            from openwakeword.model import Model
            self.model = Model(wakeword_models=models, inference_framework="onnx")
        self.score_key = self._resolve_score_key(cfg, custom)

    def _resolve_score_key(self, cfg: VoiceConfig, custom):
        names = _known_wakeword_names(self.model)
        if names:
            wanted = _normalize_wakeword_name(cfg.wakeword)
            for name in names:
                if _normalize_wakeword_name(name) == wanted:
                    return name
            if custom and len(names) == 1:
                # One custom model loaded: it is the wake word, whatever
                # its key is called — use it rather than cfg.wakeword.
                return names[0]
            raise ValueError(
                f"Wake word {cfg.wakeword!r} has no loaded openwakeword model "
                f"(loaded: {', '.join(names)}). Set VIKI_WAKEWORD to one of "
                "the loaded names, or VIKI_WAKEWORD_MODEL to the .onnx file "
                "for the wake word you want.")
        if custom:
            return Path(custom).stem
        return cfg.wakeword

    def heard(self, frame: bytes) -> bool:
        try:
            import numpy as np
            audio = np.frombuffer(frame, dtype=np.int16)
        except ImportError:
            # No numpy in this environment (e.g. the test venv): a real
            # openwakeword install always brings numpy, so this path
            # only matters for injected fake models, which ignore the
            # frame contents. array gives the same int16 sequence.
            import array
            audio = array.array("h", frame)
        scores = self.model.predict(audio)
        return scores.get(self.score_key, 0.0) >= self.cfg.wake_threshold


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
                 agent_runner=None, lamp="auto", sleep_fn=None):
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
        # Stage-failure bookkeeping (finding 36): consecutive failures in
        # run() back off instead of spinning; a handled utterance — even
        # an empty/stop skip — resets the count. _last_stage_failed is
        # set by handle_utterance so run() can tell a failed turn from a
        # skipped one without parsing its return value (""). sleep_fn is
        # injectable so tests can observe the backoff without sleeping.
        self._consecutive_failures = 0
        self._last_stage_failed = False
        self._last_backoff_seconds = 0.0
        self._sleep_fn = sleep_fn or time.sleep

    def _append_history(self, text: str, reply: str):
        """Append one turn, keeping only the recent window (finding 39b)."""
        self._history += [{"role": "user", "content": text},
                          {"role": "assistant", "content": reply}]
        if len(self._history) > HISTORY_MAX_MESSAGES:
            self._history = self._history[-HISTORY_MAX_MESSAGES:]

    def _note_success(self):
        self._consecutive_failures = 0
        self._last_stage_failed = False

    def _note_failure(self):
        """Record one failed turn and back off after repeated failures.

        The first consecutive failure costs nothing (a single hiccup
        should not stall the loop); from the second on, sleep 0.25s,
        doubling each time up to 5s, so a persistently broken stage
        cannot tight-spin. Returns the delay slept.
        """
        self._consecutive_failures += 1
        self._last_stage_failed = True
        if self._consecutive_failures >= 2:
            delay = min(5.0, 0.25 * (2 ** (self._consecutive_failures - 2)))
        else:
            delay = 0.0
        self._last_backoff_seconds = delay
        if delay:
            self._sleep_fn(delay)
        return delay

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
        # VIKI takes a single toolbox argument (model comes from VIKI_MODEL
        # inside its constructor); the old model=/tools= kwargs raised
        # TypeError, so the headless voice path never ran at all (v0.3.17).
        viki = VIKI(tb)
        events = []

        out = viki.run(text, self._history, store.search,
                       emit=lambda k, p: events.append((k, p)),
                       approve=voice_approve, think=False)
        # run() returns {"type": ..., "content": ...} — the old code read
        # a "text" key that never exists, so every reply fell through to
        # the "came up empty" fallback even when the agent answered.
        reply = out.get("content") or ""
        if out.get("type") == "approval_denied":
            reply = ("That needs your approval, boss — "
                     "tap Allow in the web UI and ask me again.")
        return reply or "Hmm, I came up empty on that one."

    def handle_utterance(self, wav_path: Path) -> str:
        """Transcribe one utterance, run the agent, speak the reply. Returns reply.

        Finding 36: every stage is guarded. A transcribe, agent, or speak
        failure is caught (not raised), the lamp is restored, and "" is
        returned — except a speak failure, where the agent's reply is
        already in history, so the turn still counts as handled for
        history purposes. The caller (run()) reads _last_stage_failed to
        decide on backoff. Empty transcripts and stop commands are
        successful skips, not failures: they restore the lamp and leave
        _last_stage_failed False, so they never trigger a backoff or an
        error state (the same skip-as-success semantics finding 38 asks
        for, applied to the skip paths this file actually has).
        """
        self._ensure_stages(with_wake=False)
        self._last_stage_failed = False
        try:
            text = self.transcriber.transcribe(wav_path).strip()
        except Exception as exc:
            print(f"transcribe failed: {exc}")
            self._last_stage_failed = True
            self._lamp_set("restore")
            return ""
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
            try:
                reply = self.agent_runner(text)
            except Exception as exc:
                print(f"agent failed: {exc}")
                self._last_stage_failed = True
                return ""
            self._append_history(text, reply)
            print(f"reply: {reply}")
            self._lamp_set("speaking")  # pink while talking back
            try:
                self.speaker.speak(reply)
            except Exception as exc:
                print(f"speak failed: {exc}")
                self._last_stage_failed = True
                return ""
            return reply
        finally:
            self._lamp_set("restore")  # back to however the lamp was

    def _handle_audio(self, audio: bytes) -> bool:
        """Write one recorded utterance to a temp wav and handle it.

        Returns True when the utterance was handled (including an
        empty/stop skip) and False when a stage failed; it never raises
        for a stage failure, so run() can keep listening. Write errors
        are likewise caught, with the lamp restored.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)
        try:
            write_wav(wav_path, audio)
            self.handle_utterance(wav_path)
            return not self._last_stage_failed
        except Exception as exc:
            print(f"utterance handling failed: {exc}")
            self._last_stage_failed = True
            self._lamp_set("restore")
            return False
        finally:
            wav_path.unlink(missing_ok=True)

    def run(self):
        self._ensure_stages()
        print(f"Listening for '{self.cfg.wakeword}' — Ctrl+C to stop.")
        frames = self.mic.frames()
        try:
            while True:
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                except Exception as exc:
                    # Mic / frame-source failure: restore the lamp, back
                    # off after repeats, and keep listening. (Keyboard-
                    # Interrupt is a BaseException, so it still stops us.)
                    print(f"mic frame failed: {exc}")
                    self._lamp_set("restore")
                    self._note_failure()
                    continue
                try:
                    heard = self.wake.heard(frame)
                except Exception as exc:
                    print(f"wake-word check failed: {exc}")
                    self._lamp_set("restore")
                    self._note_failure()
                    continue
                if heard:
                    print("wake word!")
                    self._lamp_set("wake")  # purple
                    self._lamp_set("speaking")
                    try:
                        self.speaker.speak("Yes, boss?")
                    except Exception as exc:
                        # "Yes, boss?" is a courtesy, not the turn: log
                        # it, restore the lamp, and still listen for the
                        # utterance that follows the wake word.
                        print(f"'Yes, boss?' failed: {exc}")
                        self._lamp_set("restore")
                    # The wake -> utterance path is bounded by
                    # initial_silence_seconds inside record_until_silence,
                    # so silence after the wake word returns here instead
                    # of hanging; an empty result is a handled skip.
                    try:
                        utterance_ok = self._handle_audio(
                            record_until_silence(frames, self.cfg))
                    except Exception as exc:
                        print(f"utterance recording failed: {exc}")
                        self._lamp_set("restore")
                        self._note_failure()
                        continue
                    if not utterance_ok:
                        self._note_failure()
                        continue  # back to wake-word listening, not follow-up
                    self._note_success()
                    # Follow-up window: quick replies right after VIKI
                    # speaks skip the wake word. Silence through the whole
                    # window drops back to wake-word listening. No
                    # "Yes, boss?" here — the conversation is already open.
                    while self.cfg.followup_seconds > 0:
                        try:
                            followup = listen_for_followup(frames, self.cfg)
                        except Exception as exc:
                            print(f"follow-up listen failed: {exc}")
                            self._lamp_set("restore")
                            self._note_failure()
                            break
                        if followup is None:
                            break
                        try:
                            followup_ok = self._handle_audio(followup)
                        except Exception as exc:
                            print(f"follow-up handling failed: {exc}")
                            self._lamp_set("restore")
                            self._note_failure()
                            break
                        if not followup_ok:
                            self._note_failure()
                            break  # back to wake-word listening
                        self._note_success()
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
