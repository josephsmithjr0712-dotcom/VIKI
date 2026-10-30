"""VIKI agent core: a plan -> act -> observe loop on top of Ollama tool calling."""
import inspect, json, math, os, re, secrets, sys, time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
try:
    import ollama
except ImportError:
    # Graceful degradation: agent.py must stay importable where the model
    # client isn't installed (CI sandboxes, minimal installs). VIKI raises
    # a clear error at construction time if a model call is actually needed.
    ollama = None
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):  # no .env support; os.environ still works
        return False


def _load_personality():
    p = Path(__file__).resolve().parent / "personality.md"
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


PERSONALITY = _load_personality()

SYSTEM_PROMPT = """You are VIKI, an autonomous agent running on the user's own machine. You get things done, not just answer.

How you work:
- For anything needing more than one step, call update_plan first with 2-6 short steps, then work through them and update statuses as you go.
- Use tools proactively and chain them: search, then read_webpage on the best result, then calculate or write_file as needed. Never guess at current facts, math, or file contents.
- If a tool fails or returns nothing useful, try a different query or approach before giving up.
- Answer simple questions and chit-chat directly, with no tools.
- Save lasting facts about the user with remember (never secrets).
- When the user states a lasting like, dislike, hobby, or favorite, save it with update_profile so future answers fit them better. Check get_profile when tailoring a recommendation.
- You can check the user's connections any time with check_connections (unread Gmail + today's calendar), and set up recurring checks with schedule_task (asks approval) — e.g. a morning brief.
- Risky actions (writing files, controlling devices) ask the user for approval. If denied, adapt and offer alternatives.
- Finish with a concise answer in plain language. Mention source URLs when you used the web. Never print raw JSON or tool schemas."""

EXTRACT_PROMPT = (
    "You maintain the long-term memory of an AI agent. From the USER messages below, extract durable facts worth "
    "remembering: name, preferences, projects, tools, goals, standing instructions. Only things clearly stated; never guess. "
    "Ignore one-off questions. If a message corrects or replaces a known fact, include the corrected version. "
    "One short third-person sentence each, e.g. 'User prefers Fahrenheit.' Skip facts already known and unchanged. "
    'Never store secrets. Reply as JSON: {"facts": ["..."]}'
)
EXTRACT_SCHEMA = {"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "string"}}}, "required": ["facts"]}


class Cancelled(Exception):
    pass


# Connection-level failures worth retrying before giving up on a model turn.
# ollama.RequestError is the real library's signal for broken HTTP/connection
# state; the builtins cover the rest. Resolved via getattr because test
# sandboxes stub the ollama module with only what agent.py needs.
_TRANSIENT_ERRORS = tuple(
    e for e in (getattr(ollama, "RequestError", None), ConnectionError, OSError) if e is not None
)

# Repeat-call guard: after this many consecutive identical tool calls
# (same tool, same arguments), stop executing and serve the cached result
# with a steer to change approach, instead of burning steps in a loop.
_MAX_IDENTICAL_CALLS = 3

# Empty-answer guard (v0.3.39): a model turn that emits only reasoning
# (stripped by _ThinkFilter) or nothing at all used to return "" — the
# web UI then showed "(no answer)" and the clock spoke nothing and went
# back to Listening, exactly the failure seen 2026-10-29 evening. One
# retry with a direct-answer nudge, then an honest fallback sentence
# instead of silence.
_EMPTY_ANSWER_FALLBACK = "I couldn't come up with an answer just now — please try asking again."

# Tool-result history cap: a single huge tool result (a long web page, a
# big file read) would otherwise land verbatim in every later model turn,
# bloating context and slowing the small local model. History keeps the
# first chunk plus a marker saying it was cut; the frontend's own
# tool_result slice is unchanged.
_MAX_TOOL_RESULT_CHARS = 4000


def _truncate_tool_result(result):
    """Cap a tool result for model history, marking any cut."""
    if len(result) <= _MAX_TOOL_RESULT_CHARS:
        return result
    return (result[:_MAX_TOOL_RESULT_CHARS]
            + f"\n[tool result truncated: showing first {_MAX_TOOL_RESULT_CHARS} of {len(result)} characters]")


def _args_key(args):
    """Canonical string for a tool-call argument dict (for repeat detection)."""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return repr(sorted((str(k), str(v)) for k, v in args.items()))


# Tag forms, case-insensitive with optional whitespace: <think>, <Think>,
# < think >, </think>, </ think >. A partial tag split across chunks is
# held back, never emitted mid-tag.
_THINK_OPEN = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
# A buffer suffix that could still grow into a tag: starts with "<" and
# contains no ">" yet, and its letters (if any) follow the think sequence.
_PARTIAL_TAG = re.compile(r"^<\s*/?\s*t?h?i?n?k?\s*$", re.IGNORECASE)


def _partial_tag_len(text):
    """Length of the longest suffix of text that could be a partial tag."""
    for n in range(min(len(text), 12), 0, -1):
        if _PARTIAL_TAG.match(text[-n:]):
            return n
    return 0


class _ThinkFilter:
    """Stateful stripper for <think>...</think> leaked into answer text (v0.3.38).

    The clock speaks content tokens and ignores the thinking channel, so
    reasoning that Ollama puts into the content stream — which happens
    even with think=False on some servers/models — would be spoken aloud.
    feed() returns only answer text, holding back a tail that could be a
    tag split across chunks. Nested <think> blocks are depth-counted, a
    stray </think> outside any block is stripped instead of spoken, and
    tags match case-insensitively with optional whitespace. flush() at
    stream end drops an unclosed <think> block rather than leaking
    partial reasoning, but returns any held non-think tail (ordinary
    text that only looked like a tag prefix) so it is not lost.
    """

    def __init__(self):
        self.depth = 0
        self.buf = ""

    def feed(self, text):
        self.buf += text or ""
        out = []
        while self.buf:
            if self.depth:
                m_open = _THINK_OPEN.search(self.buf)
                m_close = _THINK_CLOSE.search(self.buf)
                # Earliest tag wins: a nested open deepens, a close shallows.
                if m_close and (not m_open or m_close.start() <= m_open.start()):
                    self.depth -= 1
                    self.buf = self.buf[m_close.end():]
                    continue
                if m_open:
                    self.depth += 1
                    self.buf = self.buf[m_open.end():]
                    continue
                # No complete tag left: hold a possible partial close tag,
                # drop the rest (it is inside a think block).
                keep = _partial_tag_len(self.buf)
                self.buf = self.buf[-keep:] if keep else ""
                break
            m_open = _THINK_OPEN.search(self.buf)
            m_close = _THINK_CLOSE.search(self.buf)
            if m_close and (not m_open or m_close.start() < m_open.start()):
                # Stray close tag: emit the text before it, drop the tag.
                out.append(self.buf[:m_close.start()])
                self.buf = self.buf[m_close.end():]
                continue
            if m_open:
                out.append(self.buf[:m_open.start()])
                self.depth = 1
                self.buf = self.buf[m_open.end():]
                continue
            keep = _partial_tag_len(self.buf)
            if keep:
                out.append(self.buf[:-keep])
                self.buf = self.buf[-keep:]
            else:
                out.append(self.buf)
                self.buf = ""
            break
        return "".join(out)

    def flush(self):
        if self.depth:
            # Unclosed think block: drop it, never leak partial reasoning.
            self.buf = ""
            self.depth = 0
            return ""
        tail, self.buf = self.buf, ""
        return tail


# Cloud-model option (v0.3.18): when VIKI_CLOUD_MODEL, VIKI_CLOUD_BASE_URL
# and VIKI_CLOUD_API_KEY are all set, model turns go to an OpenAI-compatible
# chat-completions endpoint instead of local Ollama. Any cloud failure falls
# back to local Ollama — a turn never dies on the cloud. The key is only
# ever sent in the Authorization header, never logged, printed, or put in a
# URL. It lives in the server .env, never in code. Tool schemas pass
# through unchanged: local schemas already use OpenAI's function format.
# Gemini shortcut (v0.3.39): a gemini-* model needs no base URL — the
# Google OpenAI-compatible endpoint is the default, keyed by
# VIKI_GEMINI_API_KEY (or VIKI_CLOUD_API_KEY).
_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _cloud_config():
    """Cloud config from env, or None when not fully set.

    Generic OpenAI-compatible endpoint: VIKI_CLOUD_MODEL +
    VIKI_CLOUD_BASE_URL + VIKI_CLOUD_API_KEY.

    Gemini shortcut (v0.3.39): Google's Gemini API exposes an
    OpenAI-compatible endpoint, so a Gemini model needs no base URL —
    set VIKI_GEMINI_API_KEY (from Google AI Studio, kept in server .env
    only, never in chat) and VIKI_CLOUD_MODEL (or VIKI_GEMINI_MODEL) to
    e.g. gemini-2.5-flash (fast, best for voice) or gemini-2.5-pro
    (smarter, slower). An explicit VIKI_CLOUD_BASE_URL always wins, so
    a proxy or Vertex AI endpoint can still be used. VIKI_CLOUD_API_KEY
    is accepted as the key for a Gemini model too. Empty values count
    as unset, so a half-configured .env stays local. Timeout is seconds
    per HTTP read (VIKI_CLOUD_TIMEOUT, default 60). The key is only
    ever sent in the Authorization header, never logged or in a URL."""
    model = (os.getenv("VIKI_CLOUD_MODEL") or os.getenv("VIKI_GEMINI_MODEL") or "").strip()
    base = (os.getenv("VIKI_CLOUD_BASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("VIKI_CLOUD_API_KEY") or os.getenv("VIKI_GEMINI_API_KEY") or "").strip()
    if not base and (model.lower() == "gemini" or model.lower().startswith("gemini-")):
        base = _GEMINI_OPENAI_BASE_URL
    if not (model and base and key):
        return None
    try:
        timeout = float(os.getenv("VIKI_CLOUD_TIMEOUT", "60"))
    except (TypeError, ValueError):
        timeout = 60.0
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = 60.0
    return {"model": model, "base_url": base, "api_key": key, "timeout": timeout}


def _cloud_messages(messages):
    """Translate a local message list to OpenAI chat format.

    Assistant tool calls get synthetic ids (local calls carry none) and the
    following tool messages are paired to them in order; tool_name is
    dropped (OpenAI links via tool_call_id instead). Raises ValueError on
    a shape that cannot be paired, so the caller falls back to local."""
    out, pending, n = [], [], 0
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", {}) if isinstance(c, dict) else getattr(c, "function", {})
                name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", {})
                call_id = f"call_{n}"
                n += 1
                pending.append(call_id)
                calls.append({"id": call_id, "type": "function",
                              "function": {"name": name,
                                           "arguments": args if isinstance(args, str) else json.dumps(args or {})}})
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        elif role == "tool":
            if not pending:
                raise ValueError("tool message without a preceding tool call")
            out.append({"role": "tool", "tool_call_id": pending.pop(0),
                        "content": m.get("content") or ""})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    if pending:
        raise ValueError("assistant tool calls without matching tool messages")
    return out


class VIKI:
    def __init__(self, toolbox):
        load_dotenv()
        self.model = os.getenv("VIKI_MODEL", "llama3.2:latest")
        if ollama is None:
            raise RuntimeError(
                "The 'ollama' package is not installed, so VIKI cannot talk to a model. "
                "Install it with: pip install ollama"
            )
        self.client = ollama.Client(host=os.getenv("VIKI_ADDRESS", "http://localhost:11434"))
        try:
            self.max_steps = max(1, int(os.getenv("VIKI_MAX_STEPS", "10")))
        except (TypeError, ValueError):
            self.max_steps = 10
        try:
            self.retries = max(0, int(os.getenv("VIKI_RETRIES", "2")))
        except (TypeError, ValueError):
            self.retries = 2
        try:
            self.retry_delay = max(0.0, float(os.getenv("VIKI_RETRY_DELAY", "1.0")))
        except (TypeError, ValueError):
            self.retry_delay = 1.0
        self.toolbox = toolbox
        self.think_ok = None
        self.tools_ok = None  # mirrors think_ok: False once a model refuses tools

    def warmup(self):
        """Pre-load the model into Ollama's memory so the first real request is fast.
        Call once at startup. Failures are logged, never fatal."""
        try:
            # Minimal non-streaming request forces Ollama to load the model.
            # keep_alive: -1 tells Ollama to hold it in memory indefinitely
            # (overridden by the server's OLLAMA_KEEP_ALIVE if set).
            self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                keep_alive=-1,
                options={"num_predict": 1},
            )
            print(f"[warmup] model {self.model} pre-loaded and held in memory")
        except Exception as e:
            print(f"[warmup] could not pre-load {self.model}: {e} (will load on first request)")

    def _stream_cloud(self, cfg, messages, tools, think, emit, cancel):
        """One model turn via an OpenAI-compatible chat-completions endpoint.

        Returns (content, thinking, calls) like _stream; thinking is always
        "" here (cloud deltas don't expose reasoning). Raises on ANY failure
        (network, non-2xx, bad payload, unpaired tool call) so the caller
        can fall back to local Ollama. The API key travels only in the
        Authorization header and is never included in any error."""
        if think:
            # Cloud deltas carry no reasoning channel: say so up front
            # instead of silently dropping a requested think=True.
            emit("status", "Cloud model doesn't expose reasoning; answering without it.")
        payload = {"model": cfg["model"], "messages": _cloud_messages(messages),
                   "stream": True, "temperature": 0.2, "top_p": 0.9}
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(
            cfg["base_url"] + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + cfg["api_key"]},
            method="POST")
        content, parts, order = [], {}, []
        filt = _ThinkFilter()
        next_idx = 0
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
            for raw in resp:
                if cancel.is_set():
                    raise Cancelled()
                line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)  # raises on bad payload -> fallback
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    clean = filt.feed(delta["content"])
                    if clean:
                        content.append(clean)
                        emit("token", clean)
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    raw_idx = tc.get("index")
                    if isinstance(raw_idx, int) and not isinstance(raw_idx, bool):
                        idx = raw_idx
                        if (idx in parts and parts[idx]["name"] and fn.get("name")
                                and fn["name"] != parts[idx]["name"]):
                            raise ValueError("cloud tool call index reused with a different name")
                        next_idx = max(next_idx, idx + 1)
                    elif fn.get("name") or not order:
                        # Provider omitted the index and this chunk names
                        # a call (or is the first chunk): start a new
                        # sequential call instead of merging into index 0.
                        idx = next_idx
                        next_idx += 1
                    else:
                        # Index-less continuation chunk (arguments only):
                        # it belongs to the most recent call.
                        idx = order[-1]
                    if idx not in parts:
                        parts[idx] = {"name": "", "arguments": ""}
                        order.append(idx)
                    if fn.get("name"):
                        parts[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        parts[idx]["arguments"] += fn["arguments"]
        tail = filt.flush()
        if tail:
            content.append(tail)
            emit("token", tail)
        calls = []
        for idx in order:
            part = parts[idx]
            if not part["name"]:
                raise ValueError("cloud tool call without a name")
            try:
                args = json.loads(part["arguments"] or "{}")
            except ValueError:
                raise ValueError("cloud tool call with unparsable arguments")
            if not isinstance(args, dict):
                raise ValueError("cloud tool call arguments are not an object")
            calls.append(SimpleNamespace(function=SimpleNamespace(name=part["name"], arguments=args)))
        return "".join(content), "", calls

    def _stream(self, messages, tools, think, emit, cancel):
        """One model turn. Degrades gracefully when thinking/tools are unsupported.
        Transient connection errors are retried (VIKI_RETRIES, default 2) with
        exponential backoff (VIKI_RETRY_DELAY seconds, default 1.0) before the
        failure surfaces as an error result.
        When the cloud-model env vars are all set (and the turn has no images,
        which the cloud path does not translate), the turn goes to the cloud
        first; any cloud failure falls back to local Ollama below. The
        exception text is never surfaced: it could theoretically echo
        request details."""
        cloud = _cloud_config()
        if cloud and not any(m.get("images") for m in messages):
            # Finding 7 (judgment call, flagged for veto): cloud circuit
            # breaker. After N consecutive cloud failures the cloud path
            # pauses for a cooldown instead of being re-tried (and
            # re-failed) on every turn; the local model carries the
            # turns in between. Defaults: 3 failures, 5-minute cooldown
            # (VIKI_CLOUD_BREAKER_FAILURES / VIKI_CLOUD_BREAKER_COOLDOWN).
            # No response memoisation: caching cloud answers risks
            # serving stale replies, so only the failure path is gated.
            if time.monotonic() < getattr(self, "_cloud_cooldown_until", 0):
                pass  # breaker open: skip straight to the local model
            else:
                cloud_tokens = []

                def _cloud_emit(kind, payload):
                    # Finding 8 (judgment call): if the cloud stream
                    # already emitted tokens and then fails, the local
                    # fallback would re-stream the whole answer on top
                    # of them. Record tokens so the fallback can clear
                    # them first (reset_answer) instead of duplicating.
                    if kind == "token":
                        cloud_tokens.append(payload)
                    emit(kind, payload)

                try:
                    result = self._stream_cloud(cloud, messages, tools, think, _cloud_emit, cancel)
                    self._cloud_failures = 0
                    return result
                except Cancelled:
                    raise
                except Exception:
                    failures = getattr(self, "_cloud_failures", 0) + 1
                    self._cloud_failures = failures
                    try:
                        threshold = int(os.getenv("VIKI_CLOUD_BREAKER_FAILURES", "3"))
                    except (TypeError, ValueError):
                        threshold = 3
                    if threshold > 0 and failures >= threshold:
                        try:
                            cooldown = float(os.getenv("VIKI_CLOUD_BREAKER_COOLDOWN", "300"))
                        except (TypeError, ValueError):
                            cooldown = 300.0
                        self._cloud_cooldown_until = time.monotonic() + max(0.0, cooldown)
                        emit("status", f"Cloud model paused after {failures} repeated failures; using the local model for the next {cooldown:g} seconds.")
                    if cloud_tokens:
                        emit("reset_answer", "")
                    emit("status", "Cloud model unavailable, falling back to the local model.")
        retries_left, delay = self.retries, self.retry_delay
        while True:
            kw = dict(model=self.model, messages=messages, stream=True, options={"temperature": 0.2, "top_p": 0.9})
            if tools: kw["tools"] = tools
            # Always send think explicitly (v0.3.37): omitting it let the
            # model default win, and qwen3 defaults thinking ON — so a
            # turn with reasoning off still paid for a long reasoning
            # trace before answering. think=False here is a no-op for
            # models without thinking support.
            kw["think"] = think
            content, thinking, calls = [], [], []
            filt = _ThinkFilter()
            try:
                for chunk in self.client.chat(**kw):
                    if cancel.is_set(): raise Cancelled()
                    m = chunk.message
                    if getattr(m, "thinking", None):
                        thinking.append(m.thinking); emit("thinking", m.thinking)
                    if m.content:
                        clean = filt.feed(m.content)
                        if clean:
                            content.append(clean); emit("token", clean)
                    if m.tool_calls: calls.extend(m.tool_calls)
                tail = filt.flush()
                if tail:
                    content.append(tail); emit("token", tail)
                return "".join(content), "".join(thinking), calls
            except ollama.ResponseError as e:
                # Finding 8: this attempt already streamed some tokens
                # to the UI (and, on the clock, to speech). A retry
                # re-streams the whole answer, so clear the partial
                # text first instead of letting it duplicate.
                if content:
                    emit("reset_answer", "")
                err = str(e).lower()
                if think and "does not support thinking" in err:
                    self.think_ok, think = False, False
                    emit("status", f"{self.model} can't expose reasoning; showing steps instead.")
                elif tools and "does not support tools" in err:
                    self.tools_ok = False
                    tools = None
                    emit("status", f"{self.model} has no tool support, so I'm answering without tools. Try a model like llama3.1, qwen3, or mistral-nemo.")
                elif any(m.get("images") for m in messages) and any(
                        w in err for w in ("image", "vision", "multimodal")):
                    # Model can't see images: strip them and continue on text.
                    for m in messages:
                        m.pop("images", None)
                    emit("status", f"{self.model} can't process images, so I'm working from your text only. "
                                   "A vision model like qwen3-vl or gemma3 would see them.")
                else:
                    # Server-side failure (5xx) or rate limit (429):
                    # retry like a transient error instead of dying.
                    status = getattr(e, "status_code", None)
                    if status is None:
                        m = re.search(r"\b(429|5\d\d)\b", err)
                        status = int(m.group(1)) if m else None
                    if status == 429 or (isinstance(status, int) and 500 <= status <= 599):
                        if retries_left > 0:
                            retries_left -= 1
                            emit("status", f"{self.model} returned {status}, retrying ({retries_left} left)...")
                            if cancel.wait(delay):
                                raise Cancelled()
                            delay *= 2
                            continue
                    raise
            except _TRANSIENT_ERRORS as e:
                # Finding 8: same partial-token clear as above before retrying.
                if content:
                    emit("reset_answer", "")
                # Network hiccup (dropped connection, reset peer, timeout):
                # retry the whole turn instead of killing the run.
                if retries_left > 0:
                    retries_left -= 1
                    emit("status", f"Connection hiccup talking to {self.model}, retrying ({retries_left} left)...")
                    if cancel.wait(delay):
                        raise Cancelled()
                    delay *= 2
                    continue
                raise

    def run(self, prompt, history, memory, emit, approve, think=False, cancel=None, images=None, profile=""):
        """Run the agent loop. emit(kind, payload) streams events; approve(name, args) -> bool.
        memory is a list of fact strings, or a callable(query) -> list for relevance-ranked recall.
        images: optional list of raw image bytes attached to this turn (Ollama vision format).
        If the model can't process images, _stream strips them and continues on text."""
        import threading
        cancel = cancel or threading.Event()
        used, steps = [], 0
        try:
            system = (PERSONALITY + "\n\n" if PERSONALITY else "") + SYSTEM_PROMPT
            # Tell the model what smart-home devices actually exist, so it uses
            # real entity_ids instead of guessing. Refreshed at server startup.
            try:
                from tools import ha_entities_text
                ha_list = ha_entities_text()
                if ha_list:
                    system += ("\n\nYour smart home devices (from Home Assistant - use these exact entity_ids, "
                               "never invent new ones):\n" + ha_list)
            except Exception:
                pass
            facts = memory(prompt) if callable(memory) else (memory or [])
            if facts:
                system += "\n\nKnown about the user (use when relevant, don't recite):\n" + "\n".join(f"- {m}" for m in facts)
            if profile:
                system += "\n\nUser profile (likes, dislikes, hobbies, favorites — use to tailor answers, don't recite):\n" + profile
            messages = [{"role": "system", "content": system}, *history]
            user_msg = {"role": "user", "content": prompt}
            if images:
                user_msg["images"] = images
            messages.append(user_msg)
            tools = self.toolbox.schemas()
            # Toolboxes that predate the cancel argument (older code,
            # test doubles) keep working: only pass cancel when the
            # run method actually accepts it.
            try:
                _run_params = inspect.signature(self.toolbox.run).parameters
                run_accepts_cancel = ("cancel" in _run_params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in _run_params.values()))
            except (TypeError, ValueError):
                run_accepts_cancel = False
            # Repeat-call guard state: the model sometimes loops on the same
            # failing (or pointless) tool call. Track consecutive identical
            # calls so we can nudge it, then stop re-executing.
            last_key, last_result, repeats = None, None, 0
            empty_retries = 0
            while steps < self.max_steps:
                steps += 1
                emit("status", f"Step {steps}: thinking" if steps > 1 else "Working on it")
                turn_tools = None if self.tools_ok is False else tools
                content, thinking, calls = self._stream(messages, turn_tools, think and self.think_ok is not False, emit, cancel)
                if thinking: self.think_ok = True
                if calls: self.tools_ok = True
                if not calls:
                    if not content.strip():
                        if empty_retries < 1:
                            empty_retries += 1
                            messages.append({"role": "user", "content": (
                                "You returned no answer text. Answer the user's question "
                                "directly in a short reply, with no reasoning preamble.")})
                            continue
                        return {"type": "text", "content": _EMPTY_ANSWER_FALLBACK,
                                "tools": used, "steps": steps}
                    return {"type": "text", "content": content.strip(), "tools": used, "steps": steps}
                emit("reset_answer", "")  # text streamed before a tool call is narration, not the answer
                norm_calls = []
                for c in calls:
                    try:
                        fn = c.function
                        name = fn.name if fn is not None else None
                        raw_args = fn.arguments if fn is not None else None
                        if not isinstance(name, str) or not name:
                            raise ValueError("missing tool name")
                        if raw_args is None:
                            args = {}
                        elif isinstance(raw_args, dict):
                            args = dict(raw_args)
                        elif isinstance(raw_args, str):
                            args = json.loads(raw_args)
                            if not isinstance(args, dict):
                                raise ValueError("arguments are not an object")
                        else:
                            raise ValueError("arguments are not an object")
                        norm_calls.append((name, args))
                    except (ValueError, AttributeError, TypeError):
                        # Malformed call from the model (string args that
                        # don't parse, missing name): tell the model and
                        # keep looping instead of crashing the turn.
                        norm_calls.append(None)
                assistant_calls = []
                for nc in norm_calls:
                    if nc is None:
                        assistant_calls.append({"function": {"name": "", "arguments": {}}})
                    else:
                        assistant_calls.append({"function": {"name": nc[0], "arguments": nc[1]}})
                messages.append({"role": "assistant", "content": content, "tool_calls": assistant_calls})
                defer_note = None
                for i, nc in enumerate(norm_calls):
                    if cancel.is_set(): raise Cancelled()
                    if nc is None:
                        cid = secrets.token_urlsafe(16)
                        result = "Malformed tool call ignored: no usable tool name or arguments. Call a real tool with a JSON object of arguments, or answer directly."
                        emit("tool_result", {"id": cid, "result": result})
                        messages.append({"role": "tool", "tool_name": "", "content": result})
                        continue
                    name, args = nc
                    cid = secrets.token_urlsafe(16)
                    key = (name, _args_key(args))
                    repeats = repeats + 1 if key == last_key else 1
                    last_key = key
                    emit("tool_start", {"id": cid, "name": name, "args": args})
                    executed = False
                    if repeats >= _MAX_IDENTICAL_CALLS:
                        # Stuck in a loop: hand back the previous result
                        # instead of executing again. This also avoids
                        # re-prompting approval for a call already seen.
                        result = (last_result if last_result is not None
                                  else "No previous result.") + (
                            " You have now called this tool with identical arguments "
                            f"{repeats} times in a row. Do not call it again with these "
                            "arguments -- try different arguments, a different tool, "
                            "or answer with what you have.")
                        emit("status", f"Skipped {name}: identical call #{repeats} in a row")
                    elif self.toolbox.needs_approval(name, args) and not approve(cid, name, args):
                        result = "The user denied this action. Do not retry it; offer an alternative."
                    else:
                        if name == "update_plan": emit("plan", args.get("steps") or [])
                        if run_accepts_cancel:
                            result = self.toolbox.run(name, args, cancel)
                        else:
                            result = self.toolbox.run(name, args)
                        executed = True
                    if repeats < _MAX_IDENTICAL_CALLS:
                        last_result = result
                    emit("tool_result", {"id": cid, "result": result[:1500]})
                    # Map card: after a maps tool actually runs, hand the
                    # frontend the structured route payload (geometry,
                    # endpoints, links, traffic) so it can render the map
                    # in-chat. Skipped/denied calls get no card, and a
                    # stale route (different endpoints) is never shown.
                    if executed and name in ("maps_directions", "maps_distance", "maps_traffic"):
                        try:
                            from connectors.maps import take_map_payload
                            payload = take_map_payload(args.get("origin"), args.get("destination"))
                        except Exception:
                            payload = None
                        if payload:
                            emit("map", payload)
                    used.append({"name": name, "arguments": args, "result": result[:500]})
                    messages.append({"role": "tool", "tool_name": name, "content": _truncate_tool_result(result)})
                    if repeats == 2:
                        defer_note = (name, result)
                if defer_note:
                    # Deferred until after ALL tool messages of this turn:
                    # a note inserted mid-turn would leave later tool
                    # messages without a preceding assistant call.
                    dname, dresult = defer_note
                    messages.append({"role": "user", "content": (
                        f"System note: you just called {dname} with identical arguments "
                        f"twice in a row. The last result was: {dresult[:400]} "
                        "Do not keep repeating this call -- try different arguments, "
                        "a different tool, or answer with what you have.")})
            # Out of steps: force a wrap-up with what we have.
            messages.append({"role": "user", "content": "Step limit reached. Summarize what you found and what is left."})
            try:
                content, _, _ = self._stream(messages, None, False, emit, cancel)
            except Cancelled:
                raise
            except Exception:
                content = ""
            return {"type": "text", "content": content.strip() or _EMPTY_ANSWER_FALLBACK,
                    "tools": used, "steps": steps}
        except Cancelled:
            return {"type": "cancelled", "content": "Stopped.", "tools": used, "steps": steps}
        except Exception as e:
            # User-facing text stays generic; the detail goes to the
            # server log only, never into a chat bubble.
            print(f"[agent] run failed: {type(e).__name__}: {e}", file=sys.stderr)
            return {"type": "error", "content": "Could not reach the model. Is Ollama running and is the model pulled?",
                    "tools": used, "steps": steps}

    def extract_facts(self, user_messages, known=None):
        if not user_messages: return []
        body = "Already known:\n" + ("\n".join(f"- {k}" for k in known or []) or "(nothing)") + \
               "\n\nUser messages:\n" + "\n".join(f"- {m[:500]}" for m in user_messages[-40:])
        try:
            r = self.client.chat(model=self.model, format=EXTRACT_SCHEMA, options={"temperature": 0},
                                 messages=[{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": body}])
            facts = json.loads(r.message.content or "{}").get("facts", [])
        except Exception:
            return []
        return [f for f in facts if isinstance(f, str)]
