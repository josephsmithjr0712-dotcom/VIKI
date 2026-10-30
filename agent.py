"""VIKI agent core: a plan -> act -> observe loop on top of Ollama tool calling."""
import json, os, secrets, time
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


# Cloud-model option (v0.3.18): when VIKI_CLOUD_MODEL, VIKI_CLOUD_BASE_URL
# and VIKI_CLOUD_API_KEY are all set, model turns go to an OpenAI-compatible
# chat-completions endpoint instead of local Ollama. Any cloud failure falls
# back to local Ollama — a turn never dies on the cloud. The key is only
# ever sent in the Authorization header, never logged, printed, or put in a
# URL. It lives in the server .env, never in code. Tool schemas pass
# through unchanged: local schemas already use OpenAI's function format.
def _cloud_config():
    """Cloud config from env, or None when not fully set.

    Empty values count as unset, so a half-configured .env stays local.
    Timeout is seconds per HTTP read (VIKI_CLOUD_TIMEOUT, default 60)."""
    model = (os.getenv("VIKI_CLOUD_MODEL") or "").strip()
    base = (os.getenv("VIKI_CLOUD_BASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("VIKI_CLOUD_API_KEY") or "").strip()
    if not (model and base and key):
        return None
    try:
        timeout = float(os.getenv("VIKI_CLOUD_TIMEOUT", "60"))
    except (TypeError, ValueError):
        timeout = 60.0
    if timeout <= 0:
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
        self.max_steps = int(os.getenv("VIKI_MAX_STEPS", "10"))
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
                    content.append(delta["content"])
                    emit("token", delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    if idx not in parts:
                        parts[idx] = {"name": "", "arguments": ""}
                        order.append(idx)
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        parts[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        parts[idx]["arguments"] += fn["arguments"]
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
            try:
                return self._stream_cloud(cloud, messages, tools, think, emit, cancel)
            except Cancelled:
                raise
            except Exception:
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
            try:
                for chunk in self.client.chat(**kw):
                    if cancel.is_set(): raise Cancelled()
                    m = chunk.message
                    if getattr(m, "thinking", None):
                        thinking.append(m.thinking); emit("thinking", m.thinking)
                    if m.content:
                        content.append(m.content); emit("token", m.content)
                    if m.tool_calls: calls.extend(m.tool_calls)
                return "".join(content), "".join(thinking), calls
            except ollama.ResponseError as e:
                err = str(e).lower()
                if think and "does not support thinking" in err:
                    self.think_ok, think = False, False
                    emit("status", f"{self.model} can't expose reasoning; showing steps instead.")
                elif tools and "does not support tools" in err:
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
                    raise
            except _TRANSIENT_ERRORS as e:
                # Network hiccup (dropped connection, reset peer, timeout):
                # retry the whole turn instead of killing the run.
                if retries_left > 0:
                    retries_left -= 1
                    emit("status", f"Connection hiccup talking to {self.model}, retrying ({retries_left} left)...")
                    time.sleep(delay)
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
        tools, used, steps = self.toolbox.schemas(), [], 0
        # Repeat-call guard state: the model sometimes loops on the same
        # failing (or pointless) tool call. Track consecutive identical
        # calls so we can nudge it, then stop re-executing.
        last_key, last_result, repeats = None, None, 0
        try:
            while steps < self.max_steps:
                steps += 1
                emit("status", f"Step {steps}: thinking" if steps > 1 else "Working on it")
                content, thinking, calls = self._stream(messages, tools, think and self.think_ok is not False, emit, cancel)
                if thinking: self.think_ok = True
                if not calls:
                    return {"type": "text", "content": content.strip(), "tools": used, "steps": steps}
                emit("reset_answer", "")  # text streamed before a tool call is narration, not the answer
                messages.append({"role": "assistant", "content": content, "tool_calls": [
                    {"function": {"name": c.function.name, "arguments": dict(c.function.arguments or {})}} for c in calls]})
                for i, c in enumerate(calls):
                    if cancel.is_set(): raise Cancelled()
                    name, args = c.function.name, dict(c.function.arguments or {})
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
                        messages.append({"role": "user", "content": (
                            f"System note: you just called {name} with identical arguments "
                            f"twice in a row. The last result was: {result[:400]} "
                            "Do not keep repeating this call -- try different arguments, "
                            "a different tool, or answer with what you have.")})
            # Out of steps: force a wrap-up with what we have.
            messages.append({"role": "user", "content": "Step limit reached. Summarize what you found and what is left."})
            content, _, _ = self._stream(messages, None, False, emit, cancel)
            return {"type": "text", "content": content.strip(), "tools": used, "steps": steps}
        except Cancelled:
            return {"type": "cancelled", "content": "Stopped.", "tools": used, "steps": steps}
        except Exception as e:
            return {"type": "error", "content": f"Could not reach the model: {e}. Is Ollama running and is '{self.model}' pulled?"}

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
