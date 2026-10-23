"""VIKI agent core: a plan -> act -> observe loop on top of Ollama tool calling."""
import json, os, secrets, time
from pathlib import Path
import ollama
from dotenv import load_dotenv


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


class VIKI:
    def __init__(self, toolbox):
        load_dotenv()
        self.model = os.getenv("VIKI_MODEL", "llama3.2:latest")
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

    def _stream(self, messages, tools, think, emit, cancel):
        """One model turn. Degrades gracefully when thinking/tools are unsupported.
        Transient connection errors are retried (VIKI_RETRIES, default 2) with
        exponential backoff (VIKI_RETRY_DELAY seconds, default 1.0) before the
        failure surfaces as an error result."""
        retries_left, delay = self.retries, self.retry_delay
        while True:
            kw = dict(model=self.model, messages=messages, stream=True, options={"temperature": 0.2, "top_p": 0.9})
            if tools: kw["tools"] = tools
            if think: kw["think"] = True
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

    def run(self, prompt, history, memory, emit, approve, think=False, cancel=None):
        """Run the agent loop. emit(kind, payload) streams events; approve(name, args) -> bool.
        memory is a list of fact strings, or a callable(query) -> list for relevance-ranked recall."""
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
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": prompt}]
        tools, used, steps = self.toolbox.schemas(), [], 0
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
                    emit("tool_start", {"id": cid, "name": name, "args": args})
                    if self.toolbox.needs_approval(name, args) and not approve(cid, name, args):
                        result = "The user denied this action. Do not retry it; offer an alternative."
                    else:
                        if name == "update_plan": emit("plan", args.get("steps") or [])
                        result = self.toolbox.run(name, args)
                    emit("tool_result", {"id": cid, "result": result[:1500]})
                    used.append({"name": name, "arguments": args, "result": result[:500]})
                    messages.append({"role": "tool", "tool_name": name, "content": result})
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
