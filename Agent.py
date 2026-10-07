import os
import re
import json
import urllib.parse
import urllib.request

import ollama
from dotenv import load_dotenv

SYSTEM_PROMPT = (
    "You are VIKI, a helpful AI assistant. Answer questions directly with text. "
    "Only use a tool when the user explicitly asks for what that tool provides "
    "(for example, weather or a forecast)."
)

MAX_TOOL_ROUNDS = 3

# Small models over-call tools, so we only offer the weather tool when the
# prompt actually looks like a weather question.
WEATHER_PATTERN = re.compile(
    r"\b(weather|temperature|forecast|rain(ing)?|snow(ing)?|humidity|windy|degrees)\b",
    re.IGNORECASE,
)


EXTRACT_PROMPT = (
    "You maintain the long-term memory of an AI assistant. From the USER "
    "messages below, extract durable facts worth remembering in future "
    "conversations: the user's name, preferences, projects, hardware or tools "
    "they use, goals, and standing instructions about how they want the "
    "assistant to behave.\n"
    "Rules:\n"
    "- Only include things the user clearly stated about themselves. Never guess.\n"
    "- Ignore one-off questions, small talk, and details that only matter for "
    "the current task.\n"
    "- Write each fact as one short sentence in the third person, "
    "e.g. 'User prefers Fahrenheit.'\n"
    "- Skip anything already in the known list.\n"
    "- Never store passwords, API keys, tokens or other secrets.\n"
    "- If nothing qualifies, return an empty list.\n"
    'Reply as JSON: {"facts": ["...", "..."]}'
)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}


def _emit(on_event, kind, text):
    if on_event:
        on_event(kind, text)


# --- Tool implementations -------------------------------------------------
def get_current_weather(location: str, unit: str = "Fahrenheit") -> str:
    """Fetch current weather from wttr.in (no API key needed)."""
    try:
        url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.load(r)
        cur = data["current_condition"][0]
        use_f = (unit or "").lower() == "fahrenheit"
        temp = cur["temp_F"] if use_f else cur["temp_C"]
        desc = cur["weatherDesc"][0]["value"]
        return f"{location}: {desc}, {temp}°{'F' if use_f else 'C'}"
    except Exception as e:
        return f"Weather lookup failed for {location}: {e}"


TOOL_REGISTRY = {"get_current_weather": get_current_weather}


class VIKI:
    def __init__(self):
        load_dotenv()
        self.model = os.getenv("VIKI_MODEL", "llama3.2:latest")
        host = os.getenv("VIKI_ADDRESS", "http://localhost:11434")
        self.client = ollama.Client(host=host)
        self.think_supported = None  # unknown until first attempt

    def get_available_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": (
                        "Get the current weather or temperature for a location. "
                        "ONLY use if the user explicitly asks about weather or "
                        "forecast. NEVER use for general geography questions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

    def _consume(self, stream, on_event):
        """Read a streamed reply, forwarding thinking/answer text as it arrives."""
        content, thinking, calls = [], [], []
        for chunk in stream:
            msg = chunk.message
            t = getattr(msg, "thinking", None)
            if t:
                thinking.append(t)
                _emit(on_event, "thinking", t)
            if msg.content:
                content.append(msg.content)
                _emit(on_event, "token", msg.content)
            if msg.tool_calls:
                calls.extend(msg.tool_calls)
        return "".join(content), "".join(thinking), calls

    def _chat(self, messages, tools, think, on_event):
        """Stream a reply; degrade gracefully if thinking/tools are unsupported."""
        opts = {"temperature": 0.1, "top_p": 0.9}
        while True:
            kwargs = dict(model=self.model, messages=messages, options=opts, stream=True)
            if tools:
                kwargs["tools"] = tools
            if think:
                kwargs["think"] = True
            try:
                return self._consume(self.client.chat(**kwargs), on_event)
            except ollama.ResponseError as e:
                err = str(e).lower()
                if think and "does not support thinking" in err:
                    self.think_supported = False
                    think = False
                    _emit(on_event, "status",
                          f"{self.model} can't show its reasoning, so I'll show my steps instead.")
                    continue
                if tools and "does not support tools" in err:
                    tools = None
                    _emit(on_event, "status",
                          f"{self.model} has no tool support, answering without tools.")
                    continue
                raise

    def extract_facts(self, user_messages, known=None):
        """Ask the model for durable facts about the user. Returns a list of strings."""
        if not user_messages:
            return []
        known = known or []
        content = (
            "Already known:\n"
            + ("\n".join(f"- {k}" for k in known) or "(nothing)")
            + "\n\nUser messages:\n"
            + "\n".join(f"- {m[:500]}" for m in user_messages[-40:])
        )
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": content},
                ],
                format=EXTRACT_SCHEMA,
                options={"temperature": 0},
            )
            data = json.loads(resp.message.content or "{}")
        except Exception:
            return []
        facts = data.get("facts", []) if isinstance(data, dict) else []
        return [f for f in facts if isinstance(f, str)]

    def viki_response(self, prompt: str, enable_tools: bool = True,
                      history=None, on_event=None, think: bool = False,
                      memory=None):
        """
        memory:   list of long-term facts about the user from earlier sessions.
        history:  prior {"role", "content"} user/assistant turns.
        on_event: callback(kind, text). kinds:
                  "status"   - what VIKI is doing (shown when think=True)
                  "thinking" - the model's own reasoning (thinking models only)
                  "token"    - a piece of the final answer
                  "tool"     - a tool call and its result
        """
        system = SYSTEM_PROMPT
        if memory:
            system += (
                "\n\nThings you know about the user from earlier sessions "
                "(use them only when relevant, and don't bring them up "
                "unprompted):\n" + "\n".join(f"- {m}" for m in memory)
            )

        messages = [
            {"role": "system", "content": system},
            # Few-shot example showing plain answers are fine
            {"role": "user", "content": "What is the capital of Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."},
            *(history or []),
            {"role": "user", "content": prompt},
        ]

        # Gate: only offer tools when the prompt (or the previous user turn,
        # for follow-ups like "what about Paris?") looks relevant
        recent = [m["content"] for m in (history or []) if m["role"] == "user"][-1:]
        use_tools = enable_tools and bool(WEATHER_PATTERN.search(" ".join(recent + [prompt])))
        tools = self.get_available_tools() if use_tools else None
        executed, all_thinking = [], []

        if use_tools:
            _emit(on_event, "status", "This looks like a weather question, so I'll offer the weather tool.")
        else:
            _emit(on_event, "status", "No tool needed, answering directly.")

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                use_think = think and self.think_supported is not False
                content, thinking, calls = self._chat(messages, tools, use_think, on_event)
                if thinking:
                    self.think_supported = True
                    all_thinking.append(thinking)

                if not calls:
                    content = content.strip()
                    if executed and not content:
                        content = executed[-1]["result"]
                    return {
                        "type": "tool_call" if executed else "text",
                        "content": content,
                        "thinking": "\n".join(all_thinking),
                        "tool_calls": executed,
                    }

                # Keep the assistant's tool-call turn in the history
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {"function": {"name": tc.function.name,
                                      "arguments": dict(tc.function.arguments or {})}}
                        for tc in calls
                    ],
                })

                for tc in calls:
                    name = tc.function.name
                    args = dict(tc.function.arguments or {})

                    func = TOOL_REGISTRY.get(name)
                    if func is None:
                        result = f"Unknown tool: {name}"
                    else:
                        try:
                            result = func(**args)
                        except TypeError as e:
                            result = f"Bad arguments for {name}: {e}"

                    executed.append({"name": name, "arguments": args, "result": result})
                    arg_text = ", ".join(f"{k}={v}" for k, v in args.items())
                    _emit(on_event, "tool", f"{name}({arg_text}) -> {result}")
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": str(result)}
                    )

                # Tool results are in. Remove tools so the model must write
                # the final answer instead of calling the tool again, and
                # tell it to actually report the result.
                tools = None
                messages.append({
                    "role": "user",
                    "content": (
                        "Using the tool result above, answer my original question "
                        "in one sentence. You must include the exact temperature "
                        f"and conditions from the result: {prompt}"
                    ),
                })

            return {"type": "error", "content": "Too many tool-call rounds."}
        except Exception as e:
            return {"type": "error", "content": f"Ollama API Error: {e}"}