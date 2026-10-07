import os
import sys
import time
import itertools
import threading
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()  # load .env BEFORE reading any settings below

from Agent import VIKI  # noqa: E402
from memory import SessionStore  # noqa: E402

# --- Settings -------------------------------------------------------------
MODEL = os.getenv("VIKI_MODEL", "llama3.2:latest")
ADDRESS = os.getenv("VIKI_ADDRESS", "http://localhost:11434")
# Working-memory window AND checkpoint interval (in messages; 2 per exchange).
# Every time the transcript grows by this much, the session is saved and the
# new messages are mined for long-term memory.
MAX_HISTORY = 20
AUTO_MEMORY = os.getenv("VIKI_AUTO_MEMORY", "on").lower() != "off"

# --- Styling --------------------------------------------------------------
if os.name == "nt":
    os.system("")  # enables ANSI colors in the Windows terminal

RESET, BOLD, DIM, ITALIC = "\033[0m", "\033[1m", "\033[2m", "\033[3m"
CYAN, GREEN, RED, YELLOW = "\033[36m", "\033[32m", "\033[31m", "\033[33m"

HELP_TEXT = f"""{BOLD}Commands{RESET}
  /think            show or hide VIKI's thinking (steps + model reasoning)
  /memory           list what VIKI remembers long-term
  /remember <fact>  add something to long-term memory
  /forget <n|all>   remove a memory by number, or all of them
  /save             save this session now
  /reset            save this session and start a fresh one
  /clear            clear the screen
  /help             show this list
  /exit             save and quit (Ctrl+C also works)
"""


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def banner(thinking_on, store):
    clear_screen()
    print(f"{CYAN}{BOLD}=== VIKI ==={RESET}  {DIM}online{RESET}")
    print(f"{DIM}model: {MODEL}  |  server: {ADDRESS}{RESET}")
    if not os.getenv("VIKI_MODEL"):
        print(f"{YELLOW}VIKI_MODEL not set in .env; using default.{RESET}")
    if not os.getenv("VIKI_ADDRESS"):
        print(f"{YELLOW}VIKI_ADDRESS not set in .env; using default.{RESET}")
    if store.warning:
        print(f"{YELLOW}{store.warning}{RESET}")
    state = "on" if thinking_on else "off"
    mem = len(store.facts)
    print(f"{DIM}thinking: {state}  |  memory: {mem} saved  |  /help for commands{RESET}\n")


class Spinner:
    """Animated indicator shown while waiting for the model."""

    def __init__(self, label="VIKI is thinking"):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for frame in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{DIM}{frame} {self.label}...{RESET}")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (len(self.label) + 8) + "\r")
        sys.stdout.flush()


class Renderer:
    """Turns agent events into live terminal output."""

    def __init__(self, show_thinking=True):
        self.show = show_thinking
        self.spinner = None
        self.section = None  # None | "thinking" | "answer"
        self.answer_started = False

    def _spin(self):
        self._stop_spin()
        self.spinner = Spinner()
        self.spinner.start()

    def _stop_spin(self):
        if self.spinner:
            self.spinner.stop()
            self.spinner = None

    def _end_section(self):
        if self.section:
            print(RESET)
            self.section = None

    def start(self):
        self._spin()

    def handle(self, kind, text):
        if kind == "thinking":
            if not self.show:
                return
            self._stop_spin()
            if self.section != "thinking":
                self._end_section()
                sys.stdout.write(f"{DIM}{ITALIC}thinking: ")
                self.section = "thinking"
            sys.stdout.write(text)
            sys.stdout.flush()
        elif kind == "token":
            self._stop_spin()
            if self.section != "answer":
                self._end_section()
                sys.stdout.write(f"{CYAN}{BOLD}VIKI >{RESET} ")
                self.section = "answer"
                self.answer_started = True
            sys.stdout.write(text)
            sys.stdout.flush()
        elif kind in ("status", "tool"):
            if kind == "status" and not self.show:
                return
            self._stop_spin()
            self._end_section()
            tag = "*" if kind == "status" else "[tool]"
            print(f"{DIM}  {tag} {text}{RESET}")
            self._spin()

    def finish(self):
        self._stop_spin()
        self._end_section()


class Session:
    """The full transcript of one conversation."""

    def __init__(self):
        now = datetime.now()
        self.id = now.strftime("%Y-%m-%d_%H%M%S")
        self.started = now.isoformat(timespec="seconds")
        self.messages = []   # full transcript, never trimmed
        self.archived = 0    # how many messages were already mined for memory

    def add(self, role, content, tools=None):
        msg = {
            "role": role,
            "content": content,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        if tools:
            msg["tools"] = tools
        self.messages.append(msg)

    def window(self, n=MAX_HISTORY):
        """The last n messages, in the shape the model expects."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages[-n:]]

    def pending_user_texts(self):
        return [m["content"] for m in self.messages[self.archived:] if m["role"] == "user"]


def checkpoint(viki, store, session):
    """Save the session file, then mine new messages for long-term memory."""
    if not session.messages:
        return
    path = store.save_session(session.id, session.started, viki.model, session.messages)
    print(f"{DIM}  [saved] {path.name}{RESET}")

    if not AUTO_MEMORY:
        session.archived = len(session.messages)
        return

    texts = session.pending_user_texts()
    if texts:
        spinner = Spinner("Updating long-term memory")
        spinner.start()
        try:
            facts = viki.extract_facts(texts, store.memory_texts())
        except KeyboardInterrupt:
            spinner.stop()
            print(f"{DIM}  (memory update skipped){RESET}")
            return
        spinner.stop()
        for fact in store.add_facts(facts, source=session.id, limit=5):
            print(f"{DIM}  [memory] + {fact}{RESET}")
    session.archived = len(session.messages)


def handle_command(user, viki, store, session, thinking_on):
    """Run a slash command. Returns (session, thinking_on, should_exit)."""
    name, _, arg = user.partition(" ")
    name, arg = name.lower(), arg.strip()

    if name in ("/exit", "/quit"):
        return session, thinking_on, True
    elif name == "/help":
        print(HELP_TEXT)
    elif name == "/think":
        thinking_on = not thinking_on
        print(f"{DIM}Thinking display {'on' if thinking_on else 'off'}.{RESET}\n")
    elif name == "/clear":
        banner(thinking_on, store)
    elif name == "/save":
        checkpoint(viki, store, session)
        print()
    elif name == "/reset":
        checkpoint(viki, store, session)
        session = Session()
        print(f"{DIM}New session started.{RESET}\n")
    elif name == "/memory":
        facts = store.facts
        if not facts:
            print(f"{DIM}Memory is empty.{RESET}\n")
        else:
            print(f"{BOLD}Long-term memory{RESET} {DIM}({len(facts)}){RESET}")
            for i, f in enumerate(facts, 1):
                print(f"  {i}. {f['text']}")
            print()
    elif name == "/remember":
        if not arg:
            print(f"{DIM}Usage: /remember <fact>{RESET}\n")
        elif store.add_facts([arg], source="manual"):
            print(f"{DIM}Remembered.{RESET}\n")
        else:
            print(f"{DIM}Not saved (already known, too long, or looks like a secret).{RESET}\n")
    elif name == "/forget":
        if arg.lower() == "all":
            store.clear()
            print(f"{DIM}Memory cleared.{RESET}\n")
        elif arg.isdigit():
            removed = store.remove(int(arg))
            msg = f"Forgot: {removed}" if removed else "No memory with that number."
            print(f"{DIM}{msg}{RESET}\n")
        else:
            print(f"{DIM}Usage: /forget <number> or /forget all{RESET}\n")
    else:
        print(f"{DIM}Unknown command. Type /help.{RESET}\n")
    return session, thinking_on, False


def main():
    viki = VIKI()  # create once, not on every prompt
    store = SessionStore()
    session = Session()
    thinking_on = False
    banner(thinking_on, store)

    try:
        while True:
            try:
                user = input(f"{GREEN}{BOLD}YOU  >{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user:
                continue
            if user.lower() in ("exit", "quit"):
                break
            if user.startswith("/"):
                session, thinking_on, done = handle_command(
                    user, viki, store, session, thinking_on
                )
                if done:
                    break
                continue

            renderer = Renderer(show_thinking=thinking_on)
            renderer.start()
            try:
                response = viki.viki_response(
                    user,
                    history=session.window(),
                    on_event=renderer.handle,
                    think=thinking_on,
                    memory=store.memory_texts(),
                )
            except KeyboardInterrupt:
                renderer.finish()
                print(f"{DIM}(cancelled){RESET}\n")
                continue
            renderer.finish()

            if response["type"] == "error":
                print(f"{RED}{BOLD}VIKI >{RESET} {RED}{response['content']}{RESET}\n")
                continue

            # If nothing was streamed (e.g. fallback to the raw tool result), print it
            if not renderer.answer_started and response["content"]:
                print(f"{CYAN}{BOLD}VIKI >{RESET} {response['content']}")
            print()

            session.add("user", user)
            session.add("assistant", response["content"], tools=response.get("tool_calls"))

            # Limit reached: save the session and mine it for long-term memory
            if len(session.messages) % MAX_HISTORY == 0:
                checkpoint(viki, store, session)
                print()
    finally:
        # Runs on exit, Ctrl+C, EOF, or a crash
        checkpoint(viki, store, session)

    print(f"{DIM}Goodbye!{RESET}")


if __name__ == "__main__":
    main()