"""Chat threads for VIKI: one main chat plus any number of side chats.

Each thread is a JSON file under the session folder (``chat_<id>.json``),
written atomically like the memory store, so a restart or crash can't
corrupt it. A thread keeps its own message list, so side chats never
leak into the main chat's context (and vice versa).

Thread ids are sanitized the same way session ids are in memory.py:
only letters, digits, underscore and hyphen survive.
"""
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _atomic_write(path: Path, data: dict):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _clean_id(raw):
    """Return a safe thread id, or None if the input is unusable."""
    if not isinstance(raw, str) or not raw:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", raw)
    return cleaned or None


def context_history(messages, keep=20):
    """Build the history an agent turn actually sees (v0.3.8).

    Short threads pass through unchanged (role/content only). For a
    thread longer than ``keep`` messages, the older messages are replaced
    by one deterministic summary entry — a count plus the first line of
    each older user message (deduped, capped) — followed by the most
    recent ``keep`` messages verbatim. Nothing is deleted from storage;
    this only compresses what goes into the model's context, so a long
    chat keeps its earlier topics instead of silently dropping them
    past the 20-message window. Pure function, no model call, so the
    summary is stable and testable.
    """
    try:
        keep = int(keep)
    except (TypeError, ValueError):
        keep = 20
    if keep < 1:
        keep = 1
    msgs = [{"role": m.get("role", ""), "content": m.get("content", "")}
            for m in (messages or []) if isinstance(m, dict)]
    if len(msgs) <= keep:
        return msgs
    older, recent = msgs[:-keep], msgs[-keep:]
    topics = []
    for m in older:
        if m["role"] != "user":
            continue
        first = " ".join(str(m["content"]).splitlines()[:1]).strip()
        first = " ".join(first.split())[:80]
        if first and first not in topics:
            topics.append(first)
        if len(topics) >= 8:
            break
    if topics:
        summary = (f"[Summary of the {len(older)} earlier messages in this "
                   f"chat: {'; '.join(topics)}]")
    else:
        summary = f"[Summary of the {len(older)} earlier messages in this chat.]"
    return [{"role": "system", "content": summary}] + recent


class ChatStore:
    def __init__(self, folder=None):
        base = folder or os.getenv("VIKI_SESSION_DIR") or Path(__file__).resolve().parent / "session"
        self.dir = Path(base)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id):
        return self.dir / f"chat_{thread_id}.json"

    def create(self, title="", kind="side"):
        """Create a thread. kind is 'main' or 'side'. Returns the thread dict."""
        thread_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat(timespec="seconds")
        thread = {
            "id": thread_id,
            "title": " ".join(str(title or "").split()) or ("Main chat" if kind == "main" else "New chat"),
            "kind": "main" if kind == "main" else "side",
            "created": now,
            "updated": now,
            "messages": [],
        }
        _atomic_write(self._path(thread_id), thread)
        return dict(thread)

    def get(self, thread_id):
        """Load a thread, or None if it doesn't exist / the id is unsafe."""
        tid = _clean_id(thread_id)
        if not tid:
            return None
        p = self._path(tid)
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict) or data.get("id") != tid:
            return None
        data.setdefault("messages", [])
        return data

    def save(self, thread):
        """Persist a thread dict. Auto-titles untitled threads from their
        first user message. Returns the saved thread."""
        tid = _clean_id(thread.get("id"))
        if not tid:
            raise ValueError("thread needs a safe id")
        thread = dict(thread)
        thread["id"] = tid
        thread["updated"] = datetime.now().isoformat(timespec="seconds")
        msgs = thread.get("messages") or []
        if thread.get("title") in ("", "New chat") and msgs:
            first = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            if first.strip():
                thread["title"] = " ".join(first.split())[:60]
        _atomic_write(self._path(tid), thread)
        return thread

    def list_threads(self, limit=50):
        """All threads, newest first. Returns summaries (no message bodies)."""
        files = sorted(self.dir.glob("chat_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for p in files[:limit]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                msgs = data.get("messages", [])
            except (json.JSONDecodeError, OSError):
                continue
            out.append({
                "id": data.get("id", p.stem.replace("chat_", "")),
                "title": data.get("title", "New chat"),
                "kind": data.get("kind", "side"),
                "created": data.get("created", ""),
                "updated": data.get("updated", ""),
                "message_count": len(msgs),
                "preview": next((m.get("content", "")[:80] for m in msgs if m.get("role") == "user"), ""),
            })
        return out

    def search(self, query, limit=20):
        """Case-insensitive substring search across all threads.
        Returns matches with a snippet around the hit; empty query
        returns nothing."""
        q = (query or "").strip().lower()
        if not q:
            return []
        out = []
        for summary in self.list_threads(limit=200):
            thread = self.get(summary["id"])
            if not thread:
                continue
            for i, m in enumerate(thread.get("messages") or []):
                content = str(m.get("content") or "")
                low = content.lower()
                at = low.find(q)
                if at < 0:
                    continue
                start = max(0, at - 40)
                out.append({
                    "chat_id": thread["id"],
                    "title": thread.get("title", "New chat"),
                    "kind": thread.get("kind", "side"),
                    "index": i,
                    "role": m.get("role", ""),
                    "snippet": content[start:at + len(q) + 40],
                })
                if len(out) >= limit:
                    return out
        return out

    def rename(self, thread_id, title):
        """Rename a thread. Returns the thread, or None if it doesn't exist."""
        thread = self.get(thread_id)
        if not thread:
            return None
        title = " ".join(str(title or "").split())
        if title:
            thread["title"] = title[:80]
            return self.save(thread)
        return thread

    def delete(self, thread_id, protect_main=False):
        """Delete a thread file. Returns True if one was removed.
        With protect_main, the main chat refuses to go away."""
        tid = _clean_id(thread_id)
        if not tid:
            return False
        if protect_main:
            thread = self.get(tid)
            if thread and thread.get("kind") == "main":
                return False
        p = self._path(tid)
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                return False
            return True
        return False

    def ensure_main(self):
        """Return the main chat thread, creating it if none exists."""
        for t in self.list_threads(limit=200):
            if t.get("kind") == "main":
                full = self.get(t["id"])
                if full:
                    return full
        return self.create(title="Main chat", kind="main")
