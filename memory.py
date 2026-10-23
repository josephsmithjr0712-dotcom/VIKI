"""Disk storage for VIKI: session transcripts and long-term memory.

Layout (inside the session folder):
    session/session_2026-10-06_153012.json   one file per session (full transcript)
    session/memory.json                      durable facts used in future sessions
"""
import os
import re
import json
from datetime import datetime
from pathlib import Path

MAX_FACTS = 40          # keeps the memory block small enough for small models
MAX_FACT_LEN = 200
MAX_NEW_PER_EXTRACTION = 5

# Facts that look like secrets are never stored (the file is plain text).
SECRET_RE = re.compile(
    r"(password|passwd|api[ _-]?key|secret|token|ssn|social security|credit card|\b\d{12,}\b)",
    re.IGNORECASE,
)


def _atomic_write(path: Path, data: dict):
    """Write JSON to a temp file, then swap it in, so a crash can't corrupt the file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


class SessionStore:
    def __init__(self, folder=None):
        base = folder or os.getenv("VIKI_SESSION_DIR") or Path(__file__).resolve().parent / "session"
        self.dir = Path(base)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.dir / "memory.json"
        self.warning = None
        self._facts = self._load()

    # --- long-term memory -------------------------------------------------
    def _load(self):
        if not self.memory_path.exists():
            return []
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            return [
                f for f in data.get("facts", [])
                if isinstance(f, dict) and isinstance(f.get("text"), str)
            ]
        except (json.JSONDecodeError, OSError, AttributeError):
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup = self.memory_path.with_name(f"memory.corrupt-{stamp}.json")
            try:
                os.replace(self.memory_path, backup)
                self.warning = f"memory.json was unreadable; moved to {backup.name} and started fresh."
            except OSError:
                self.warning = "memory.json was unreadable; starting fresh."
            return []

    def _save_memory(self):
        _atomic_write(
            self.memory_path,
            {"updated": datetime.now().isoformat(timespec="seconds"), "facts": self._facts},
        )

    @property
    def facts(self):
        return list(self._facts)

    def memory_texts(self):
        return [f["text"] for f in self._facts]

    def search(self, query, limit=8):
        """Relevance-ranked recall: returns up to `limit` fact texts most
        relevant to the query (word-overlap scoring). Used by the agent loop."""
        if not query or not self._facts:
            return []
        qwords = set(_norm(query).split())
        if not qwords:
            return []
        scored = []
        for f in self._facts:
            text = f.get("text", "")
            fwords = set(_norm(text).split())
            overlap = len(qwords & fwords)
            if overlap:
                # slight preference for shorter, more specific facts
                scored.append((overlap / (1 + len(fwords) ** 0.5), text))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [t for _, t in scored[:limit]]

    def add_facts(self, facts, source="manual", limit=None):
        """Add new facts (deduplicated, filtered). Returns the ones actually added."""
        added = []
        seen = {_norm(t) for t in self.memory_texts()}
        for raw in facts:
            if not isinstance(raw, str):
                continue
            text = " ".join(raw.split())
            if not (3 <= len(text) <= MAX_FACT_LEN) or SECRET_RE.search(text):
                continue
            key = _norm(text)
            if not key or key in seen:
                continue
            seen.add(key)
            self._facts.append(
                {"text": text, "added": datetime.now().strftime("%Y-%m-%d"), "source": source}
            )
            added.append(text)
            if limit and len(added) >= limit:
                break
        if len(self._facts) > MAX_FACTS:
            del self._facts[:-MAX_FACTS]  # drop the oldest
        if added:
            self._save_memory()
        return added

    def remove(self, index: int):
        """Remove fact by 1-based index. Returns its text, or None if out of range."""
        if not 1 <= index <= len(self._facts):
            return None
        text = self._facts.pop(index - 1)["text"]
        self._save_memory()
        return text

    def clear(self):
        self._facts = []
        self._save_memory()

    # --- sessions ---------------------------------------------------------
    def list_sessions(self, limit=50):
        """List saved sessions, newest first. Returns [{id, started, saved, model, message_count}]."""
        files = sorted(self.dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for p in files[:limit]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                msgs = data.get("messages", [])
                out.append({
                    "id": data.get("id", p.stem.replace("session_", "")),
                    "started": data.get("started", ""),
                    "saved": data.get("saved", ""),
                    "model": data.get("model", ""),
                    "message_count": len(msgs),
                    # first user message as preview
                    "preview": next((m.get("content", "")[:80] for m in msgs if m.get("role") == "user"), ""),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def load_session(self, session_id):
        """Load a saved session's full transcript. Returns dict or None."""
        # sanitize: only allow safe chars in session_id to prevent path traversal
        import re as _re
        if not _re.match(r'^[A-Za-z0-9_\-]+$', session_id):
            return None
        p = self.dir / f"session_{session_id}.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_session(self, session_id, started, model, messages) -> Path:
        path = self.dir / f"session_{session_id}.json"
        _atomic_write(
            path,
            {
                "id": session_id,
                "started": started,
                "saved": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "messages": messages,
            },
        )
        return path