"""User profile for VIKI: what she has learned about her user.

Structured where memory facts are free text: likes, dislikes, hobbies,
favorites (key -> value, e.g. "coffee": "black, no sugar") and general
notes. Stored as ``profile.json`` in the session folder, written
atomically. The agent reads it into its system prompt, so VIKI can
tailor answers without reciting the profile back.

Values are plain text, never secrets: the same secret filter as
memory.py applies, and entries are capped so the prompt block stays
small for small models.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

from memory import SECRET_RE

KINDS = ("like", "dislike", "hobby", "note")
MAX_ENTRIES = 30       # per list kind
MAX_VALUE_LEN = 120
MAX_FAVORITES = 30

# Phrases in a memory fact that signal a profile entry. Checked in order;
# the first matching kind wins. Kept deliberately narrow so ordinary facts
# ("User works at Walmart") never become profile entries.
# Dislike is checked BEFORE like on purpose: a negated preference such as
# "User doesn't like cilantro" contains the word "like", so checking like
# first misfiled every negated like as a like.
_PROFILE_PATTERNS = (
    ("dislike", re.compile(r"\b(dislikes?|hates?|can't stand|doesn'?t like|don'?t like|does not like|do not like|not a fan of)\b", re.I)),
    ("like", re.compile(r"\b(likes?|loves?|enjoys?|prefers?|is a fan of|into)\b", re.I)),
    ("hobby", re.compile(r"\b(hobby|hobbies|plays?|collects?)\b", re.I)),
)


def _clean(value):
    if not isinstance(value, str):
        return ""
    value = " ".join(value.split())
    return value if 2 <= len(value) <= MAX_VALUE_LEN and not SECRET_RE.search(value) else ""


class UserProfile:
    def __init__(self, folder=None):
        base = folder or os.getenv("VIKI_SESSION_DIR") or Path(__file__).resolve().parent / "session"
        self.dir = Path(base)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "profile.json"
        self._data = self._load()

    def _load(self):
        if not self.path.exists():
            return {"likes": [], "dislikes": [], "hobbies": [], "favorites": {}, "notes": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("profile is not an object")
        except (json.JSONDecodeError, OSError, ValueError):
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup = self.path.with_name(f"profile.corrupt-{stamp}.json")
            try:
                os.replace(self.path, backup)
            except OSError:
                pass
            return {"likes": [], "dislikes": [], "hobbies": [], "favorites": {}, "notes": []}
        out = {"likes": [], "dislikes": [], "hobbies": [], "favorites": {}, "notes": []}
        for kind in ("likes", "dislikes", "hobbies", "notes"):
            vals = data.get(kind)
            if isinstance(vals, list):
                out[kind] = [v for v in vals if isinstance(v, str)][:MAX_ENTRIES]
        favs = data.get("favorites")
        if isinstance(favs, dict):
            out["favorites"] = {str(k)[:40]: v for k, v in list(favs.items())[:MAX_FAVORITES]
                                if isinstance(v, str)}
        return out

    def _save(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"updated": datetime.now().isoformat(timespec="seconds"), **self._data},
                      f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def as_dict(self):
        """A copy of the profile, for API responses."""
        return {"likes": list(self._data["likes"]), "dislikes": list(self._data["dislikes"]),
                "hobbies": list(self._data["hobbies"]),
                "favorites": dict(self._data["favorites"]), "notes": list(self._data["notes"])}

    def add(self, kind, value, favorite_key=""):
        """Add one entry. kind: like|dislike|hobby|note|favorite.
        Returns True if something was actually stored."""
        kind = str(kind or "").strip().lower()
        if kind == "favorite":
            key = " ".join(str(favorite_key or "").split())[:40]
            value = _clean(value)
            if not key or not value:
                return False
            self._data["favorites"][key] = value
            if len(self._data["favorites"]) > MAX_FAVORITES:
                oldest = next(iter(self._data["favorites"]))
                del self._data["favorites"][oldest]
            self._save()
            return True
        mapping = {"like": "likes", "dislike": "dislikes", "hobby": "hobbies", "note": "notes"}
        lst = mapping.get(kind)
        if not lst:
            return False
        value = _clean(value)
        if not value or value in self._data[lst]:
            return False
        self._data[lst].append(value)
        del self._data[lst][:-MAX_ENTRIES]
        self._save()
        return True

    def remove(self, kind, value):
        """Remove one entry. Returns True if something was removed."""
        kind = str(kind or "").strip().lower()
        if kind == "favorite":
            if value in self._data["favorites"]:
                del self._data["favorites"][value]
                self._save()
                return True
            return False
        mapping = {"like": "likes", "dislike": "dislikes", "hobby": "hobbies", "note": "notes"}
        lst = mapping.get(kind)
        if not lst or value not in self._data[lst]:
            return False
        self._data[lst].remove(value)
        self._save()
        return True

    def text(self):
        """Prompt-ready block. Empty string when nothing is known yet."""
        d = self._data
        lines = []
        if d["likes"]:
            lines.append("Likes: " + "; ".join(d["likes"]))
        if d["dislikes"]:
            lines.append("Dislikes: " + "; ".join(d["dislikes"]))
        if d["hobbies"]:
            lines.append("Hobbies: " + "; ".join(d["hobbies"]))
        if d["favorites"]:
            lines.append("Favorites: " + "; ".join(f"{k}: {v}" for k, v in d["favorites"].items()))
        if d["notes"]:
            lines.append("Notes: " + "; ".join(d["notes"]))
        return "\n".join(lines)

    def import_from_facts(self, facts):
        """Scan free-text memory facts for stated preferences and import
        them as profile entries. Returns the entries actually added."""
        added = []
        for fact in facts or []:
            if not isinstance(fact, str):
                continue
            for kind, pattern in _PROFILE_PATTERNS:
                if pattern.search(fact):
                    if self.add(kind, fact):
                        added.append((kind, fact))
                    break
        return added
