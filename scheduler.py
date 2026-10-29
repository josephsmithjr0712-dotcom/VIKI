"""Task scheduler for VIKI: cron-like recurring jobs stored in JSON.

This is the core of the scheduled-tasks feature Joey asked for
(e.g. "check my email for urgent mail every 15 minutes"). It only
decides *when* a job is due and records what ran — the agent loop (or
a later wiring step) supplies the `executor` that actually does the
work, so the scheduler itself never needs a model, network, or clock
daemon to be testable.

Job kinds:
    interval: run every N minutes (``every_minutes`` >= 1)
    daily:    run once a day at a wall-clock time (``at`` == "HH:MM")

Storage: ``<folder>/tasks.json`` (folder from the constructor, the
``VIKI_SCHEDULER_DIR`` env var, or ``<repo>/scheduler`` by default).
Writes are atomic (temp file + replace) like the memory store, and a
corrupt file is moved aside instead of crashing the loop.

Anti-spam rules, by design:
  - A job runs at most once per interval / once per day, even if the
    scheduler is ticked repeatedly or VIKI restarts mid-day.
  - A missed run fires once on the next tick, never in a catch-up burst.
  - A failing job records its error and does not block other jobs.
"""
import json
import os
import re
import uuid
from datetime import datetime, time, timedelta
from pathlib import Path

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _atomic_write(path: Path, data: dict):
    """Write JSON to a temp file, then swap it in, so a crash can't corrupt the file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "task"


class Scheduler:
    def __init__(self, folder=None):
        # Folder order: explicit arg, VIKI_SCHEDULER_DIR, then the session
        # dir (so tests that point VIKI_SESSION_DIR at a temp dir stay
        # hermetic), then <repo>/scheduler.
        session_dir = os.getenv("VIKI_SESSION_DIR")
        base = (folder or os.getenv("VIKI_SCHEDULER_DIR")
                or (str(Path(session_dir) / "scheduler") if session_dir else None)
                or Path(__file__).resolve().parent / "scheduler")
        self.dir = Path(base)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.dir / "tasks.json"
        self.warning = None
        self._tasks = self._load()

    # --- persistence ------------------------------------------------------
    def _load(self):
        if not self.tasks_path.exists():
            return []
        try:
            data = json.loads(self.tasks_path.read_text(encoding="utf-8"))
            tasks = data.get("tasks", [])
            return [t for t in tasks if isinstance(t, dict) and isinstance(t.get("id"), str)]
        except (json.JSONDecodeError, OSError, AttributeError):
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup = self.tasks_path.with_name(f"tasks.corrupt-{stamp}.json")
            try:
                os.replace(self.tasks_path, backup)
                self.warning = f"tasks.json was unreadable; moved to {backup.name} and started fresh."
            except OSError:
                self.warning = "tasks.json was unreadable; starting fresh."
            return []

    def _save(self):
        _atomic_write(
            self.tasks_path,
            {"updated": datetime.now().isoformat(timespec="seconds"), "tasks": self._tasks},
        )

    # --- managing jobs ----------------------------------------------------
    def add(self, name, kind, action="", every_minutes=None, at=None):
        """Add a job. Returns the job dict. Raises ValueError on bad input."""
        name = " ".join(str(name).split())
        if not name:
            raise ValueError("task name is required")
        if kind == "interval":
            if not isinstance(every_minutes, (int, float)) or every_minutes < 1:
                raise ValueError("interval tasks need every_minutes >= 1")
            spec = {"every_minutes": int(every_minutes)}
        elif kind == "daily":
            m = _TIME_RE.match(str(at or ""))
            if not m:
                raise ValueError('daily tasks need at="HH:MM" (24-hour)')
            spec = {"at": f"{m.group(1)}:{m.group(2)}"}
        elif kind == "once":
            # One-shot job: runs a single time at a full date+time, then
            # turns itself off (see run_due). Strict formats only, so a
            # typo can't silently create a job that never fires.
            raw = str(at or "").strip()
            when = None
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
                try:
                    when = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if when is None:
                raise ValueError('once tasks need at="YYYY-MM-DD HH:MM"')
            spec = {"at": when.strftime("%Y-%m-%d %H:%M")}
        else:
            raise ValueError(f"unknown task kind: {kind!r} (expected 'interval', 'daily' or 'once')")
        job = {
            "id": f"{_slug(name)}-{uuid.uuid4().hex[:6]}",
            "name": name,
            "kind": kind,
            "action": " ".join(str(action or "").split()),
            "enabled": True,
            "created": datetime.now().isoformat(timespec="seconds"),
            "last_run": None,
            "run_count": 0,
            "last_result": None,
            "last_error": None,
            **spec,
        }
        self._tasks.append(job)
        self._save()
        return dict(job)

    def remove(self, job_id):
        """Remove a job by id. Returns True if one was removed."""
        before = len(self._tasks)
        self._tasks = [t for t in self._tasks if t["id"] != job_id]
        if len(self._tasks) != before:
            self._save()
            return True
        return False

    def set_enabled(self, job_id, enabled):
        """Enable/disable a job. Returns True if the job exists."""
        for t in self._tasks:
            if t["id"] == job_id:
                t["enabled"] = bool(enabled)
                self._save()
                return True
        return False

    def rename(self, job_id, name):
        """Rename a job. The id, schedule and run history are untouched,
        so renaming never loses a check's history or changes when it
        runs. Returns True if the job exists. Raises ValueError on a
        blank name."""
        name = " ".join(str(name).split())
        if not name:
            raise ValueError("task name is required")
        for t in self._tasks:
            if t["id"] == job_id:
                t["name"] = name
                self._save()
                return True
        return False

    def list_tasks(self):
        """All jobs, oldest first. Returns copies so callers can't mutate state."""
        return [dict(t) for t in self._tasks]

    def tasks_with_next_run(self, now=None):
        """list_tasks() plus a computed ``next_run`` ISO timestamp (or None).

        Presentation only: the stored jobs are untouched, and a disabled
        or already-fired job reports None, so no caller ever shows a
        next run for something that will never run again."""
        now = now or datetime.now()
        out = []
        for t in self.list_tasks():
            nxt = self.next_run(t, now)
            t["next_run"] = nxt.isoformat(timespec="seconds") if nxt else None
            out.append(t)
        return out

    # --- due logic --------------------------------------------------------
    @staticmethod
    def _parse_last_run(job):
        raw = job.get("last_run")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def is_due(self, job, now=None):
        """True if an enabled job should run at `now` (default: right now)."""
        if not job.get("enabled"):
            return False
        now = now or datetime.now()
        last = self._parse_last_run(job)
        if job["kind"] == "interval":
            if last is None:
                return True  # brand-new job runs once right away
            return now >= last + timedelta(minutes=job["every_minutes"])
        if job["kind"] == "daily":
            hh, mm = (int(x) for x in job["at"].split(":"))
            if last is not None and last.date() == now.date():
                return False  # already ran today: once per day, no repeats
            return now.time() >= time(hh, mm)
        if job["kind"] == "once":
            if last is not None:
                return False  # already fired: one-shot jobs never repeat
            try:
                when = datetime.strptime(job["at"], "%Y-%m-%d %H:%M")
            except (KeyError, ValueError):
                return False  # malformed job: never fire it by accident
            return now >= when
        return False

    def due(self, now=None):
        """Jobs due at `now`, in creation order."""
        return [dict(t) for t in self._tasks if self.is_due(t, now)]

    def next_run(self, job, now=None):
        """When the job will next run (None if disabled). Purely informational."""
        if not job.get("enabled"):
            return None
        now = now or datetime.now()
        last = self._parse_last_run(job)
        if job["kind"] == "interval":
            if last is None:
                return now
            return last + timedelta(minutes=job["every_minutes"])
        if job["kind"] == "once":
            if last is not None:
                return None  # already fired
            try:
                when = datetime.strptime(job["at"], "%Y-%m-%d %H:%M")
            except (KeyError, ValueError):
                return None
            return now if when <= now else when
        hh, mm = (int(x) for x in job["at"].split(":"))
        candidate = datetime.combine(now.date(), time(hh, mm))
        if last is not None and last.date() == now.date():
            return candidate + timedelta(days=1)
        if candidate <= now:
            return now  # overdue today: due now, not tomorrow
        return candidate

    # --- running ----------------------------------------------------------
    def run_due(self, executor, now=None):
        """Run every due job through ``executor(job)`` and record the outcome.

        The job is claimed (last_run stamped) *before* the executor runs,
        so a crash or restart can't re-fire the same run — the anti-spam
        guarantee. An executor failure is recorded on the job and never
        stops the other jobs. Returns [{id, name, result, error}].
        """
        now = now or datetime.now()
        outcomes = []
        for job in self._tasks:
            if not self.is_due(job, now):
                continue
            job["last_run"] = now.isoformat(timespec="seconds")
            job["run_count"] = job.get("run_count", 0) + 1
            job["last_result"] = None
            job["last_error"] = None
            try:
                job["last_result"] = str(executor(dict(job)))
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed silently
                job["last_error"] = f"{type(exc).__name__}: {exc}"
            if job["kind"] == "once":
                job["enabled"] = False  # one-shot: fired, so switch it off
            outcomes.append({
                "id": job["id"],
                "name": job["name"],
                "result": job["last_result"],
                "error": job["last_error"],
            })
        if outcomes:
            self._save()
        return outcomes

    def run_one(self, job_id, executor, now=None):
        """Run one job right now, whether or not it is due or enabled.

        A paused job stays paused after a manual run (this is a test
        fire, not a resume), and a one-shot job is fired by it — it
        switches off, exactly as a scheduled firing would, so it can
        never run twice. Returns the outcome dict, or None if the id
        is unknown. Executor failures are recorded, never raised.
        """
        now = now or datetime.now()
        for job in self._tasks:
            if job["id"] != job_id:
                continue
            job["last_run"] = now.isoformat(timespec="seconds")
            job["run_count"] = job.get("run_count", 0) + 1
            job["last_result"] = None
            job["last_error"] = None
            try:
                job["last_result"] = str(executor(dict(job)))
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed silently
                job["last_error"] = f"{type(exc).__name__}: {exc}"
            if job["kind"] == "once":
                job["enabled"] = False  # manual run fires the one-shot
            self._save()
            return {
                "id": job["id"],
                "name": job["name"],
                "result": job["last_result"],
                "error": job["last_error"],
            }
        return None
