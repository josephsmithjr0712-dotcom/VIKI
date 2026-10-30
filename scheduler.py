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

Finding 34 (judgment call, flagged for veto): all stored times are
naive local wall-clock times, deliberately. A daily job always fires
at its wall-clock slot, even across a DST transition (the slot is
recomputed from the current date each tick). An interval job measures
wall-clock minutes, so across a DST transition its next firing may
shift by the transition offset once — accepted as the cost of
wall-clock storage; absolute-time (UTC) storage was rejected because
it would make daily slots drift from the wall-clock time Joey set.

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

# Actions the scheduled-task executor knows how to run (see server.py
# _run_connection_job). change_action() and the task endpoints validate
# against this single list, so a typo can't silently create a job that
# never runs.
TASK_ACTIONS = ("connection-check", "urgent-email-check", "follow-up-nudges")


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
    @staticmethod
    def validate_spec(kind, every_minutes=None, at=None):
        """Return the stored spec dict for a kind, raising ValueError on
        bad input. Shared by add() and reschedule() (and the PATCH route,
        which validates before applying anything), so a schedule is
        never half-validated in two places."""
        if kind == "interval":
            # bool is an int subclass, and a non-integral float would
            # silently truncate (2.5 -> 2): reject both outright.
            if (isinstance(every_minutes, bool)
                    or not isinstance(every_minutes, (int, float))
                    or every_minutes < 1
                    or (isinstance(every_minutes, float) and not every_minutes.is_integer())):
                raise ValueError("interval tasks need every_minutes >= 1 (whole minutes)")
            return {"every_minutes": int(every_minutes)}
        if kind == "daily":
            m = _TIME_RE.match(str(at or ""))
            if not m:
                raise ValueError('daily tasks need at="HH:MM" (24-hour)')
            return {"at": f"{m.group(1)}:{m.group(2)}"}
        if kind == "once":
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
            return {"at": when.strftime("%Y-%m-%d %H:%M")}
        raise ValueError(f"unknown task kind: {kind!r} (expected 'interval', 'daily' or 'once')")

    def add(self, name, kind, action="", every_minutes=None, at=None):
        """Add a job. Returns the job dict. Raises ValueError on bad input.
        A blank action means connection-check (the historical default);
        any other action must be one of TASK_ACTIONS, so a typo can't
        create a job the executors will never run."""
        name = " ".join(str(name).split())
        if not name:
            raise ValueError("task name is required")
        action = " ".join(str(action or "").split())
        if action and action not in TASK_ACTIONS:
            raise ValueError(f"unknown task action: {action!r} "
                             f"(expected one of {', '.join(TASK_ACTIONS)})")
        spec = self.validate_spec(kind, every_minutes, at)
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

    def reschedule(self, job_id, kind, every_minutes=None, at=None):
        """Change a job's schedule in place. The id, name, action and
        run history are untouched, so changing when a check runs no
        longer means deleting it and recreating it. The new spec is
        validated before anything is touched, so a bad schedule leaves
        the old one exactly as it was. Returns True if the job exists.
        Raises ValueError on a bad kind or spec. Note: last_run is
        kept, so a daily job that already ran today and is rescheduled
        to another time today will not re-fire until tomorrow — except
        a fired one-shot: rescheduling it clears last_run and switches
        it back on, so the new slot actually fires instead of staying
        dead."""
        spec = self.validate_spec(kind, every_minutes, at)
        for t in self._tasks:
            if t["id"] == job_id:
                t.pop("every_minutes", None)
                t.pop("at", None)
                t["kind"] = kind
                t.update(spec)
                if kind == "once":
                    t["last_run"] = None
                    t["enabled"] = True
                self._save()
                return True
        return False

    def duplicate(self, job_id):
        """Copy a job under a new id. The schedule and action carry
        over exactly, but the copy starts with fresh history (no last
        run, zero runs) and the original is untouched — so trying a
        variation of a check no longer means recreating it by hand.
        Note: the copy inherits the original's enabled flag, so a copy
        of a disabled job — including a fired one-shot, which switches
        itself off — starts disabled (enabled=False) and must be
        resumed before it can run, even though its history is fresh.
        Returns the new job dict, or None if the id is unknown."""
        for t in self._tasks:
            if t["id"] != job_id:
                continue
            copy = self.add(f"{t['name']} (copy)", t["kind"],
                            action=t.get("action", ""),
                            every_minutes=t.get("every_minutes"),
                            at=t.get("at"))
            if not t.get("enabled", True):
                self.set_enabled(copy["id"], False)
                copy["enabled"] = False
            return copy
        return None

    def change_action(self, job_id, action):
        """Change what a job does, in place. The id, name, schedule and
        run history are untouched, so switching a check from e.g. the
        connection check to the urgent-email check no longer means
        deleting it and recreating it. Returns True if the job exists.
        Raises ValueError on an unknown action."""
        action = " ".join(str(action or "").split())
        if action not in TASK_ACTIONS:
            raise ValueError(f"unknown task action: {action!r} "
                             f"(expected one of {', '.join(TASK_ACTIONS)})")
        for t in self._tasks:
            if t["id"] == job_id:
                t["action"] = action
                self._save()
                return True
        return False

    def prune_finished_once(self):
        """Remove one-shot jobs that have already fired, and return how
        many were removed. A once job with a last_run can never run
        again (see is_due), so it only clutters the list; pending
        one-shots (no last_run yet) and every recurring job are kept,
        even paused ones, because those may still run."""
        before = len(self._tasks)
        self._tasks = [t for t in self._tasks
                       if not (t.get("kind") == "once"
                               and t.get("last_run"))]
        removed = before - len(self._tasks)
        if removed:
            self._save()
        return removed

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
            dt = datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is not None:
            # A timezone-stamped last_run (e.g. written by another tool)
            # must not crash naive comparisons: normalise to naive
            # local time. An unparseable value means "never ran".
            dt = dt.astimezone().replace(tzinfo=None)
        return dt

    def is_due(self, job, now=None):
        """True if an enabled job should run at `now` (default: right now)."""
        if not job.get("enabled"):
            return False
        now = now or datetime.now()
        last = self._parse_last_run(job)
        if job["kind"] == "interval":
            # A hand-edited tasks.json can hold a job with a missing or
            # non-numeric every_minutes; never let one bad job crash the
            # whole tick — it simply never fires.
            every = job.get("every_minutes")
            if (isinstance(every, bool) or not isinstance(every, (int, float))
                    or every < 1):
                return False
            if last is None:
                return True  # brand-new job runs once right away
            return now >= last + timedelta(minutes=every)
        if job["kind"] == "daily":
            try:
                hh, mm = (int(x) for x in job["at"].split(":"))
                slot = time(hh, mm)
            except (KeyError, ValueError, AttributeError, TypeError):
                return False  # malformed job: never fire it by accident
            if last is not None and last.date() == now.date():
                return False  # already ran today: once per day, no repeats
            return now.time() >= slot
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
            # Same malformed-job guard as is_due: a bad spec reports no
            # next run instead of crashing the task list.
            every = job.get("every_minutes")
            if (isinstance(every, bool) or not isinstance(every, (int, float))
                    or every < 1):
                return None
            if last is None:
                return now
            candidate = last + timedelta(minutes=every)
            # Overdue (e.g. the server was down past a slot): the job is
            # due now, so never report a next run in the past — daily
            # and once jobs already clamp the same way.
            return candidate if candidate > now else now
        if job["kind"] == "once":
            if last is not None:
                return None  # already fired
            try:
                when = datetime.strptime(job["at"], "%Y-%m-%d %H:%M")
            except (KeyError, ValueError):
                return None
            return now if when <= now else when
        try:
            hh, mm = (int(x) for x in job["at"].split(":"))
            slot = time(hh, mm)
        except (KeyError, ValueError, AttributeError, TypeError):
            return None  # malformed job: no next run, no crash
        candidate = datetime.combine(now.date(), slot)
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
            # Persist the claim immediately: if the executor crashes the
            # process mid-tick, the run is still recorded and won't
            # re-fire on the next tick.
            self._save()
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
            self._save()
        return outcomes

    def run_one(self, job_id, executor, now=None):
        """Run one job right now, whether or not it is due or enabled.

        A paused job stays paused after a manual run (this is a test
        fire, not a resume), and a one-shot job is fired by it — it
        switches off, exactly as a scheduled firing would, so it can
        never run twice. Returns the outcome dict, or None if the id
        is unknown. Executor failures are recorded, never raised.

        Finding 14 (judgment call, flagged for veto): what a manual
        run does to last_run depends on kind. A daily job's slot is
        wall-clock time, so a manual test fire does NOT stamp
        last_run — the job still fires at its slot later (otherwise
        testing the morning brief at 5pm would silently eat that
        day's scheduled run). An interval job's schedule is "every N
        minutes since the last run", so a manual run DOES reset that
        clock. A one-shot job is consumed by any run, manual or not.
        """
        now = now or datetime.now()
        for job in self._tasks:
            if job["id"] != job_id:
                continue
            if job["kind"] != "daily":
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
