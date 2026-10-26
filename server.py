"""VIKI web server. Run: python server.py  then open http://localhost:7860"""
import hmac
import json, os, queue, secrets, threading, time, uuid
from datetime import datetime, timedelta
from pathlib import Path
try:
    from dotenv import load_dotenv, set_key
except ImportError:
    # Graceful degradation (v0.2.22): server.py must stay importable where
    # python-dotenv isn't installed. load_dotenv is a no-op; set_key raising
    # is already swallowed by _get_or_create, so generated secrets live in
    # os.environ for the session instead of being persisted to .env.
    def load_dotenv(*_args, **_kwargs):
        return False

    def set_key(*_args, **_kwargs):
        raise RuntimeError("python-dotenv is not installed; cannot persist .env values")
load_dotenv()
from flask import Flask, Response, jsonify, request, send_from_directory, session

from agent import VIKI
from chats import ChatStore, context_history
from connectors.digest import connection_check
from memory import SessionStore
from profile import UserProfile
from scheduler import Scheduler
from tools import Toolbox, WORKSPACE, refresh_ha_entities, ha_entities_text
from uploads import stage_files, discard, video_frames, read_text, UploadError, MAX_BYTES, MAX_FILES

ENV_FILE = Path(__file__).resolve().parent / ".env"

app = Flask(__name__, static_folder="static")

# --- Auth setup: secret key + web password, persisted in .env ---
def _get_or_create(key, gen):
    """Returns (value, was_created). Persists new values to .env."""
    v = os.getenv(key)
    if not v:
        v = gen()
        try:
            set_key(str(ENV_FILE), key, v)
        except Exception:
            pass
        os.environ[key] = v
        return v, True
    return v, False

SECRET, _ = _get_or_create("VIKI_SECRET_KEY", lambda: secrets.token_hex(32))
WEB_PASSWORD, _pw_created = _get_or_create("VIKI_WEB_PASSWORD", lambda: secrets.token_urlsafe(16))
if _pw_created:
    print("=" * 60)
    print(f"  VIKI web password (saved to .env as VIKI_WEB_PASSWORD):\n  {WEB_PASSWORD}")
    print("=" * 60)

app.secret_key = SECRET
app.permanent_session_lifetime = timedelta(days=30)
# Cap total upload body: 3 files x 10 MB + headroom. Flask rejects larger
# bodies with 413 before we ever read them.
app.config["MAX_CONTENT_LENGTH"] = (MAX_BYTES + 1024 * 1024) * MAX_FILES
# Staged attachments: id -> record. Single-use: each ID is consumed by exactly
# one chat turn, then the files are deleted. Never served back to clients.
staged_uploads = {}

# Simple per-IP rate limiting on login attempts: 5 failures -> 5 min lockout.
_login_failures = {}

def _rate_limited(ip):
    now = time.time()
    fails = [t for t in _login_failures.get(ip, []) if now - t < 300]
    _login_failures[ip] = fails
    return len(fails) >= 5

def _record_failure(ip):
    _login_failures.setdefault(ip, []).append(time.time())

def _authed():
    return bool(session.get("viki_auth"))

@app.before_request
def _require_auth():
    # Public: login page + its POST, and static assets (needed to render login).
    if request.path == "/login" or request.path.startswith("/static/"):
        return None
    if not _authed():
        if request.path.startswith("/api/"):
            return jsonify(error="login required"), 401
        return send_from_directory("static", "login.html")
    return None

@app.get("/login")
def login_page():
    if _authed():
        return send_from_directory("static", "index.html")
    return send_from_directory("static", "login.html")

@app.post("/login")
def login():
    ip = request.remote_addr or "unknown"
    if _rate_limited(ip):
        return jsonify(error="too many attempts, try again in 5 minutes"), 429
    body = request.get_json(silent=True) or {}
    pw = body.get("password", "")
    if pw and hmac.compare_digest(pw, WEB_PASSWORD):
        session["viki_auth"] = True
        session.permanent = bool(body.get("remember"))
        _login_failures.pop(ip, None)
        return jsonify(ok=True)
    _record_failure(ip)
    return jsonify(error="wrong password"), 401

@app.post("/logout")
def logout():
    session.clear()
    return jsonify(ok=True)

store = SessionStore()
chat_store = ChatStore()
user_profile = UserProfile()
task_scheduler = Scheduler()
# Grow the profile from what VIKI already remembers (no model call: just
# pattern-matching over stored facts). Runs once at startup.
try:
    user_profile.import_from_facts(store.memory_texts())
except Exception:
    pass
viki = VIKI(Toolbox(store))
# Pre-load the LLM at startup so the first request doesn't pay load time.
# Runs in a thread so a slow/unreachable Ollama doesn't block server boot.
threading.Thread(target=viki.warmup, daemon=True).start()
# Pull the Home Assistant entity list so VIKI knows what devices exist.
threading.Thread(target=refresh_ha_entities, daemon=True).start()
AUTO_MEMORY = os.getenv("VIKI_AUTO_MEMORY", "on").lower() != "off"
sessions, approvals, cancels = {}, {}, {}
# Live chat threads (main + side chats), keyed by thread id. Persisted via
# ChatStore on every checkpoint; the dict is just a working cache.
threads = {}


def _brief_path():
    return task_scheduler.dir / "brief.json"


def _load_brief():
    try:
        data = json.loads(_brief_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"latest": None, "updated": None, "history": []}


def _save_brief(text):
    """Persist the latest connection-check brief (history capped at 7)."""
    data = _load_brief()
    now = datetime.now().isoformat(timespec="seconds")
    data["latest"] = text
    data["updated"] = now
    history = data.get("history") or []
    history.append({"at": now, "brief": text})
    data["history"] = history[-7:]
    tmp = _brief_path().with_name("brief.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _brief_path())
    return data


def _run_connection_job(job):
    """Executor for scheduled jobs. 'connection-check' and
    'urgent-email-check' run on their own today; anything else is
    recorded, not silently dropped."""
    if job.get("action") == "urgent-email-check":
        # Urgent-email smart announcements (v0.3.10): yesterday's mail
        # is not worth waking the house for, so only the last day's
        # inbox is scanned. Speaking is presence-gated and deduped by
        # announce_urgent_emails; a missing Google connection or a
        # failed search is recorded, never raised into the tick loop.
        try:
            from connectors import google as google_mod

            if not google_mod.token_exists():
                return "urgent-email-check: Google not connected"
            messages = google_mod.gmail_search_structured(
                "in:inbox newer_than:1d", max_results=20)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            return f"urgent-email-check failed: {type(exc).__name__}"
        try:
            from smart_announce import announce_urgent_emails

            return "urgent-email-check: " + announce_urgent_emails(messages)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            return f"urgent-email-check failed: {type(exc).__name__}"
    if job.get("action") == "connection-check":
        brief = connection_check()
        _save_brief(brief)
        # Morning-brief announcements (v0.3.7): a scheduled brief is
        # spoken on the Home Mini when Joey is home (presence-gated,
        # never repeated — see smart_announce.announce_brief). Speaking
        # is best-effort: it never blocks saving the brief, and the
        # outcome is recorded on the job for the scheduler's record.
        try:
            from smart_announce import announce_brief

            ok, reason = announce_brief(brief)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            return f"connection brief saved; brief not announced: {type(exc).__name__}"
        if ok:
            return "connection brief saved; brief announced"
        return f"connection brief saved; brief not announced: {reason}"
    return f"No built-in action for {job.get('action')!r}."


def _scheduler_tick():
    """Background loop: every 60s, run due jobs. Never dies on an error —
    a failed tick just means the next minute's tick tries again."""
    while True:
        time.sleep(60)
        try:
            task_scheduler.run_due(_run_connection_job)
        except Exception:
            pass


threading.Thread(target=_scheduler_tick, daemon=True).start()


def get_thread(cid):
    """Live thread by id, loading from disk if needed. None if unknown."""
    if cid in threads:
        return threads[cid]
    t = chat_store.get(cid)
    if t:
        t.setdefault("archived", 0)
        threads[cid] = t
    return threads.get(cid)


def checkpoint_thread(t):
    """Persist a thread + extract durable facts, mirroring checkpoint()."""
    chat_store.save(t)
    if AUTO_MEMORY:
        texts = [m["content"] for m in t["messages"][t.get("archived", 0):] if m["role"] == "user"]
        if len(texts) >= 3:
            store.add_facts(viki.extract_facts(texts, store.memory_texts()), source=t["id"], limit=5)
            t["archived"] = len(t["messages"])


def get_session(sid):
    if sid not in sessions:
        now = datetime.now()
        sessions[sid] = {"id": now.strftime("%Y-%m-%d_%H%M%S"), "started": now.isoformat(timespec="seconds"),
                         "messages": [], "archived": 0}
    return sessions[sid]


def checkpoint(s):
    store.save_session(s["id"], s["started"], viki.model, s["messages"])
    if AUTO_MEMORY:
        texts = [m["content"] for m in s["messages"][s["archived"]:] if m["role"] == "user"]
        if len(texts) >= 3:
            store.add_facts(viki.extract_facts(texts, store.memory_texts()), source=s["id"], limit=5)
            s["archived"] = len(s["messages"])


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/state")
def state():
    return jsonify(model=viki.model, tools=viki.toolbox.enabled, memory=store.memory_texts(),
                   files=len(list(WORKSPACE.rglob("*"))), warning=store.warning,
                   profile=user_profile.text())


@app.get("/api/sessions")
def sessions_list():
    """List past conversation sessions, newest first."""
    return jsonify(sessions=store.list_sessions())


@app.get("/api/sessions/<sid>")
def session_load(sid):
    """Load a past session's full transcript."""
    data = store.load_session(sid)
    if not data:
        return jsonify(error="session not found"), 404
    return jsonify(session=data)


def chat_thread(body):
    """SSE chat turn inside a main/side thread (v0.3.0). Mirrors the
    legacy session handler, but history and storage belong to the thread,
    and the user profile is passed into the agent run."""
    cid = body.get("chat")
    t = get_thread(cid)
    if not t:
        return jsonify(error="chat not found"), 404
    prompt = (body.get("message") or "").strip()
    think = bool(body.get("think"))
    aids = body.get("attachments") or []
    if not prompt and not aids:
        return jsonify(error="empty message"), 400
    q, cancel = queue.Queue(), threading.Event()
    cancels[cid] = cancel
    history = context_history(t["messages"])

    images, notes, consumed = [], [], []
    for aid in aids[:MAX_FILES]:
        rec = staged_uploads.pop(aid, None)
        if not rec:
            continue
        consumed.append(rec)
        if rec["kind"] == "image":
            images.append(Path(rec["path"]).read_bytes())
        elif rec["kind"] == "video":
            frames = video_frames(rec["path"])
            images.extend(frames)
            extra = "" if frames else " (frame extraction needs ffmpeg on the server)"
            notes.append(f"[attached video: {rec['name']} ({rec['size'] // 1024} KB){extra}]")
        else:
            notes.append(f"[attached file: {rec['name']}]\n```\n{read_text(rec['path'])}\n```")
    if notes:
        prompt = "\n\n".join(notes + ([prompt] if prompt else []))

    turn_maps = []

    def emit(kind, payload):
        if kind == "map":
            turn_maps.append(payload)
        q.put({"kind": kind, "data": payload})

    def approve(cid_, name, args):
        ev = threading.Event(); approvals[cid_] = {"event": ev, "ok": False}
        emit("approval", {"id": cid_, "name": name, "args": args})
        ev.wait(timeout=180)
        return approvals.pop(cid_, {}).get("ok", False)

    def work():
        try:
            res = viki.run(prompt, history, store.search, emit, approve, think, cancel,
                           images=images or None, profile=user_profile.text())
            if res["type"] != "error":
                now = datetime.now().isoformat(timespec="seconds")
                assistant_msg = {"role": "assistant", "content": res["content"], "time": now,
                                 "tools": res.get("tools")}
                if turn_maps:
                    assistant_msg["maps"] = turn_maps
                t["messages"] += [{"role": "user", "content": prompt, "time": now}, assistant_msg]
            emit("done", res)
        finally:
            discard(consumed)
        q.put(None)
        try: checkpoint_thread(t)
        except Exception: pass

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (item := q.get()) is not None:
            yield f"data: {json.dumps(item)}\n\n"
    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/chats")
def chats_list():
    """List threads (main + side chats), newest first. Creates the main
    chat on first use so the UI always has somewhere to land."""
    chat_store.ensure_main()
    return jsonify(chats=chat_store.list_threads())


@app.post("/api/chats")
def chats_create():
    """Create a new side chat. Returns the thread."""
    b = request.get_json(silent=True) or {}
    t = chat_store.create(title=b.get("title", ""), kind="side")
    t["archived"] = 0
    threads[t["id"]] = t
    return jsonify(chat=t)


@app.get("/api/chats/<cid>")
def chats_get(cid):
    t = get_thread(cid)
    if not t:
        return jsonify(error="chat not found"), 404
    return jsonify(chat=t)


@app.delete("/api/chats/<cid>")
def chats_delete(cid):
    if not chat_store.delete(cid, protect_main=True):
        t = chat_store.get(cid)
        if t and t.get("kind") == "main":
            return jsonify(error="the main chat can't be deleted"), 400
        return jsonify(error="chat not found"), 404
    threads.pop(cid, None)
    return jsonify(ok=True)


@app.get("/api/search")
def search_chats():
    """Search all chat threads. Multi-word queries match messages
    containing all the words, in any order. Query param q."""
    q = request.args.get("q", "")
    return jsonify(results=chat_store.search(q))


@app.post("/api/chats/<cid>/rename")
def chats_rename(cid):
    b = request.get_json(silent=True) or {}
    t = chat_store.rename(cid, b.get("title", ""))
    if not t:
        return jsonify(error="chat not found"), 404
    if cid in threads:
        threads[cid]["title"] = t["title"]
    return jsonify(chat=t)


@app.route("/api/profile", methods=["GET", "POST", "DELETE"])
def profile_api():
    b = request.get_json(silent=True) or {}
    if request.method == "POST":
        user_profile.add(b.get("kind", ""), b.get("value", ""), b.get("favorite_key", ""))
    if request.method == "DELETE":
        user_profile.remove(b.get("kind", ""), b.get("value", ""))
    return jsonify(profile=user_profile.as_dict())


@app.get("/api/tasks")
def tasks_list():
    return jsonify(tasks=task_scheduler.list_tasks())


@app.post("/api/tasks")
def tasks_create():
    """Create a recurring job from the UI. The action must be one of
    the executor's built-in actions ('connection-check' by default,
    or 'urgent-email-check'); anything else is rejected so a typo
    can't silently create a job that never runs."""
    b = request.get_json(silent=True) or {}
    action = b.get("action") or "connection-check"
    if action not in ("connection-check", "urgent-email-check"):
        return jsonify(error=f"unknown task action: {action!r}"), 400
    try:
        minutes = int(b["every_minutes"]) if b.get("every_minutes") not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify(error="every_minutes must be a whole number"), 400
    try:
        job = task_scheduler.add(b.get("name", "connection check"), b.get("kind", "daily"),
                                 action=action, every_minutes=minutes, at=b.get("at"))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(task=job)


@app.delete("/api/tasks/<jid>")
def tasks_delete(jid):
    if not task_scheduler.remove(jid):
        return jsonify(error="task not found"), 404
    return jsonify(ok=True)


@app.get("/api/brief")
def brief_get():
    return jsonify(brief=_load_brief())


@app.post("/api/brief/refresh")
def brief_refresh():
    """Run a connection check right now and save it as the latest brief."""
    data = _save_brief(connection_check())
    return jsonify(brief=data)


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True, silent=True) or {}
    if body.get("chat"):
        return chat_thread(body)
    sid = body.get("session", "default")
    prompt = (body.get("message") or "").strip()
    think = bool(body.get("think"))
    aids = body.get("attachments") or []
    if not prompt and not aids:
        return jsonify(error="empty message"), 400
    s, q, cancel = get_session(sid), queue.Queue(), threading.Event()
    cancels[sid] = cancel
    history = context_history(s["messages"])

    # Consume staged attachments (single-use IDs; unknown IDs are ignored).
    # Images go to the model; video becomes frames; text is inlined.
    images, notes, consumed = [], [], []
    for aid in aids[:MAX_FILES]:
        rec = staged_uploads.pop(aid, None)
        if not rec:
            continue
        consumed.append(rec)
        if rec["kind"] == "image":
            images.append(Path(rec["path"]).read_bytes())
        elif rec["kind"] == "video":
            frames = video_frames(rec["path"])
            images.extend(frames)
            extra = "" if frames else " (frame extraction needs ffmpeg on the server)"
            notes.append(f"[attached video: {rec['name']} ({rec['size'] // 1024} KB){extra}]")
        else:
            notes.append(f"[attached file: {rec['name']}]\n```\n{read_text(rec['path'])}\n```")
    if notes:
        prompt = "\n\n".join(notes + ([prompt] if prompt else []))

    # Map cards render live from the "map" SSE event, but a reopened session
    # only has its stored messages — so keep this turn's map payloads and
    # attach them to the stored assistant message (v0.2.33), letting the
    # History tab re-render the cards instead of text alone.
    turn_maps = []

    def emit(kind, payload):
        if kind == "map":
            turn_maps.append(payload)
        q.put({"kind": kind, "data": payload})

    def approve(cid, name, args):
        ev = threading.Event(); approvals[cid] = {"event": ev, "ok": False}
        emit("approval", {"id": cid, "name": name, "args": args})
        ev.wait(timeout=180)
        return approvals.pop(cid, {}).get("ok", False)

    def work():
        try:
            res = viki.run(prompt, history, store.search, emit, approve, think, cancel,
                           images=images or None)
            if res["type"] != "error":
                now = datetime.now().isoformat(timespec="seconds")
                assistant_msg = {"role": "assistant", "content": res["content"], "time": now,
                                 "tools": res.get("tools")}
                if turn_maps:
                    assistant_msg["maps"] = turn_maps
                s["messages"] += [{"role": "user", "content": prompt, "time": now}, assistant_msg]
            emit("done", res)
        finally:
            # Staged files are single-use: delete them whether the turn
            # succeeded or not. Nothing uploaded lingers on disk.
            discard(consumed)
        q.put(None)
        try: checkpoint(s)
        except Exception: pass

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (item := q.get()) is not None:
            yield f"data: {json.dumps(item)}\n\n"
    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/upload")
def upload():
    """Stage attachments for the next chat message.

    Files are validated by magic bytes (never by extension or claimed MIME),
    stored under throwaway UUID names in the OS temp dir, and returned as
    single-use IDs. SVG/HTML, executables, and anything unrecognized are
    rejected. Nothing staged is ever served back or executed.
    """
    try:
        items = [(f.filename, f.read(MAX_BYTES + 1)) for f in request.files.getlist("files")]
        staged = stage_files(items)
    except UploadError as e:
        return jsonify(error=str(e)), 400
    for rec in staged:
        staged_uploads[rec["id"]] = rec
    return jsonify(files=[{k: rec[k] for k in ("id", "kind", "mime", "name", "size")}
                          for rec in staged])


@app.post("/api/approve")
def approve_route():
    b = request.get_json(force=True, silent=True) or {}
    a = approvals.get(b.get("id"))
    if a:
        a["ok"] = bool(b.get("ok")); a["event"].set()
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    b = request.get_json(force=True, silent=True) or {}
    c = cancels.get(b.get("chat") or b.get("session", "default"))
    if c: c.set()
    return jsonify(ok=True)


@app.post("/api/reset")
def reset():
    sid = (request.get_json(force=True, silent=True) or {}).get("session", "default")
    if sid in sessions:
        try: checkpoint(sessions.pop(sid))
        except Exception: sessions.pop(sid, None)
    return jsonify(ok=True)


@app.post("/api/tts")
def tts():
    """Synthesize text with Edge TTS (Microsoft neural voices). Returns MP3 audio."""
    try:
        from connectors.edge_tts import speak
    except ImportError:
        return jsonify(error="edge-tts not installed: pip install edge-tts"), 501
    text = (request.get_json(force=True, silent=True) or {}).get("text", "")
    if not text.strip():
        return jsonify(error="empty text"), 400
    try:
        audio = speak(text)
    except Exception as e:
        return jsonify(error=f"TTS failed: {e}"), 502
    return Response(audio, mimetype="audio/mpeg", headers={"Cache-Control": "no-cache"})


@app.route("/api/memory", methods=["GET", "POST", "DELETE"])
def memory():
    b = request.get_json(silent=True) or {}
    if request.method == "POST": store.add_facts([b.get("fact", "")], source="manual")
    if request.method == "DELETE":
        store.clear() if b.get("all") else store.remove(int(b.get("index", 0)))
    return jsonify(memory=store.memory_texts())


if __name__ == "__main__":
    port = int(os.getenv("VIKI_PORT", "7860"))
    host = os.getenv("VIKI_HOST", "0.0.0.0")
    print(f"VIKI ready: http://{host}:{port}  (model {viki.model})")
    app.run(host=host, port=port, threaded=True)
