"""VIKI web server. Run: python server.py  then open http://localhost:7860"""
import json, os, queue, threading, uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, Response, jsonify, request, send_from_directory

from agent import VIKI
from memory import SessionStore
from tools import Toolbox, WORKSPACE

app = Flask(__name__, static_folder="static")
store = SessionStore()
viki = VIKI(Toolbox(store))
# Pre-load the LLM at startup so the first request doesn't pay load time.
# Runs in a thread so a slow/unreachable Ollama doesn't block server boot.
threading.Thread(target=viki.warmup, daemon=True).start()
AUTO_MEMORY = os.getenv("VIKI_AUTO_MEMORY", "on").lower() != "off"
sessions, approvals, cancels = {}, {}, {}


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
                   files=len(list(WORKSPACE.rglob("*"))), warning=store.warning)


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    sid, prompt, think = body.get("session", "default"), (body.get("message") or "").strip(), bool(body.get("think"))
    if not prompt:
        return jsonify(error="empty message"), 400
    s, q, cancel = get_session(sid), queue.Queue(), threading.Event()
    cancels[sid] = cancel
    history = [{"role": m["role"], "content": m["content"]} for m in s["messages"][-20:]]

    def emit(kind, payload):
        q.put({"kind": kind, "data": payload})

    def approve(cid, name, args):
        ev = threading.Event(); approvals[cid] = {"event": ev, "ok": False}
        emit("approval", {"id": cid, "name": name, "args": args})
        ev.wait(timeout=180)
        return approvals.pop(cid, {}).get("ok", False)

    def work():
        res = viki.run(prompt, history, store.search, emit, approve, think, cancel)
        if res["type"] != "error":
            now = datetime.now().isoformat(timespec="seconds")
            s["messages"] += [{"role": "user", "content": prompt, "time": now},
                              {"role": "assistant", "content": res["content"], "time": now, "tools": res.get("tools")}]
        emit("done", res)
        q.put(None)
        try: checkpoint(s)
        except Exception: pass

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (item := q.get()) is not None:
            yield f"data: {json.dumps(item)}\n\n"
    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/approve")
def approve_route():
    b = request.get_json(force=True)
    a = approvals.get(b.get("id"))
    if a:
        a["ok"] = bool(b.get("ok")); a["event"].set()
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    c = cancels.get(request.get_json(force=True).get("session", "default"))
    if c: c.set()
    return jsonify(ok=True)


@app.post("/api/reset")
def reset():
    sid = request.get_json(force=True).get("session", "default")
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
