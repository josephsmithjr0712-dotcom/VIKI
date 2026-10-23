# VIKI

**Your own personal AI assistant — running on your machine, not in someone else's cloud.**

VIKI is a local-first autonomous agent powered by Ollama. She plans multi-step tasks, uses tools, controls your smart home, manages your email and calendar, browses the web, and talks back — all from hardware you own.

---

## ✨ What she does

- **💬 Chat with streaming** — fast, natural conversation through a polished web UI or the Android companion app
- **🧠 She plans** — breaks complex requests into steps, works through them, and shows her progress live
- **🔧 She uses tools** — web search, page reading, file read/write, calculator, weather, and more
- **🏠 Smart home control** — Home Assistant (lights, switches, media players), Google Home Mini announcements, Roku, Plex
- **📧📅 Google integration** — Gmail search/read/send, Calendar list/create
- **🎙️ Voice mode** — wake-word activation ("hey jarvis" by default), speech-to-text, neural text-to-speech
- **🧠 She remembers** — long-term memory with relevance-ranked recall and automatic fact extraction
- **🔄 She updates herself** — checks GitHub releases and applies updates with your approval, preserving your config and memory
- **🔒 Password-protected** — web login with persistent sessions; risky actions ask for approval first

---

## 🚀 Quick start

### Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) running with a model pulled (e.g. `ollama pull qwen3:8b`)
- A Debian/Ubuntu server, Mac, or any machine that can run Python

### Install

```bash
git clone https://github.com/josephsmithjr0712-dotcom/VIKI.git
cd VIKI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file — everything is optional except the model:

```env
# --- Core ---
VIKI_MODEL=qwen3:8b
VIKI_ADDRESS=http://localhost:11434
VIKI_HOST=0.0.0.0
VIKI_PORT=7860

# --- Web login (auto-generated on first run if blank) ---
VIKI_WEB_PASSWORD=choose-something-strong

# --- Self-updater ---
VIKI_UPDATE_REPO=your-github-user/VIKI
GITHUB_TOKEN=ghp_...

# --- Home Assistant ---
HA_URL=http://homeassistant.local:8123
HA_TOKEN=your-long-lived-token
LAMP_ENTITY=light.lamp          # status lamp for voice mode

# --- Google (run: python -m connectors.google_auth) ---
# Creates google_token.json after a one-time OAuth flow.

# --- GitHub ---
GITHUB_TOKEN=ghp_...

# --- Plex ---
PLEX_URL=http://plex.local:32400
PLEX_TOKEN=...

# --- Spotify (run: python -m connectors.spotify_auth) ---
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...

# --- Voice ---
VIKI_TTS_VOICE=en-US-AriaNeural   # Edge TTS voice
```

### Run

```bash
python server.py
```

Open **http://localhost:7860** — you'll get a login screen on first visit. The password is auto-generated and printed in the server log (and saved to `.env` as `VIKI_WEB_PASSWORD`).

### Voice mode (optional)

```bash
python voice_loop.py
```

Say **"hey jarvis"**, then talk. She answers through your Home Mini (or local speakers). Needs a microphone. On first run it downloads the wake-word and speech models.

---

## 🖥️ Web UI

The web interface mirrors the Android companion app: aurora backdrop, the spinning VIKI orb, glassmorphism cards, and a bottom tab bar.

- **Chat** — streaming responses, tool-call cards, plan checklists, approval buttons
- **History** — browse and reopen past conversations
- **Settings** — toggles (show reasoning, read replies aloud), memory manager, tool/connector status

It also works great as a mobile web app — add it to your home screen.

---

## 🔌 Connectors

| Connector | What it does | Setup |
|---|---|---|
| Google | Gmail search/read/send, Calendar list/create | `python -m connectors.google_auth` |
| GitHub | Repo search, issues, PRs | `GITHUB_TOKEN` in `.env` |
| Browser | Headless Chromium: navigate, read, click, fill | `pip install playwright && python -m playwright install chromium` |
| Home Assistant | Lights, switches, media, scenes, full entity dump | `HA_URL` + `HA_TOKEN` |
| Home Mini | Voice announcements via TTS | Auto-discovers via Home Assistant |
| Roku | Launch apps, media control | Via Home Assistant |
| Plex | Search library, play on any player | `PLEX_URL` + `PLEX_TOKEN` |
| Spotify | Search, play, devices, volume | `python -m connectors.spotify_auth` |
| Edge TTS | Microsoft neural voices for replies | `pip install edge-tts` |

Don't use a connector? Leave it unconfigured — VIKI simply won't offer those tools.

---

## 🧰 Built-in tools

`web_search` · `read_webpage` · `calculate` · `get_weather` · `current_time` · `list_files` · `read_file` · `write_file` · `remember` · `forget` · `update_plan` · `viki_check_update` · `viki_apply_update` · plus every connector above.

**Approvals:** risky actions (sending email, writing files, controlling locks/alarms, applying updates) pause for your Allow/Deny. Routine stuff — lights, media, searches — just happens.

---

## 🔄 Self-updater

VIKI checks GitHub releases for new versions:

- `viki_check_update` — safe, runs anytime, no approval needed
- `viki_apply_update` — downloads the release, backs up your install, swaps in new files, restarts, and **preserves** `.env`, tokens, `workspace/`, and `memory/`. Requires your approval.

---

## 🔒 Security

- Web UI sits behind a password login with signed session cookies ("keep me signed in" lasts 30 days)
- Login is rate-limited (5 failures → 5-minute lockout per IP)
- Approval IDs are cryptographically random — no guessing
- Browser tool blocks `file:`, `javascript:`, and `data:` URLs and private-network addresses
- No secrets in code or git history; tokens live in `.env` (gitignored) or local token files
- Binds `0.0.0.0` by default for LAN/Tailscale access — set `VIKI_HOST=127.0.0.1` if you only want localhost

---

## 🧪 Development

```bash
# Run the full test suite (each file runs standalone, no pytest needed)
for t in tests/test_*.py; do python3 "$t"; done
```

The suite covers the agent loop, memory, tools, updater, connectors (mocked), and the web UI's JS.

### Project layout

```
VIKI/
├── server.py          # Flask web server + API + login
├── agent.py           # Plan → act → observe loop over Ollama
├── tools.py           # Tool specs, approval gates, built-ins
├── memory.py          # Long-term memory + session store
├── updater.py         # Self-update from GitHub releases
├── voice_loop.py      # Wake-word → STT → agent → TTS pipeline
├── personality.md     # Who VIKI is
├── connectors/        # Google, GitHub, HA, Spotify, Plex, Roku, TTS…
├── static/            # Web UI (index.html, login.html)
└── tests/             # Test suite
```

---

## 📱 Android companion app

A native Kotlin + Jetpack Compose app with the same design language, streaming chat, approval buttons, voice input, and an on-device LLM fallback for offline use. It talks to this server over your LAN.

---

## 📄 License

MIT — do what you want with it. Your data stays yours: everything runs locally, and nothing phones home.
