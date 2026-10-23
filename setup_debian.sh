#!/bin/bash
# VIKI server setup for Debian 13. Run on the server (NOT the Mac).
# Copy the whole VIKI folder to the server first, then: bash setup_debian.sh
#
# What it does:
#   1. Installs system packages (Python venv, portaudio for mic/speakers)
#   2. Creates a venv and installs all VIKI requirements (web + voice)
#   3. Installs Playwright Chromium for the browser tool
#   4. Installs systemd services: viki-server (web UI) + viki-voice (wake word)
set -e

VIKI_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "== VIKI deploy: $VIKI_DIR =="

# 1. System packages
echo "-- installing system packages (sudo needed) --"
sudo apt update
sudo apt install -y python3-venv python3-pip portaudio19-dev libportaudio2 \
    libsndfile1 mpv curl

# 2. Venv + Python deps
if [ ! -d "$VIKI_DIR/.venv" ]; then
    echo "-- creating venv --"
    python3 -m venv "$VIKI_DIR/.venv"
fi
echo "-- installing Python requirements (takes a few minutes) --"
"$VIKI_DIR/.venv/bin/pip" install --upgrade pip
"$VIKI_DIR/.venv/bin/pip" install -r "$VIKI_DIR/requirements.txt"
# Voice stack is optional: tflite-runtime (needed by openwakeword) has no
# Python 3.13 wheels, so this fails on Debian 13. The web UI works fine
# without it; voice can be added later on a Python 3.11 venv if needed.
if ! "$VIKI_DIR/.venv/bin/pip" install -r "$VIKI_DIR/requirements-voice.txt"; then
    echo "!! Voice requirements failed (expected on Python 3.13) — continuing without voice."
    echo "!! To enable voice later, use a Python 3.11 venv or wait for tflite-runtime wheels."
fi

# 3. Playwright browser
echo "-- installing Playwright Chromium --"
"$VIKI_DIR/.venv/bin/python" -m playwright install chromium

# 4. .env reminder
if [ ! -f "$VIKI_DIR/.env" ]; then
    echo "!! No .env found — copy yours from the Mac before starting the services."
fi

# 5. systemd services
# viki-server (web UI) always; viki-voice only if the voice deps installed.
echo "-- installing systemd services (sudo needed) --"
sudo cp "$VIKI_DIR/viki-server.service" /etc/systemd/system/
sudo sed -i "s|/opt/viki|$VIKI_DIR|g" /etc/systemd/system/viki-server.service
sudo systemctl enable viki-server
if "$VIKI_DIR/.venv/bin/python" -c "import openwakeword" 2>/dev/null; then
    sudo cp "$VIKI_DIR/viki-voice.service" /etc/systemd/system/
    sudo sed -i "s|/opt/viki|$VIKI_DIR|g" /etc/systemd/system/viki-voice.service
    sudo systemctl enable viki-voice
    VOICE_OK=1
else
    echo "!! openwakeword not installed — skipping viki-voice service (web UI only)."
    VOICE_OK=0
fi
sudo systemctl daemon-reload

echo ""
echo "== Done. Next steps: =="
echo "  1. Make sure $VIKI_DIR/.env is filled in (Ollama, HA, API keys)."
if [ "$VOICE_OK" = "1" ]; then
echo "  2. Mic check: $VIKI_DIR/.venv/bin/python -c 'import sounddevice; print(sounddevice.query_devices())'"
echo "  3. Sanity:    cd $VIKI_DIR && .venv/bin/python voice_loop.py --check"
echo "  4. Start:     sudo systemctl start viki-server viki-voice"
else
echo "  2. Start:     sudo systemctl start viki-server"
echo "     (voice loop skipped — no wake-word engine on Python 3.13)"
fi
echo "  5. Web UI:    http://<server-ip>:7860"
echo "  6. Logs:      journalctl -u viki-server -f"
