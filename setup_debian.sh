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
"$VIKI_DIR/.venv/bin/pip" install -r "$VIKI_DIR/requirements-voice.txt"

# 3. Playwright browser
echo "-- installing Playwright Chromium --"
"$VIKI_DIR/.venv/bin/python" -m playwright install chromium

# 4. .env reminder
if [ ! -f "$VIKI_DIR/.env" ]; then
    echo "!! No .env found — copy yours from the Mac before starting the services."
fi

# 5. systemd services

echo "-- installing systemd services (sudo needed) --"
for svc in viki-server viki-voice; do
    sudo cp "$VIKI_DIR/$svc.service" /etc/systemd/system/
    sudo sed -i "s|/opt/viki|$VIKI_DIR|g" "/etc/systemd/system/$svc.service"
done
sudo systemctl daemon-reload
sudo systemctl enable viki-server viki-voice

echo ""
echo "== Done. Next steps: =="
echo "  1. Make sure $VIKI_DIR/.env is filled in (Ollama, HA, API keys)."
echo "  2. Mic check: $VIKI_DIR/.venv/bin/python -c 'import sounddevice; print(sounddevice.query_devices())'"
echo "  3. Sanity:    cd $VIKI_DIR && .venv/bin/python voice_loop.py --check"
echo "  4. Start:     sudo systemctl start viki-server viki-voice"
echo "  5. Web UI:    http://<server-ip>:7860"
echo "  6. Logs:      journalctl -u viki-server -f  /  journalctl -u viki-voice -f"
