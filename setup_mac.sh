#!/bin/bash
# VIKI one-shot Mac setup - run: bash setup_mac.sh
set -e
cd "$(dirname "$0")"

PY=/usr/local/bin/python3
if [ ! -x "$PY" ]; then PY=python3; fi

echo "=== 1/5 Creating virtual environment ==="
if [ ! -d ".venv" ]; then
  $PY -m venv .venv
  echo "venv created"
else
  echo "venv already exists, keeping it"
fi

echo "=== 2/5 Installing dependencies (this takes a few minutes) ==="
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt

echo "=== 3/5 Installing Playwright browser ==="
.venv/bin/python -m playwright install chromium 2>&1 | tail -2

echo "=== 4/5 Running tests ==="
PASS=0; FAIL=0
for t in tests/test_*.py; do
  if .venv/bin/python "$t" > /dev/null 2>&1; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); echo "FAILED: $t"; fi
done
echo "Tests: $PASS passed, $FAIL failed"

echo "=== 5/5 Google auth ==="
if [ -f "connectors/client_secrets.json" ]; then
  echo "Opening Google sign-in..."
  .venv/bin/python -m connectors.google_auth || echo "(skip for now - run '.venv/bin/python -m connectors.google_auth' later)"
else
  echo "SKIP: put your client_secrets.json in connectors/ first"
fi

echo ""
echo "Done! Start VIKI with: .venv/bin/python server.py"
echo "Then open http://localhost:7860 in your browser."
