#!/usr/bin/env bash
# update-viki.sh — one-command VIKI update for the HomeLab server (Debian + systemd).
#
#   bash update-viki.sh            update and restart, no questions asked
#   bash update-viki.sh --dry-run  show what would happen, change nothing
#
# (Run it via bash, not ./ — the GitHub push path doesn't preserve the
# executable bit, so don't count on chmod +x surviving a fresh clone.)
#
# What it does:
#   1. Stashes any dirty local changes (tracked AND untracked, tagged with a
#      timestamp) so `git pull` can never block on "your local changes would
#      be overwritten by merge" again.
#   2. Hard-resets the checkout to origin/main.
#   3. Restarts the viki-server systemd service.
#   4. Prints the service status.
#
# Config is safe: .env, workspace/, memory/ and tokens are gitignored, so the
# stash and the hard reset never touch them.
#
# HomeLab-server-only. NOT the Intel-Mac update path — the Mac has no systemd,
# so this script refuses to run where systemd isn't running.

set -euo pipefail

SERVICE="viki-server"
BRANCH="main"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            echo "usage: $0 [--dry-run]"
            exit 0 ;;
        *) echo "!! unknown option: $arg (usage: $0 [--dry-run])" >&2; exit 2 ;;
    esac
done

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN: $*"
    else
        "$@"
    fi
}

cd "$(dirname "$0")"

# --- 1. systemd-only guard -----------------------------------------------
if ! command -v systemctl >/dev/null 2>&1 || [ ! -d /run/systemd/system ]; then
    echo "!! update-viki.sh is HomeLab-server-only (needs systemd)." >&2
    echo "!! Not running it on this machine. On the Intel Mac, pull and restart VIKI manually." >&2
    exit 1
fi

# --- 2. must be the VIKI checkout ----------------------------------------
if [ ! -d .git ]; then
    echo "!! not a git checkout — run this from inside the VIKI repo." >&2
    exit 1
fi
REMOTE="$(git remote get-url origin 2>/dev/null || true)"
case "$REMOTE" in
    *josephsmithjr0712-dotcom/VIKI*) ;;
    *) echo "!! origin ($REMOTE) is not the VIKI repo. Aborting." >&2; exit 1 ;;
esac

STAMP="$(date +%Y-%m-%d-%H%M)"

# --- 3. stash dirty local changes so the update can't be blocked ---------
if [ -n "$(git status --porcelain)" ]; then
    echo "-- stashing local changes as 'pre-update-$STAMP' --"
    run git stash push -u -m "pre-update-$STAMP"
else
    echo "-- working tree clean, nothing to stash --"
fi

# --- 4. hard-reset to origin/main ----------------------------------------
echo "-- updating to origin/$BRANCH --"
run git fetch origin
run git reset --hard "origin/$BRANCH"

# --- 5. restart the web UI ------------------------------------------------
echo "-- restarting $SERVICE --"
run sudo systemctl restart "$SERVICE"

# --- 6. report ------------------------------------------------------------
echo "-- $SERVICE status --"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN: systemctl is-active $SERVICE && systemctl status $SERVICE"
else
    systemctl is-active "$SERVICE"
    sudo systemctl status "$SERVICE" --no-pager --lines=10 || true
fi

echo "done."
