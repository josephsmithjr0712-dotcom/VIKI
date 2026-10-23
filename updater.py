"""VIKI self-updater.

Checks your private GitHub repo for new releases, downloads them,
applies them safely (your .env, tokens, and data are never touched),
and restarts VIKI.

Setup in .env:
    VIKI_UPDATE_REPO=owner/repo      # your private GitHub repo
    GITHUB_TOKEN=...                 # already set for the GitHub connector

Usage:
    python updater.py check          # is there a newer release?
    python updater.py update         # download + apply + restart (asks first)

As tools (the agent can use these):
    viki_check_update()              # safe, read-only
    viki_apply_update()              # risky — needs your approval

Versioning: a VERSION file in the repo root (e.g. "0.2.0").
Releases: GitHub Releases with tag like "v0.3.0" and a zip asset,
or just use the auto-generated source zipball.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION_FILE = HERE / "VERSION"

# Never overwritten by an update — your secrets and data live here.
PRESERVE = {
    ".env",
    ".venv",
    ".git",
    "workspace",
    "backups",
    "__pycache__",
}
# Token/secret files inside connectors/ (matched by substring).
PRESERVE_SUBSTRINGS = ("token", "secret")


def _repo():
    repo = os.getenv("VIKI_UPDATE_REPO", "").strip()
    if not repo or "/" not in repo:
        raise RuntimeError(
            "VIKI_UPDATE_REPO is not set in .env (e.g. VIKI_UPDATE_REPO=joey/viki)."
        )
    return repo


def _gh_token():
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set in .env.")
    return token


def local_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"


def _gh_api(path: str):
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {_gh_token()}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "viki-updater",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"GitHub repo '{_repo()}' not found or no releases yet. "
                "Create a release in your private repo first."
            )
        if e.code == 401:
            raise RuntimeError("GITHUB_TOKEN rejected (401). Check the token.")
        raise RuntimeError(f"GitHub API error {e.code}: {e.reason}")


def _norm(v: str) -> tuple:
    v = v.strip().lstrip("v")
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts + [0] * (3 - len(parts)))


def latest_release() -> dict:
    """Return {version, tag, zip_url, notes} for the newest GitHub release."""
    data = _gh_api(f"/repos/{_repo()}/releases/latest")
    tag = data.get("tag_name", "")
    # Prefer an uploaded asset zip, else the auto zipball.
    zip_url = data.get("zipball_url", "")
    for asset in data.get("assets", []):
        name = asset.get("name", "").lower()
        if name.endswith(".zip"):
            zip_url = asset.get("browser_download_url", zip_url)
            break
    return {
        "version": tag.lstrip("v"),
        "tag": tag,
        "zip_url": zip_url,
        "notes": data.get("body", "")[:500],
    }


def check() -> dict:
    """Check for updates. Read-only. Returns status dict."""
    local = local_version()
    try:
        rel = latest_release()
    except RuntimeError as e:
        return {"ok": False, "error": str(e), "local": local}
    update_available = _norm(rel["version"]) > _norm(local)
    return {
        "ok": True,
        "local": local,
        "latest": rel["version"],
        "tag": rel["tag"],
        "update_available": update_available,
        "notes": rel["notes"],
        "zip_url": rel["zip_url"] if update_available else "",
    }


def _download(url: str, dest: Path):
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {_gh_token()}", "User-Agent": "viki-updater"},
    )
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 64)


def _should_preserve(rel: Path) -> bool:
    name = rel.name
    if name in PRESERVE or str(rel).startswith(".git"):
        return True
    low = name.lower()
    return any(s in low for s in PRESERVE_SUBSTRINGS)


def apply_update(zip_url: str = "") -> str:
    """Download and apply the latest release. Returns a status message.

    Safe: backs up the current install (minus secrets/data/venv) to
    backups/, then overlays the new files. .env, tokens, workspace/,
    and .venv are never touched.
    """
    st = check()
    if not st["ok"]:
        return f"Update check failed: {st['error']}"
    if not st["update_available"]:
        return f"Already on the latest version ({st['local']})."
    url = zip_url or st["zip_url"]
    if not url:
        return "No downloadable asset found on the release."

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = HERE / "backups" / f"viki-{st['local']}-{stamp}"
    work = Path(tempfile.mkdtemp(prefix="viki-update-"))
    try:
        # 1. Download
        archive = work / "release.zip"
        _download(url, archive)

        # 2. Extract (strip the top-level repo folder GitHub adds)
        with zipfile.ZipFile(archive) as z:
            z.extractall(work / "extracted")
        extracted = work / "extracted"
        tops = [p for p in extracted.iterdir() if p.is_dir()]
        src = tops[0] if len(tops) == 1 else extracted

        # 3. Sanity: the release must contain the core files
        for required in ("agent.py", "server.py", "tools.py", "VERSION"):
            if not (src / required).exists():
                return (
                    f"Release looks incomplete (missing {required}). "
                    "Aborted — nothing was changed."
                )

        # 4. Back up current install (excluding preserved paths)
        backup_dir.mkdir(parents=True, exist_ok=True)
        for item in HERE.iterdir():
            if item.name == "backups":
                continue
            rel = item.relative_to(HERE)
            if _should_preserve(rel):
                continue
            dest = backup_dir / rel
            if item.is_dir():
                shutil.copytree(item, dest, symlinks=True)
            else:
                shutil.copy2(item, dest)

        # 5. Overlay new files (never touching preserved paths)
        for item in src.rglob("*"):
            rel = item.relative_to(src)
            if _should_preserve(rel):
                continue
            dest = HERE / rel
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

        return (
            f"Updated {st['local']} -> {st['latest']}. "
            f"Backup at backups/{backup_dir.name}. Restart VIKI to run it."
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def restart() -> str:
    """Restart VIKI after an update. Uses systemd on the server, re-exec locally."""
    # Systemd (Debian server deployment)
    if Path("/run/systemd/system").exists():
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "viki-server", "viki-voice"],
                check=True, timeout=30,
            )
            return "Restarted viki-server and viki-voice via systemd."
        except Exception as e:
            return f"systemd restart failed: {e}"
    # Local dev: re-exec the server (the old process exits, new one takes over)
    print("Re-executing server.py with the new code...")
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, str(HERE / "server.py")])


def main(argv):
    if len(argv) < 2 or argv[1] not in ("check", "update", "restart"):
        print(__doc__)
        return 1
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    try:
        if argv[1] == "check":
            st = check()
            if not st["ok"]:
                print(f"Check failed: {st['error']}")
                return 1
            print(f"Local:  {st['local']}")
            print(f"Latest: {st['latest']}")
            print("Update available!" if st["update_available"] else "You're up to date.")
            if st["notes"]:
                print(f"\nRelease notes:\n{st['notes']}")
        elif argv[1] == "update":
            if "--yes" not in argv:
                st = check()
                if not st.get("update_available"):
                    print(f"Nothing to do ({st.get('local')}).")
                    return 0
                answer = input(f"Update {st['local']} -> {st['latest']}? [y/N] ").strip().lower()
                if answer != "y":
                    print("Aborted.")
                    return 0
            print(apply_update())
        elif argv[1] == "restart":
            print(restart())
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
