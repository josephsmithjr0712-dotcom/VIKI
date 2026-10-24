"""Secure attachment staging for VIKI's web UI.

Security design (deliberately paranoid):
- Never trust the client. File type is decided by magic bytes, never by the
  filename extension or the claimed Content-Type.
- SVG / HTML / XML are rejected outright: they can carry scripts, and an
  endpoint that stores or echoes them is a stored-XSS waiting to happen.
- Staged files live under the OS temp dir (never in the workspace), are named
  by UUID (never by the user-supplied filename — this also kills path
  traversal), and are single-use: an upload ID is consumed by one chat turn
  and the files are deleted right after. Orphans older than an hour are swept
  on every upload.
- Hard caps: 10 MB per file, 3 files per message, 4 video frames, 100 KB of
  inlined text. Staged files are never executed, never served back as
  downloads, and only ever sent to the local Ollama — never to the internet.
"""
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024   # per file
MAX_FILES = 3                  # per message
MAX_TEXT_CHARS = 100_000       # inlined text cap
MAX_VIDEO_FRAMES = 4
ORPHAN_TTL_SECS = 3600

# mime -> safe extension, keyed by what WE sniffed (not what the client claimed)
_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "text/plain": ".txt",
}


class UploadError(ValueError):
    """Raised when an upload fails validation. Safe to show to the user."""


def sniff(data: bytes):
    """Identify (kind, mime) from magic bytes.

    Returns (None, None) for anything not explicitly allowed — including
    SVG/HTML/XML, which all start with '<'.
    """
    if not data:
        return None, None
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image", "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image", "image/jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image", "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image", "image/webp"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "video", "video/mp4"
    if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
        return "video", "video/webm"
    if data[:1024].lstrip().startswith(b"<"):
        return None, None  # markup of any kind: SVG, HTML, XML — reject
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None, None
    if b"\x00" in data:
        return None, None  # null bytes = binary masquerading as text
    return "text", "text/plain"


def _sweep_orphans(base: Path):
    now = time.time()
    try:
        for d in base.glob("viki-up-*"):
            try:
                if now - d.stat().st_mtime > ORPHAN_TTL_SECS:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
    except OSError:
        pass


def stage_files(items, base=None):
    """Stage uploaded files securely.

    items: iterable of (filename, data: bytes). At most MAX_FILES are kept.
    Returns a list of dicts: {id, kind, mime, name, size, path}.
    Raises UploadError on any problem; nothing is left staged on failure.
    """
    base = Path(base or tempfile.gettempdir())
    _sweep_orphans(base)
    items = list(items)[:MAX_FILES]
    if not items:
        raise UploadError("no files provided")
    tmpdir = Path(tempfile.mkdtemp(prefix="viki-up-", dir=str(base)))
    staged = []
    try:
        for filename, data in items:
            label = (filename or "file")[:80]
            if not data:
                raise UploadError(f"{label} is empty")
            if len(data) > MAX_BYTES:
                raise UploadError(f"{label} exceeds the 10 MB limit")
            kind, mime = sniff(data)
            if kind is None:
                raise UploadError(
                    f"{label}: unsupported or unsafe file type. "
                    "Images (PNG/JPEG/GIF/WebP), video (MP4/WebM), and plain text only.")
            fid = uuid.uuid4().hex
            path = tmpdir / (fid + _EXT[mime])
            path.write_bytes(data)
            staged.append({
                "id": fid, "kind": kind, "mime": mime,
                # Path.name strips any directory components: no traversal.
                "name": Path(label).name or "file",
                "size": len(data), "path": str(path),
            })
        return staged
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def discard(staged):
    """Delete staged files. Best effort; never raises."""
    seen = set()
    for s in staged:
        try:
            d = str(Path(s["path"]).parent)
        except Exception:
            continue
        if d not in seen:
            seen.add(d)
            shutil.rmtree(d, ignore_errors=True)


def video_frames(path, max_frames=MAX_VIDEO_FRAMES):
    """Extract up to max_frames evenly-spaced JPEG frames via ffmpeg.

    Returns a list of JPEG bytes, or [] when ffmpeg is unavailable or
    extraction fails. Never raises.
    """
    if not shutil.which("ffmpeg"):
        return []
    outdir = Path(tempfile.mkdtemp(prefix="viki-fr-"))
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vf", "fps=1/8", "-vframes", str(max_frames),
             str(outdir / "f%02d.jpg")],
            timeout=90, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return [p.read_bytes() for p in sorted(outdir.glob("f*.jpg"))[:max_frames]]
    except Exception:
        return []
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def read_text(path, max_chars=MAX_TEXT_CHARS):
    """Read a staged text file, capped. Never raises."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""
