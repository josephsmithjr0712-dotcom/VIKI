"""GitHub connector for VIKI: repo/code search, file reads, issues.

Setup: create a personal access token (classic or fine-grained) at
https://github.com/settings/tokens and set GITHUB_TOKEN in the environment
(or .env). No OAuth dance needed. Tools enable automatically once the token
is set. Uses the REST API directly — no extra dependencies.
"""
import base64
import json
import os
import urllib.parse
import urllib.request

API = "https://api.github.com"
UA = {"User-Agent": "Mozilla/5.0 (VIKI agent)"}


def _token():
    # Read at call time, not import time: a token added after the
    # server started must work without a restart.
    return os.getenv("GITHUB_TOKEN", "")


def token_exists() -> bool:
    return bool(_token())


def _api(path: str, method: str = "GET", data: dict | None = None):
    """Call the GitHub API. Returns (ok, parsed_json_or_error_string)."""
    token = _token()
    if not token:
        return False, "GitHub is not connected. Set GITHUB_TOKEN in the environment first."
    headers = dict(UA)
    headers["Authorization"] = f"Bearer {token}"
    headers["Accept"] = "application/vnd.github+json"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(API + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return True, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            detail = json.load(e).get("message", "")
        except Exception:
            detail = ""
        return False, f"GitHub API error {e.code}: {detail or e.reason}"
    except Exception as e:
        return False, f"GitHub request failed: {e}"


def github_search_repos(query, max_results=5):
    ok, res = _api(f"/search/repositories?q={urllib.parse.quote(query)}&per_page={int(max_results)}")
    if not ok:
        return res
    items = res.get("items", [])
    if not items:
        return "No repositories found."
    return "\n\n".join(
        f"{r['full_name']} ★{r['stargazers_count']}\n{r.get('description') or '(no description)'}\n{r['html_url']}"
        for r in items
    )


def github_search_code(query, max_results=5):
    ok, res = _api(f"/search/code?q={urllib.parse.quote(query)}&per_page={int(max_results)}")
    if not ok:
        return res
    items = res.get("items", [])
    if not items:
        return "No code matches found."
    return "\n\n".join(
        f"{c['repository']['full_name']} / {c['path']}\n{c['html_url']}" for c in items
    )


def github_read_file(repo, path, ref=""):
    """Read a file from a repo ('owner/name', 'path/to/file')."""
    q = f"/repos/{repo}/contents/{urllib.parse.quote(path)}"
    if ref:
        q += f"?ref={urllib.parse.quote(ref)}"
    ok, res = _api(q)
    if not ok:
        return res
    if isinstance(res, dict) and res.get("encoding") == "base64" and "content" in res:
        try:
            text = base64.b64decode(res["content"]).decode("utf-8", "ignore")
        except Exception as e:
            return f"Could not decode file: {e}"
        return text[:20000] or "(empty file)"
    return "Not a file (maybe a directory)."


def github_list_issues(repo, state="open", max_results=10):
    ok, res = _api(
        f"/repos/{repo}/issues?state={urllib.parse.quote(state)}&per_page={int(max_results)}"
    )
    if not ok:
        return res
    if not res:
        return f"No {state} issues in {repo}."
    out = []
    for i in res:
        labels = ",".join(l["name"] for l in i.get("labels", []))
        out.append(f"#{i['number']} {i['title']}" + (f" [{labels}]" if labels else ""))
    return "\n".join(out)


def github_create_issue(repo, title, body=""):
    ok, res = _api(f"/repos/{repo}/issues", method="POST", data={"title": title, "body": body})
    if not ok:
        return res
    return f"Created issue #{res.get('number')} in {repo}: {res.get('html_url')}"
