"""One-time Google OAuth setup for VIKI.

Run from the VIKI folder:
    python -m connectors.google_auth

1. If client_secrets.json is missing, this prints where to get it:
   Google Cloud Console -> APIs & Services -> Credentials
   -> Create Credentials -> OAuth client ID -> Desktop app -> Download JSON.
   Save it as connectors/client_secrets.json (or set GOOGLE_CLIENT_SECRETS_FILE).
2. A browser window opens for approval. The token is saved to
   GOOGLE_TOKEN_FILE (default connectors/google_token.json).
3. Restart server.py — the Google tools enable automatically.
"""
from connectors.google import SCOPES, SECRETS_FILE, TOKEN_FILE


def main():
    if not SECRETS_FILE.exists():
        print(f"Missing {SECRETS_FILE}.\n")
        print("1. Go to https://console.cloud.google.com/apis/credentials")
        print("2. Create Credentials -> OAuth client ID -> Desktop app")
        print("3. Download the JSON and save it as:")
        print(f"     {SECRETS_FILE}")
        print("   (or set GOOGLE_CLIENT_SECRETS_FILE to its path)")
        print("4. Enable the Gmail API and Google Calendar API for the project,")
        print("   then re-run:  python -m connectors.google_auth")
        raise SystemExit(1)
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(SECRETS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    print(f"Saved token to {TOKEN_FILE}. Google tools are now enabled — restart server.py.")


if __name__ == "__main__":
    main()
