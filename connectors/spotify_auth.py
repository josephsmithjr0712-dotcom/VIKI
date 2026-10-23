"""One-time Spotify OAuth setup for VIKI.

1. Create an app at https://developer.spotify.com/dashboard.
2. In the app settings, add this Redirect URI:
     http://127.0.0.1:8888/callback
   (or set SPOTIFY_REDIRECT_URI to your own and add that instead).
3. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env.
4. Run:  python -m connectors.spotify_auth
   A browser window opens for approval; the token is cached at
   connectors/.spotify_cache and refreshes itself afterwards.

Note: API playback control needs Spotify Premium.
"""
import os
import sys


def main():
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env first.")
        print("Create an app at https://developer.spotify.com/dashboard")
        raise SystemExit(1)
    from spotipy.oauth2 import SpotifyOAuth
    from connectors.spotify import SCOPES, REDIRECT, CACHE

    auth = SpotifyOAuth(
        client_id=cid, client_secret=secret, redirect_uri=REDIRECT,
        scope=SCOPES, cache_path=str(CACHE), open_browser=True,
    )
    token = auth.get_access_token(as_dict=False)
    if token:
        print(f"Saved Spotify token to {CACHE}. Spotify tools are now enabled.")
    else:
        print("Authorization failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
