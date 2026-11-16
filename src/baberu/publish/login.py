"""baberu-nekocap-login: saves your NekoCap login to .env in one step.

It opens nekocap.com and puts a one-line script on the clipboard. Pasting that
into the browser console copies your session token back to the clipboard;
this command then reads it from there, checks it with NekoCap, and saves it.
"""
from __future__ import annotations

import getpass
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import dotenv

from baberu.errors import BaberuError
from baberu.publish.nekocap import NekoCapClient
from baberu.settings import NekoCapSettings

ENV_KEY = "NEKOCAP_SESSION_TOKEN"
# Finds Parse's saved user whatever the app id, and copies its token.
CONSOLE_SNIPPET = ("copy(JSON.parse(localStorage[Object.keys(localStorage)"
                   ".find(k => k.endsWith('/currentUser'))]).sessionToken)")


def _set_clipboard(text: str) -> bool:
    try:
        if sys.platform == "win32":
            subprocess.run(["clip"], input=text.encode("utf-8"), check=True)
        elif sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        else:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _get_clipboard() -> str:
    try:
        if sys.platform == "win32":
            command = ["powershell", "-NoProfile", "-Command", "Get-Clipboard"]
        elif sys.platform == "darwin":
            command = ["pbpaste"]
        else:
            command = ["xclip", "-selection", "clipboard", "-o"]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _env_file() -> Path:
    found = dotenv.find_dotenv(usecwd=False) or dotenv.find_dotenv(usecwd=True)
    return Path(found) if found else Path.cwd() / ".env"


def _looks_like_token(text: str) -> bool:
    return text.startswith("r:") and " " not in text and len(text) > 10


def _wait_for_token(timeout_sec: float = 300, interval_sec: float = 1.0) -> str:
    """Without a terminal to press Enter in, watches the clipboard for the token instead."""
    print("3. Waiting for the token to appear on the clipboard (up to 5 minutes)...", flush=True)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        text = _get_clipboard()
        if _looks_like_token(text):
            return text
        time.sleep(interval_sec)
    return ""


def main() -> None:
    settings = NekoCapSettings()
    already = _get_clipboard()
    if _looks_like_token(already):  # the console step was already done
        _save(already, settings)
        return
    print("1. Sign in to nekocap.com in the browser window that opens (if you aren't already).")
    webbrowser.open("https://nekocap.com")
    if _set_clipboard(CONSOLE_SNIPPET):
        print("2. Press F12, click Console, paste (Ctrl+V) and press Enter.")
        print("   (If Chrome asks, type 'allow pasting' first, then paste again.)")
    else:
        print(f"2. Press F12, click Console, and run:\n   {CONSOLE_SNIPPET}")
    try:
        input("3. Then come back here and press Enter... ")
    except EOFError:  # no keyboard (e.g. started by another program): watch the clipboard instead
        token = _wait_for_token()
    else:
        token = _get_clipboard()
        if not _looks_like_token(token):
            token = getpass.getpass("Couldn't read it from the clipboard. Paste the token here (hidden): ").strip()
    if not token:
        sys.exit("No token given; nothing was saved.")
    _save(token, settings)


def _save(token: str, settings: NekoCapSettings) -> None:
    """Checks the token with NekoCap, then saves it to .env and clears the clipboard."""
    try:
        user = NekoCapClient(token, settings.server_url, settings.app_id).user_id()
    except BaberuError as error:
        sys.exit(f"That token didn't work: {error}")

    env_file = _env_file()
    env_file.touch(exist_ok=True)
    dotenv.set_key(str(env_file), ENV_KEY, token, quote_mode="never")
    _set_clipboard("")  # don't leave the token lying on the clipboard
    print(f"Signed in to NekoCap (user {user}). Saved {ENV_KEY} in {env_file}.")


if __name__ == "__main__":
    main()
