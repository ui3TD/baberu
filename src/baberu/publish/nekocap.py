"""Uploads captions to NekoCap (nekocap.com) under the user's account.

NekoCap's backend is a Parse Server; captions are created and replaced with its
`submitCaption` and `updateCaption` cloud functions. NekoCap signs in through
the browser (Google, through Firebase); the client uses that browser's Parse
session token.

An .ass file goes up as-is: NekoCap keeps the raw file, compressed with
lz-string, and renders it with its ASS renderer. An empty track list is sent
alongside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from baberu.errors import BaberuError
from baberu.publish.lzstring import compress_to_base64

YOUTUBE = 0  # NekoCap's VideoSource.Youtube
PRIVACY = {"public": 0, "unlisted": 1}  # NekoCap's CaptionPrivacy
INVALID_SESSION = 209  # Parse's error code for an unknown or expired session token
TOKEN_HELP = ("Sign in at nekocap.com, open the browser console (F12) and run "
              "JSON.parse(localStorage['Parse/nekocap042/currentUser']).sessionToken, "
              "then put it in .env as NEKOCAP_SESSION_TOKEN.")


@dataclass(frozen=True)
class ExistingCaption:
    id: str
    tags: list[str] = field(default_factory=list)


def caption_url(caption_id: str) -> str:
    return f"https://nekocap.com/view/{caption_id}"


class NekoCapClient:
    def __init__(self, session_token: str, server_url: str, app_id: str,
                 transport: httpx.BaseTransport | None = None, timeout: float = 120.0):
        if not session_token:
            raise BaberuError(f"NekoCap needs your login. {TOKEN_HELP}")
        self._http = httpx.Client(
            base_url=server_url.rstrip("/") + "/",
            headers={"X-Parse-Application-Id": app_id, "X-Parse-Session-Token": session_token},
            transport=transport, timeout=timeout)
        self._user_id: str | None = None

    def _request(self, method: str, path: str, payload: dict | None = None) -> Any:
        try:
            response = self._http.request(method, path, json=payload)
        except httpx.HTTPError as error:
            raise BaberuError(f"Couldn't reach NekoCap: {error}") from error
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.is_error or "error" in data:
            if data.get("code") == INVALID_SESSION:
                raise BaberuError(f"Your NekoCap login has expired or is wrong. {TOKEN_HELP}")
            message = data.get("error") or response.text[:200] or response.reason_phrase
            raise BaberuError(f"NekoCap refused {path}: {message}")
        return data

    def run(self, function: str, params: dict) -> Any:
        """Calls a Parse cloud function and returns its result."""
        return self._request("POST", f"functions/{function}", params).get("result")

    def user_id(self) -> str:
        """Who the token belongs to. Changes nothing; also used as the login check."""
        if self._user_id is None:
            self._user_id = self._request("GET", "users/me")["objectId"]
        return self._user_id

    def find_own_caption(self, video_id: str, language: str) -> ExistingCaption | None:
        """The user's caption in this language for a YouTube video, if there is one."""
        captions = self.run("findCaptions", {"videoId": video_id, "videoSource": YOUTUBE}) or []
        if isinstance(captions, dict):
            captions = captions.get("captions") or []
        me = self.user_id()
        for caption in captions:
            if caption.get("captionerId") == me and caption.get("languageCode") == language:
                return ExistingCaption(caption["id"], list(caption.get("tags") or []))
        return None

    def upload(self,
               ass_text: str,
               video_id: str,
               original_title: str,
               translated_title: str,
               language: str,
               source_language: str,
               privacy: str = "public",
               existing: ExistingCaption | None = None) -> str:
        """Creates the caption, or replaces `existing`. Returns the caption's id."""
        raw_caption = {"type": "ass", "data": compress_to_base64(ass_text)}
        if existing:
            result = self.run("updateCaption", {
                "captionId": existing.id,
                "captionData": {"tracks": []},
                "rawCaption": raw_caption,
                "hasAudioDescription": False,
                "translatedTitle": translated_title,
                "selectedTags": existing.tags,
                "privacy": PRIVACY[privacy],
            })
            _check(result, "update")
            return existing.id

        result = self.run("submitCaption", {
            "caption": {
                "videoId": video_id,
                "videoSource": YOUTUBE,
                "loadedByUser": True,
                "data": {"tracks": []},
                "languageCode": language,
                "translatedTitle": translated_title,
                "userLike": None,
                "userDislike": None,
            },
            "rawCaption": raw_caption,
            "video": {"id": video_id, "source": YOUTUBE, "name": original_title, "languageCode": source_language},
            "hasAudioDescription": False,
            "privacy": PRIVACY[privacy],
        })
        _check(result, "upload")
        return result["captionId"]


def _check(result: Any, action: str) -> None:
    if not isinstance(result, dict) or result.get("status") != "success":
        error = result.get("error") if isinstance(result, dict) else result
        raise BaberuError(f"NekoCap rejected the {action}: {error or 'no reason given'}")
