"""Uploading to NekoCap. A fake Parse server stands in for nekocap.com."""
import json

import httpx
import pytest

from baberu.errors import BaberuError
from baberu.publish import nekocap
from baberu.publish.lzstring import compress_to_base64
from conftest import run_cli

SERVER = "https://neko.test/parse"
VIDEO = {"id": "ke0D9v1-P-w", "title": "ついに『とくべチューハイ』が完成！",
         "webpage_url": "https://www.youtube.com/watch?v=ke0D9v1-P-w", "channel_id": "UC1"}


# --- lz-string, checked against the JavaScript library (lz-string 1.5.0) ---

@pytest.mark.parametrize("text,expected", [
    ("a", "IZA="),
    ("ab", "IYIyA==="),
    ("🥂🥂🥂x", "nwbiFd2gPI=="),   # characters outside the BMP are two UTF-16 units in JavaScript
    ("", "Q==="),
])
def test_compression_matches_javascript(text, expected):
    assert compress_to_base64(text) == expected


# --- the client ---

class FakeParse:
    """Answers like a Parse server, and keeps what it was sent."""

    def __init__(self, captions=None, user="me1", fail=None):
        self.captions = captions or []
        self.user = user
        self.fail = fail
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        name = request.url.path.rsplit("/", 1)[-1]
        self.calls.append((name, body, dict(request.headers)))
        if self.fail:
            return httpx.Response(400, json=self.fail)
        if name == "me":
            return httpx.Response(200, json={"objectId": self.user})
        if name == "findCaptions":
            return httpx.Response(200, json={"result": self.captions})
        if name == "submitCaption":
            return httpx.Response(200, json={"result": {"status": "success", "captionId": "new1"}})
        if name == "updateCaption":
            return httpx.Response(200, json={"result": {"status": "success"}})
        return httpx.Response(404, json={"code": 141, "error": "unknown function"})

    def sent(self, name):
        return [body for call, body, _ in self.calls if call == name]


def client(fake, token="tok"):
    return nekocap.NekoCapClient(token, SERVER, "app1", transport=httpx.MockTransport(fake))


def test_a_new_caption_is_submitted_like_the_extension_does():
    fake = FakeParse()

    caption_id = client(fake).upload("[Script Info]\n🥂", "vid1", "原題", "Title", "en", "ja", "unlisted")

    body = fake.sent("submitCaption")[0]
    assert caption_id == "new1"
    assert body["rawCaption"] == {"type": "ass", "data": compress_to_base64("[Script Info]\n🥂")}
    assert body["caption"]["data"] == {"tracks": []} and body["caption"]["videoId"] == "vid1"
    assert body["caption"]["languageCode"] == "en" and body["caption"]["translatedTitle"] == "Title"
    assert body["video"] == {"id": "vid1", "source": 0, "name": "原題", "languageCode": "ja"}
    assert body["privacy"] == 1
    headers = fake.calls[0][2]
    assert headers["x-parse-application-id"] == "app1" and headers["x-parse-session-token"] == "tok"


def test_an_existing_caption_is_replaced_keeping_its_tags():
    fake = FakeParse()
    existing = nekocap.ExistingCaption("old1", ["idol"])

    assert client(fake).upload("x", "vid1", "原題", "Title", "en", "ja", existing=existing) == "old1"

    body = fake.sent("updateCaption")[0]
    assert body["captionId"] == "old1" and body["selectedTags"] == ["idol"] and body["privacy"] == 0
    assert not fake.sent("submitCaption")


def test_only_the_users_own_caption_in_that_language_is_found():
    fake = FakeParse(captions=[
        {"id": "a", "captionerId": "someone", "languageCode": "en"},
        {"id": "b", "captionerId": "me1", "languageCode": "es"},
        {"id": "c", "captionerId": "me1", "languageCode": "en", "tags": ["t"]},
    ])

    assert client(fake).find_own_caption("vid1", "en") == nekocap.ExistingCaption("c", ["t"])
    assert client(FakeParse(captions=[])).find_own_caption("vid1", "en") is None


def test_an_expired_login_says_how_to_get_a_new_one():
    fake = FakeParse(fail={"code": 209, "error": "Invalid session token"})

    with pytest.raises(BaberuError, match="NEKOCAP_SESSION_TOKEN"):
        client(fake).user_id()


def test_a_missing_token_is_refused_before_any_request():
    with pytest.raises(BaberuError, match="NEKOCAP_SESSION_TOKEN"):
        nekocap.NekoCapClient("", SERVER, "app1")


def test_a_rejected_upload_is_an_error():
    fake = FakeParse()
    fake_submit = fake.__call__

    def reject(request):
        if request.url.path.endswith("submitCaption"):
            return httpx.Response(200, json={"result": {"status": "error", "error": "too many captions"}})
        return fake_submit(request)

    with pytest.raises(BaberuError, match="too many captions"):
        nekocap.NekoCapClient("tok", SERVER, "app1", transport=httpx.MockTransport(reject)).upload(
            "x", "vid1", "t", "t", "en", "ja")


# --- the --nekocap stage ---

@pytest.fixture
def neko(workspace, monkeypatch):
    fake = FakeParse()
    monkeypatch.setenv("NEKOCAP_SESSION_TOKEN", "tok")
    monkeypatch.setattr("baberu.publish.publish.client", lambda settings: client(fake))
    (workspace / "video.info.json").write_text(json.dumps(VIDEO, ensure_ascii=False), encoding="utf-8")
    return fake


def test_nekocap_uploads_the_translation(neko, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "--nekocap", "--context", workspace / "ctx.txt")

    body = neko.sent("submitCaption")[0]
    published = (workspace / "video.en.ass").read_text(encoding="utf-8-sig")
    assert body["rawCaption"]["data"] == compress_to_base64(published)
    assert body["video"]["id"] == VIDEO["id"] and body["privacy"] == 0
    assert body["video"]["languageCode"] == "ja_JP"   # as the extension sends it
    record = json.loads(next(workspace.glob("*.baberu/en/nekocap.json")).read_text(encoding="utf-8"))
    assert record["captionId"] == "new1"


def test_an_unchanged_translation_is_not_uploaded_again(neko, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "--nekocap", "--context", workspace / "ctx.txt")
    neko.captions = [{"id": "new1", "captionerId": "me1", "languageCode": "en", "tags": []}]   # now on the site
    run_cli(monkeypatch, workspace / "video.ass", "--nekocap", "--context", workspace / "ctx.txt")

    assert len(neko.sent("submitCaption")) == 1 and not neko.sent("updateCaption")


def test_a_caption_deleted_on_the_site_is_uploaded_again(neko, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "--nekocap", "--context", workspace / "ctx.txt")
    # findCaptions does not list the first upload, as when it was deleted on nekocap.com.
    run_cli(monkeypatch, workspace / "video.ass", "--nekocap", "--context", workspace / "ctx.txt")

    assert len(neko.sent("submitCaption")) == 2


def test_a_changed_translation_replaces_the_upload(neko, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "--nekocap", "--context", workspace / "ctx.txt")
    neko.captions = [{"id": "new1", "captionerId": "me1", "languageCode": "en", "tags": ["idol"]}]
    run_cli(monkeypatch, workspace / "video.ass", "--nekocap-unlisted", "--context", workspace / "ctx.txt")

    update = neko.sent("updateCaption")[0]
    assert update["captionId"] == "new1" and update["selectedTags"] == ["idol"] and update["privacy"] == 1


def test_nekocap_needs_a_youtube_video(workspace, monkeypatch, capsys):
    monkeypatch.setenv("NEKOCAP_SESSION_TOKEN", "tok")

    with pytest.raises(SystemExit):
        run_cli(monkeypatch, workspace / "video.ass", "--nekocap", "--context", workspace / "ctx.txt")

    assert "needs a YouTube video" in capsys.readouterr().err


def test_a_dry_run_checks_the_login(neko, workspace, monkeypatch, capsys):
    run_cli(monkeypatch, workspace / "video.ass", "--nekocap", "--context", workspace / "ctx.txt", "--dry-run")

    assert [name for name, _, _ in neko.calls] == ["me"]
    assert "nekocap" in capsys.readouterr().err


# --- baberu publish ---

def test_publish_uploads_the_result_as_it_is(neko, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "video.ass", "-t", "-p", "--context", workspace / "ctx.txt")
    result = workspace / "video.en.ass"
    original = result.read_text(encoding="utf-8-sig")
    edited = original.replace("T:line2", "Setlog")
    assert edited != original
    result.write_text(edited, encoding="utf-8")
    translation_calls = len(fake_llms["main"].calls)

    run_cli(monkeypatch, "publish", result)

    body = neko.sent("submitCaption")[0]
    assert body["rawCaption"]["data"] == compress_to_base64(edited)
    assert result.read_text(encoding="utf-8") == edited                      # nothing rebuilt
    assert len(fake_llms["main"].calls) == translation_calls + 1            # only the title


def test_publish_translates_the_title_with_the_jobs_notes(neko, workspace, monkeypatch, fake_llms):
    (workspace / "ctx.txt").write_text("Glossary: みりにゃ = Mirinya", encoding="utf-8")
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")

    run_cli(monkeypatch, "publish", workspace / "video.en.ass")

    title_prompt = fake_llms["main"].calls[-1]
    assert VIDEO["title"] in title_prompt and "みりにゃ = Mirinya" in title_prompt


def test_publish_title_is_kept_and_an_unchanged_upload_is_skipped(neko, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")
    run_cli(monkeypatch, "publish", workspace / "video.en.ass", "--title", "Mirinya, First Time at Hidakaya")
    neko.captions = [{"id": "new1", "captionerId": "me1", "languageCode": "en", "tags": []}]
    calls = len(fake_llms["main"].calls)

    run_cli(monkeypatch, "publish", workspace / "video.en.ass")

    assert neko.sent("submitCaption")[0]["caption"]["translatedTitle"] == "Mirinya, First Time at Hidakaya"
    assert not neko.sent("updateCaption") and len(fake_llms["main"].calls) == calls


def test_publish_retitle_replaces_the_title(neko, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")
    run_cli(monkeypatch, "publish", workspace / "video.en.ass", "--title", "Miri-nya")
    neko.captions = [{"id": "new1", "captionerId": "me1", "languageCode": "en", "tags": []}]

    run_cli(monkeypatch, "publish", workspace / "video.en.ass", "--retitle")

    assert neko.sent("updateCaption")[0]["translatedTitle"] != "Miri-nya"


def test_publish_dry_run_uploads_nothing(neko, workspace, monkeypatch, capsys):
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")
    capsys.readouterr()

    run_cli(monkeypatch, "publish", workspace / "video.en.ass", "--dry-run", "--unlisted")

    assert not neko.sent("submitCaption")
    err = capsys.readouterr().err
    assert "Would upload (unlisted)" in err and "Title:" in err


def test_publish_needs_a_result(neko, workspace, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "publish", workspace / "video.en.ass")
    assert "does not exist" in capsys.readouterr().err


def test_nekocap_alone_is_a_full_run(neko, media, workspace, monkeypatch, capsys):
    (workspace / "show.info.json").write_text(json.dumps(VIDEO, ensure_ascii=False), encoding="utf-8")

    run_cli(monkeypatch, workspace / "show.mp4", "--nekocap", "--dry-run")

    err = capsys.readouterr().err
    assert all(name in err for name in ("transcribe", "fix", "translate", "pad", "nekocap"))


# --- baberu-nekocap-login ---

@pytest.fixture
def login(tmp_path, monkeypatch):
    from baberu.publish import login as login_module

    clipboard = {"text": ""}
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=keep-me\n", encoding="utf-8")
    monkeypatch.setattr(login_module.webbrowser, "open", lambda url: None)
    monkeypatch.setattr(login_module, "_set_clipboard", lambda text: clipboard.update(text=text) or True)
    monkeypatch.setattr(login_module, "_get_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(login_module, "_env_file", lambda: env)
    monkeypatch.setattr("builtins.input", lambda prompt="": clipboard.update(text="r:token123456") or "")
    return login_module, env, clipboard


def test_login_saves_a_working_token(login, monkeypatch):
    module, env, clipboard = login
    fake = FakeParse()
    monkeypatch.setattr(module, "NekoCapClient", lambda token, *a: client(fake, token))

    module.main()

    assert "NEKOCAP_SESSION_TOKEN=r:token123456" in env.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=keep-me" in env.read_text(encoding="utf-8")
    assert fake.calls[0][2]["x-parse-session-token"] == "r:token123456"
    assert clipboard["text"] == ""   # cleared afterwards


def test_login_refuses_a_token_that_does_not_work(login, monkeypatch):
    module, env, _ = login
    fake = FakeParse(fail={"code": 209, "error": "Invalid session token"})
    monkeypatch.setattr(module, "NekoCapClient", lambda token, *a: client(fake, token))

    with pytest.raises(SystemExit):
        module.main()

    assert "NEKOCAP_SESSION_TOKEN" not in env.read_text(encoding="utf-8")


def test_without_a_terminal_login_waits_for_the_clipboard(login, monkeypatch):
    module, env, clipboard = login
    fake = FakeParse()
    monkeypatch.setattr(module, "NekoCapClient", lambda token, *a: client(fake, token))
    def no_keyboard(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_keyboard)
    reads = iter(["", "copy(...)", "copy(...)", "r:token999999"])
    monkeypatch.setattr(module, "_get_clipboard", lambda: next(reads))
    monkeypatch.setattr(module.time, "sleep", lambda s: None)

    module.main()

    assert "NEKOCAP_SESSION_TOKEN=r:token999999" in env.read_text(encoding="utf-8")


def test_a_token_already_on_the_clipboard_is_used_straight_away(login, monkeypatch):
    module, env, clipboard = login
    fake = FakeParse()
    monkeypatch.setattr(module, "NekoCapClient", lambda token, *a: client(fake, token))
    clipboard["text"] = "r:already12345"
    monkeypatch.setattr(module.webbrowser, "open", lambda url: pytest.fail("shouldn't open the browser"))

    module.main()

    assert "NEKOCAP_SESSION_TOKEN=r:already12345" in env.read_text(encoding="utf-8")
