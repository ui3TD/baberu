"""Keyterms: the names and jargon sent to the transcription service."""
import json

import pytest

from baberu.errors import BaberuError
from baberu.providers.factory import AIToolFactory
from baberu.providers.llm.openrouter import OpenRouterProvider
from baberu.setup import config_setup
from baberu.transcription import keyterms
from conftest import FakeLLM, FakeTranscriber, run_cli

# From https://www.youtube.com/watch?v=ke0D9v1-P-w
VIDEO = {
    "id": "ke0D9v1-P-w",
    "title": "ついに『とくべチューハイ』が完成！早速みんなで飲んでみたよ🥂【乾杯】",
    "channel": "イコラブ ノイミー ニアジョイ チャンネル",
    "channel_id": "UCWHh6SdvJWrPuf1oJrHbX3A",
    "tags": ["アイドル", "＝LOVE", "イコラブ", "大谷映美里", "佐々木舞香"],
    "description": "▼出演メンバー\n大谷 映美里 (Emiri Otani)\n佐々木 舞香 (Maika Sasaki)",
    "formats": [{"format_id": "251"}],
}


# --- Cleaning ---

def test_clean_keeps_order_and_drops_duplicates():
    assert keyterms.clean([" ＝LOVE ", "イコラブ", "＝LOVE", "", "イコラブ"]) == ["＝LOVE", "イコラブ"]


def test_clean_drops_what_scribe_would_reject():
    terms = ["ok", "x" * 51, "one two three four five six", "a[b]", "back\\slash", "fine term"]

    assert keyterms.clean(terms) == ["ok", "fine term"]


def test_clean_keeps_only_the_first_max_terms():
    assert keyterms.clean([f"t{i}" for i in range(10)], max_terms=3) == ["t0", "t1", "t2"]


# --- Where terms come from ---

def test_metadata_gives_quoted_title_phrases_and_tags():
    assert keyterms.from_metadata(VIDEO) == ["とくべチューハイ", "乾杯", *VIDEO["tags"]]


def test_metadata_is_trimmed_when_saved(tmp_path):
    path = tmp_path / "source.json"
    keyterms.save_metadata(VIDEO, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "formats" not in saved and saved["channel_id"] == VIDEO["channel_id"]
    assert keyterms.load_metadata(path) == saved
    assert keyterms.load_metadata(tmp_path / "missing.json") == {}


def test_lists_skip_comments_and_blank_lines(tmp_path):
    (tmp_path / "idols.txt").write_text("# heading\n大谷映美里  # Emiri\n\n＝LOVE\n", encoding="utf-8")

    path = keyterms.resolve_list("idols", tmp_path)
    assert keyterms.read_list(path) == ["大谷映美里", "＝LOVE"]


def test_unknown_list_is_an_error(tmp_path):
    with pytest.raises(BaberuError, match="nope"):
        keyterms.resolve_list("nope", tmp_path)


def test_channel_lists_come_before_named_ones():
    channels = {VIDEO["channel_id"]: ["ikonoijoy"]}

    assert keyterms.lists_for(VIDEO, channels, ["food", "ikonoijoy"]) == ["ikonoijoy", "food"]
    assert keyterms.lists_for({}, channels, ["food"]) == ["food"]


def test_reply_parsing_removes_bullets_and_numbering():
    reply = "Keyterms:\n1. 大谷映美里\n- とくべチューハイ\n* `アサヒ空想開発局`\n\n"

    assert keyterms.parse_reply(reply) == ["大谷映美里", "とくべチューハイ", "アサヒ空想開発局"]


def test_build_puts_the_models_picks_first(tmp_path):
    (tmp_path / "group.txt").write_text("野口衣織\n大谷映美里\n", encoding="utf-8")
    llm = FakeLLM(lambda prompt: "大谷映美里\nアサヒ空想開発局")

    terms = keyterms.build(VIDEO, [tmp_path / "group.txt"], 100, llm, "Japanese")

    assert terms[:3] == ["大谷映美里", "アサヒ空想開発局", "野口衣織"]
    assert "とくべチューハイ" in terms and terms.count("大谷映美里") == 1
    assert "野口衣織" in llm.calls[0] and "出演メンバー" in llm.calls[0]


def test_build_without_a_model_uses_lists_then_tags(tmp_path):
    (tmp_path / "group.txt").write_text("野口衣織\n", encoding="utf-8")

    assert keyterms.build(VIDEO, [tmp_path / "group.txt"], 3) == ["野口衣織", "とくべチューハイ", "乾杯"]


# --- OpenRouter ---

def test_openrouter_sends_effort_and_web_search_only_when_asked():
    assert OpenRouterProvider("k", "m").request_body(False) == {"usage": {"include": True}}
    provider = OpenRouterProvider("k", "m", effort="low")
    assert provider.request_body(True) == {"usage": {"include": True}, "reasoning": {"effort": "low"}, "plugins": [{"id": "web"}]}
    assert provider.supports_grounding


# --- In a run ---

@pytest.fixture
def scribe_run(workspace, monkeypatch, fake_llms):
    """show.m4a with a yt-dlp info.json beside it, transcribed by a Scribe model that records its options."""
    seen: list[dict] = []
    transcriber = FakeTranscriber()
    monkeypatch.setattr(AIToolFactory, "get_transcription_provider",
                        staticmethod(lambda model, options=None: seen.append(options) or transcriber))
    monkeypatch.setattr("baberu.tools.av_utils.duration_seconds", lambda path: 60.0)
    (workspace / "show.m4a").write_bytes(b"audio")
    (workspace / "show.info.json").write_text(json.dumps(VIDEO, ensure_ascii=False), encoding="utf-8")
    lists = workspace / "lists"
    lists.mkdir()
    (lists / "ikonoijoy.txt").write_text("野口衣織\n", encoding="utf-8")

    config = config_setup.load_config()
    config["transcription"]["model"] = "scribe_v3"
    config["keyterms"].update(lists_dir=str(lists), llm_model="picker", confirm=False,
                              channels={VIDEO["channel_id"]: ["ikonoijoy"]})
    fake_llms["picker"] = FakeLLM(lambda prompt: "アサヒ空想開発局")
    return seen


def test_transcription_gets_the_jobs_keyterms(scribe_run, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.m4a", "-s")

    job = workspace / "show.baberu"
    sent = scribe_run[-1]["keyterms"]
    assert sent[:2] == ["アサヒ空想開発局", "野口衣織"] and "大谷映美里" in sent
    assert keyterms.read_list(job / "keyterms.txt") == sent
    assert json.loads((job / "source.json").read_text(encoding="utf-8"))["id"] == VIDEO["id"]


def test_an_edited_list_is_reused(scribe_run, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "show.m4a", "-s")
    (workspace / "show.baberu" / "keyterms.txt").write_text("えみりー\n", encoding="utf-8")

    run_cli(monkeypatch, workspace / "show.m4a", "-s", "--redo", "transcribe")

    assert scribe_run[-1]["keyterms"] == ["えみりー"]
    assert len(fake_llms["picker"].calls) == 1


def test_no_keyterms_sends_none(scribe_run, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.m4a", "-s", "--no-keyterms")

    assert "keyterms" not in scribe_run[-1]
    assert not (workspace / "show.baberu" / "keyterms.txt").exists()


def test_a_transcript_made_without_keyterms_is_still_reused(scribe_run, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.m4a", "-s", "--no-keyterms")
    calls = len(scribe_run)

    run_cli(monkeypatch, workspace / "show.m4a", "-s")

    assert len(scribe_run) == calls


def test_models_without_keyterms_skip_the_stage(scribe_run, workspace, monkeypatch, fake_llms):
    config_setup.load_config()["transcription"]["model"] = "whisper-1"

    run_cli(monkeypatch, workspace / "show.m4a", "-s")

    assert "keyterms" not in (scribe_run[-1] or {})
    assert not fake_llms["picker"].calls


def test_translation_notes_see_the_videos_details(scribe_run, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "show.m4a", "-t")

    notes_prompt = next(call for call in fake_llms["main"].calls if "Glossary" in call)
    assert VIDEO["title"] in notes_prompt and "アサヒ空想開発局" in notes_prompt and "most won't appear" in notes_prompt


def test_notes_prompt_hides_the_continuation_marker():
    from baberu.constants import CONTINUE_FLAG
    from baberu.subtitling import translation_prompts
    from conftest import make_subs

    prompt = translation_prompts.context_prompt(make_subs([(0, 1, f"前半{CONTINUE_FLAG}"), (1, 2, "後半。")]), "v", "ja", "en")

    assert CONTINUE_FLAG not in prompt and "前半\n後半。" in prompt
