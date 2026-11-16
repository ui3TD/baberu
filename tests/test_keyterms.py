"""Keyterms: the names and jargon sent to the transcription service."""
import json

import pytest
import yaml
from pydantic import ValidationError

from baberu.errors import BaberuError
from baberu.providers.factory import AIToolFactory
from baberu.providers.llm.openrouter import OpenRouterProvider
from baberu.settings import KeytermSettings
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


def write_channel(folder, name, terms, channel_id=VIDEO["channel_id"], **extra):
    data = {"channel_id": channel_id, "channel": VIDEO["channel"], "terms": terms, **extra}
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_a_channel_file_gives_its_terms(tmp_path):
    write_channel(tmp_path, "ikonoijoy", ["大谷映美里", "＝LOVE"])

    path = keyterms.channel_file(VIDEO, {VIDEO["channel_id"]: "ikonoijoy"}, tmp_path)
    assert keyterms.load_channel(path, VIDEO["channel_id"]).terms == ["大谷映美里", "＝LOVE"]


def test_a_channel_with_no_mapping_or_file_is_missing(tmp_path):
    with pytest.raises(keyterms.MissingChannelTerms, match=f"Add `{VIDEO['channel_id']}: <name>`"):
        keyterms.channel_file(VIDEO, {}, tmp_path)
    with pytest.raises(keyterms.MissingChannelTerms, match="nope.yaml"):
        keyterms.channel_file(VIDEO, {VIDEO["channel_id"]: "nope"}, tmp_path)


@pytest.mark.parametrize("extra, terms", [({"note": "x"}, ["a"]), ({}, ["a", {"term": "b"}]), ({}, "a")])
def test_a_channel_file_holds_only_a_plain_list_of_terms(tmp_path, extra, terms):
    path = write_channel(tmp_path, "bad", terms, **extra)

    with pytest.raises(BaberuError, match="not valid"):
        keyterms.load_channel(path)


def test_a_channel_file_is_for_one_channel(tmp_path):
    path = write_channel(tmp_path, "other", ["a"], channel_id="UCother")

    with pytest.raises(BaberuError, match="UCother"):
        keyterms.load_channel(path, VIDEO["channel_id"])


def test_a_channel_maps_to_one_file_not_a_list():
    with pytest.raises(ValidationError, match="one file"):
        KeytermSettings(channels={VIDEO["channel_id"]: ["ikonoijoy", "otani_emiri"]})


def test_reply_parsing_removes_bullets_and_numbering():
    reply = "Keyterms:\n1. 大谷映美里\n- とくべチューハイ\n* `アサヒ空想開発局`\n\n"

    assert keyterms.parse_reply(reply) == ["大谷映美里", "とくべチューハイ", "アサヒ空想開発局"]


def test_build_puts_the_models_picks_first():
    llm = FakeLLM(lambda prompt: "大谷映美里\nアサヒ空想開発局")

    terms = keyterms.build(VIDEO, ["野口衣織", "大谷映美里"], 100, llm, "Japanese")

    assert terms[:3] == ["大谷映美里", "アサヒ空想開発局", "野口衣織"]
    assert "とくべチューハイ" in terms and terms.count("大谷映美里") == 1
    assert "野口衣織" in llm.calls[0] and "出演メンバー" in llm.calls[0]


def test_build_without_a_model_uses_the_channels_terms_then_tags():
    assert keyterms.build(VIDEO, ["野口衣織"], 3) == ["野口衣織", "とくべチューハイ", "乾杯"]


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
    terms_dir = workspace / "keyterms"
    terms_dir.mkdir()
    write_channel(terms_dir, "ikonoijoy", ["野口衣織"])

    config = config_setup.load_config()
    config["transcription"]["model"] = "scribe_v3"
    config["keyterms"].update(dir=str(terms_dir), llm_model="picker", confirm=False,
                              channels={VIDEO["channel_id"]: "ikonoijoy"})
    fake_llms["picker"] = FakeLLM(lambda prompt: "アサヒ空想開発局")
    return seen


def test_keyterms_review_without_a_keyboard_carries_on(scribe_run, workspace, monkeypatch):
    config_setup.load_config()["keyterms"]["confirm"] = True
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)

    def end_of_input(*args):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_of_input)

    run_cli(monkeypatch, workspace / "show.m4a", "-s")

    assert "野口衣織" in scribe_run[-1]["keyterms"]


def test_transcription_gets_the_jobs_keyterms(scribe_run, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.m4a", "-s")

    job = workspace / "show.baberu"
    sent = scribe_run[-1]["keyterms"]
    assert sent[:2] == ["アサヒ空想開発局", "野口衣織"] and "大谷映美里" in sent
    assert keyterms.load(job / "keyterms.txt", 100) == sent
    assert json.loads((job / "source.json").read_text(encoding="utf-8"))["id"] == VIDEO["id"]


def test_an_edited_list_is_reused(scribe_run, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "show.m4a", "-s")
    (workspace / "show.baberu" / "keyterms.txt").write_text("えみりー\n", encoding="utf-8")

    run_cli(monkeypatch, workspace / "show.m4a", "-s", "--redo", "transcribe")

    assert scribe_run[-1]["keyterms"] == ["えみりー"]
    assert len(fake_llms["picker"].calls) == 1


def test_a_channel_with_no_file_stops_before_transcribing(scribe_run, workspace, monkeypatch, capsys, fake_llms):
    config_setup.load_config()["keyterms"]["channels"] = {}

    with pytest.raises(SystemExit):
        run_cli(monkeypatch, workspace / "show.m4a", "-s")

    error = capsys.readouterr().err
    assert f"Add `{VIDEO['channel_id']}: <name>`" in error and "Traceback" not in error
    assert not (workspace / "show.baberu" / "transcript.json").exists() and not fake_llms["picker"].calls


def test_a_missing_file_can_be_allowed(scribe_run, workspace, monkeypatch):
    config_setup.load_config()["keyterms"].update(channels={}, require_channel_file=False)

    run_cli(monkeypatch, workspace / "show.m4a", "-s")

    sent = scribe_run[-1]["keyterms"]
    assert "アサヒ空想開発局" in sent and "野口衣織" not in sent


def test_a_changed_channel_file_makes_the_keyterms_again(scribe_run, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "show.m4a", "-s")
    write_channel(workspace / "keyterms", "ikonoijoy", ["佐々木舞香"])

    run_cli(monkeypatch, workspace / "show.m4a", "-s", "--redo", "transcribe")

    assert "佐々木舞香" in scribe_run[-1]["keyterms"] and "野口衣織" not in scribe_run[-1]["keyterms"]
    assert len(fake_llms["picker"].calls) == 2


def test_redo_keyterms_makes_them_again(scribe_run, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "show.m4a", "-s")

    run_cli(monkeypatch, workspace / "show.m4a", "-s", "--redo", "keyterms", "--redo", "transcribe")

    assert len(fake_llms["picker"].calls) == 2


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
