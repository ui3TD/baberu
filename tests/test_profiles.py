import pytest

from baberu.errors import BaberuError
from baberu.providers.factory import AIToolFactory
from baberu.providers.transcription.elevenlabs import ScribeProvider
from baberu.settings import Profile, Settings
from baberu.setup import config_setup
from conftest import FakeTranscriber, run_cli


def settings_for(model: str, profiles=None) -> Settings:
    data = {"transcription": {"model": model}}
    if profiles is not None:
        data["profiles"] = profiles
    return Settings.model_validate(data)


def test_scribe_v1_keeps_its_old_diarization_threshold_by_default():
    v1 = config_setup.apply_profiles(settings_for("scribe_v1_experimental"), "ja")
    v3 = config_setup.apply_profiles(settings_for("scribe_v3"), "ja")

    assert v1.transcription.options == {"diarization_threshold": 0.1}
    assert v3.transcription.options == {}


def test_profiles_match_model_and_language_and_apply_in_order():
    profiles = [
        {"model": "scribe_v3", "settings": {"parsing": {"soft_max_chars": 30}, "anomalies": {"smear_min_sec": 5.0}}},
        {"model": "scribe", "lang": "en", "settings": {"parsing": {"delimiters": ["."], "soft_max_chars": 40}}},
    ]

    japanese = config_setup.apply_profiles(settings_for("scribe_v3", profiles), "ja")
    english = config_setup.apply_profiles(settings_for("scribe_v3", profiles), "EN")

    assert (japanese.parsing.soft_max_chars, japanese.anomalies.smear_min_sec) == (30, 5.0)
    assert "。" in japanese.parsing.delimiters
    assert (english.parsing.soft_max_chars, english.parsing.delimiters) == (40, ["."])
    assert english.parsing.hard_max_chars == Settings().parsing.hard_max_chars


def test_options_merge_instead_of_replacing():
    profiles = [{"model": "scribe", "settings": {"transcription": {"options": {"no_verbatim": True}}}}]
    settings = Settings.model_validate({"transcription": {"model": "scribe_v3", "options": {"seed": 4}},
                                        "profiles": profiles})

    assert config_setup.apply_profiles(settings, "ja").transcription.options == {"seed": 4, "no_verbatim": True}


def test_a_profile_cannot_change_translation():
    settings = settings_for("scribe_v3", [{"settings": {"translation": {"default_model": "x"}}}])

    with pytest.raises(BaberuError, match="translation"):
        config_setup.apply_profiles(settings, "ja")


def test_profile_matches_by_part_of_the_model_name():
    assert Profile(model="scribe_v1").matches("scribe_v1_experimental", "ja")
    assert not Profile(model="scribe_v1").matches("scribe_v3", "ja")
    assert Profile().matches("anything", "fr")


def test_scribe_sends_options_over_its_defaults():
    provider = ScribeProvider("key", "scribe_v3", {"no_verbatim": True, "diarize": False})

    options = provider.request_options()

    assert options["no_verbatim"] is True and options["diarize"] is False
    assert options["timestamps_granularity"] == "word"
    assert "diarization_threshold" not in options


def test_scribe_request_carries_the_options(monkeypatch, tmp_path):
    provider = ScribeProvider("key", "scribe_v3", {"numbers_format": "digits"})
    sent = {}

    class Response:
        def model_dump_json(self):
            return "{}"

        def model_dump(self):
            return {}

    monkeypatch.setattr(provider.client.speech_to_text, "convert", lambda **kwargs: sent.update(kwargs) or Response())
    audio = tmp_path / "a.ogg"
    audio.write_bytes(b"x")

    provider.transcribe(audio, lang="ja")

    assert sent["numbers_format"] == "digits" and sent["model_id"] == "scribe_v3"
    assert sent["additional_formats"][0].segment_on_silence_longer_than_s == 2.0


def test_run_passes_options_and_records_them(workspace, monkeypatch):
    seen = []
    transcriber = FakeTranscriber()
    monkeypatch.setattr(AIToolFactory, "get_transcription_provider",
                        staticmethod(lambda model, options=None: seen.append(options) or transcriber))
    monkeypatch.setattr("baberu.tools.av_utils.duration_seconds", lambda path: 60.0)
    (workspace / "show.m4a").write_bytes(b"audio")
    config_setup.load_config()["transcription"]["options"] = {"seed": 7}

    run_cli(monkeypatch, workspace / "show.m4a", "-s")

    assert seen[-1] == {"seed": 7}


def test_changed_options_stop_before_reusing(workspace, monkeypatch, capsys):
    transcriber = FakeTranscriber()
    transcriber.request_options = lambda: {"seed": 7}
    monkeypatch.setattr(AIToolFactory, "get_transcription_provider", staticmethod(lambda model, options=None: transcriber))
    monkeypatch.setattr("baberu.tools.av_utils.duration_seconds", lambda path: 60.0)
    (workspace / "show.m4a").write_bytes(b"audio")
    config = config_setup.load_config()
    config["transcription"]["options"] = {"seed": 7}
    run_cli(monkeypatch, workspace / "show.m4a", "-s")
    capsys.readouterr()

    config["transcription"]["options"] = {"seed": 8}
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, workspace / "show.m4a", "-s")

    assert "different seed" in capsys.readouterr().err and len(transcriber.calls) == 1


def test_a_dry_run_reports_changed_options(workspace, monkeypatch, capsys):
    transcriber = FakeTranscriber()
    transcriber.request_options = lambda: {"seed": 7}
    monkeypatch.setattr(AIToolFactory, "get_transcription_provider", staticmethod(lambda model, options=None: transcriber))
    monkeypatch.setattr("baberu.tools.av_utils.duration_seconds", lambda path: 60.0)
    (workspace / "show.m4a").write_bytes(b"audio")
    config = config_setup.load_config()
    config["transcription"]["options"] = {"seed": 7}
    run_cli(monkeypatch, workspace / "show.m4a", "-s")
    capsys.readouterr()

    config["transcription"]["options"] = {"seed": 8}
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, workspace / "show.m4a", "-c", "--dry-run")

    assert "different seed" in capsys.readouterr().err
