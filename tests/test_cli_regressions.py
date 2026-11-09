"""Drives the real command-line entry point with a fake LLM."""
import importlib.resources
import shutil
import subprocess
import sys

import pytest
import yaml

import baberu.__main__ as cli
from baberu.subtitling import sub_utils
from baberu.tools import av_utils
from baberu.transcription import transcript_segmented
from conftest import FakeLLM, make_subs, numbered_subs, translate_all


@pytest.fixture
def workspace(tmp_path, monkeypatch, fake_llms):
    text = importlib.resources.files("baberu.defaults").joinpath("default_config.yaml").read_text("utf-8")
    config = yaml.safe_load(text)
    config["translation"].update(confirm_auto_context=False, websearch_model="main", fallback_model="main")
    config["logging"]["log_to_file"] = False
    monkeypatch.setattr(cli.config_setup, "load_config", lambda *a: config)
    monkeypatch.setattr(cli.logging_setup, "setup_logging", lambda **k: None)
    monkeypatch.delenv("BABERU_DIR", raising=False)
    monkeypatch.delenv("BABERU_LOG_DIR", raising=False)
    fake_llms["main"] = FakeLLM(translate_all)
    (tmp_path / "ctx.txt").write_text("context", encoding="utf-8")
    sub_utils.write(numbered_subs(6), tmp_path / "video.ass")
    return tmp_path


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["baberu", *map(str, argv)])
    cli.main()


def texts(path):
    return [e.text for e in sub_utils.load(path).events]


def test_each_line_range_gets_its_own_result(workspace, monkeypatch):
    source, ctx = workspace / "video.ass", workspace / "ctx.txt"

    run(monkeypatch, source, "-t", ctx, "--model", "main", "--lines", "1-2")
    run(monkeypatch, source, "-t", ctx, "--model", "main", "--lines", "3-4")

    second = texts(workspace / "video.tr_custom.L3-4.ass")
    assert second[2:4] == ["T:line2", "T:line3"]


def test_output_is_named_for_the_target_language(workspace, monkeypatch):
    run(monkeypatch, workspace / "video.ass", "-t", workspace / "ctx.txt", "--model", "main", "--lang-to", "fr")

    assert (workspace / "video.fr.ass").exists()
    assert not (workspace / "video.en.ass").exists()


def test_english_output_keeps_its_existing_name(workspace, monkeypatch):
    run(monkeypatch, workspace / "video.ass", "-t", workspace / "ctx.txt", "--model", "main")

    assert (workspace / "video.en.ass").exists()


def test_directory_option_applies_to_absolute_input(workspace, monkeypatch):
    out = workspace / "out"

    run(monkeypatch, workspace / "video.ass", "-f", "-d", out)

    assert (out / "video.fixed.ass").exists()
    assert not (workspace / "video.fixed.ass").exists()


def test_existing_work_beside_the_input_is_still_reused(workspace, monkeypatch):
    sub_utils.write(make_subs([(0, 5000, "previously fixed")]), workspace / "video.fixed.ass")
    out = workspace / "out"

    run(monkeypatch, workspace / "video.ass", "-f", "-p", "-d", out)

    assert not (out / "video.fixed.ass").exists()
    assert texts(out / "video.padded.ass") == ["previously fixed"]


def test_relative_input_in_a_subfolder(workspace, monkeypatch):
    (workspace / "media").mkdir()
    shutil.move(workspace / "video.ass", workspace / "media" / "video.ass")
    monkeypatch.chdir(workspace)

    run(monkeypatch, "media/video.ass", "-f")

    assert (workspace / "media" / "video.fixed.ass").exists()


def test_resuming_from_cached_fix_keeps_line_range_correct(workspace, monkeypatch):
    rows = [(i, i + 1, f"s{i}") for i in range(4)] + [(100 + i * 20000, 100 + i * 20000 + 15000, f"long{i}") for i in range(6)]
    sub_utils.write(make_subs(rows), workspace / "video.ass")
    args = (workspace / "video.ass", "-f", "-p", "--lines", "1-10")

    run(monkeypatch, *args)
    (workspace / "video.padded_custom.L1-10.ass").unlink()
    run(monkeypatch, *args)

    assert (workspace / "video.padded_custom.L1-10.ass").exists()


def test_autopilot_honours_an_explicit_audio_file(workspace, monkeypatch):
    audio = workspace / "speech.m4a"
    audio.touch()
    calls = []
    monkeypatch.setattr(transcript_segmented, "find_segments", lambda *a, **k: calls.append(1) or [])

    run(monkeypatch, workspace / "video.ass", "-A", "-r", audio, "--model", "main")

    assert calls == [1]


def test_retranscribe_with_nothing_to_do_still_produces_its_file(workspace, monkeypatch):
    audio = workspace / "speech.m4a"
    audio.touch()
    monkeypatch.setattr(transcript_segmented, "find_segments", lambda *a, **k: [])

    run(monkeypatch, workspace / "video.ass", "-r", audio)

    assert (workspace / "video.2pass.ass").exists()


def test_hardcode_works_without_an_output_option(workspace, monkeypatch):
    video = workspace / "clip.mp4"
    video.touch()
    calls = []
    monkeypatch.setattr(av_utils, "hardcode_subtitles", lambda v, s, o: calls.append(o))

    run(monkeypatch, workspace / "video.ass", "--hardcode", video)

    assert calls == [workspace / "clip_subbed.mp4"]


def test_hardcode_may_name_the_input_video(workspace, monkeypatch):
    video = workspace / "clip.mp4"
    video.touch()
    monkeypatch.setattr(av_utils, "hardcode_subtitles", lambda v, s, o: None)

    run(monkeypatch, video, "--hardcode", video)


@pytest.fixture
def media(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    folder = tmp_path / "my [media] folder"
    folder.mkdir()
    video = folder / "clip.mp4"
    # Audio is deliberately the first stream.
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
         "-f", "lavfi", "-i", "anullsrc=d=1", "-map", "1:a", "-map", "0:v", "-shortest", str(video)],
        check=True,
    )
    sub_utils.write(make_subs([(0, 900, "hi")]), folder / "subs.ass")
    return folder


def test_hardcode_with_relative_subtitle_path(media, monkeypatch):
    monkeypatch.chdir(media)
    from pathlib import Path

    av_utils.hardcode_subtitles(Path("clip.mp4"), Path("subs.ass"), Path("out.mp4"))

    assert (media / "out.mp4").stat().st_size > 0


def test_hardcode_with_absolute_subtitle_path(media, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    av_utils.hardcode_subtitles(media / "clip.mp4", media / "subs.ass", media / "out.mp4")

    assert (media / "out.mp4").stat().st_size > 0
