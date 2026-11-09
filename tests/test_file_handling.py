"""How baberu behaves from the command line: which files it reads, writes and refuses."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from baberu.job import Job, enclosing_job
from baberu import legacy
from baberu.subtitling import sub_utils
from baberu.tools import av_utils
from baberu.tools.file_utils import FileKind, atomic_path, kind_of
from baberu.transcription import transcript_segmented
from baberu.providers.factory import AIToolFactory
from conftest import FakeLLM, make_subs, numbered_subs, run_cli, texts, translate_all


def fails(monkeypatch, capsys, *argv) -> str:
    """Runs a command expected to be refused, returning what it said."""
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, *argv)
    assert exit_info.value.code not in (0, None)
    error = capsys.readouterr().err
    assert "baberu: error:" in error
    assert "Traceback" not in error
    return error


# --- Doing the job ---

def test_no_flags_runs_everything_and_leaves_one_result(media, workspace, monkeypatch, capsys):
    run_cli(monkeypatch, workspace / "show.mp4")

    assert texts(workspace / "show.en.ass") == ["T:hello"]
    assert sorted(p.name for p in workspace.iterdir()) == [
        "ctx.txt", "show.baberu", "show.en.ass", "show.mp4", "video.ass",
    ]
    assert capsys.readouterr().out.strip() == str(workspace / "show.en.ass")


def test_work_folder_layout(media, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.mp4")

    job = workspace / "show.baberu"
    for name in ("manifest.json", "audio.m4a", "transcript.json", "raw.ass", "2pass.ass", "fixed.ass",
                 "en/context.txt", "en/translated.ass", "en/padded.ass"):
        assert (job / name).exists(), name


def test_asking_for_translation_runs_what_it_needs(media, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.mp4", "-t", "--context", workspace / "ctx.txt")

    assert texts(workspace / "show.en.ass") == ["T:hello"]
    assert len(media.calls) == 1
    assert not (workspace / "show.baberu" / "fixed.ass").exists()


def test_stages_not_asked_for_do_not_run(media, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.mp4", "-c")

    assert texts(workspace / "show.ja.ass") == ["hello"]
    assert not (workspace / "show.baberu" / "en").exists()


def test_impossible_request_is_refused(workspace, monkeypatch, capsys):
    assert "subtitle" in fails(monkeypatch, capsys, workspace / "video.ass", "-x").lower()


def test_retranscribe_without_audio_is_refused(workspace, monkeypatch, capsys):
    assert "audio" in fails(monkeypatch, capsys, workspace / "video.ass", "-r").lower()


def test_full_run_on_subtitles_skips_what_does_not_apply(workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass")

    assert texts(workspace / "video.en.ass")[0] == "T:line0"


def test_retranscribe_finds_audio_beside_the_input(workspace, monkeypatch, media):
    (workspace / "video.m4a").touch()
    seen = []
    monkeypatch.setattr(transcript_segmented, "find_segments", lambda *a, **k: seen.append(1) or [])

    run_cli(monkeypatch, workspace / "video.ass", "-r")

    assert seen == [1]


def test_explicit_audio_file_is_used(workspace, monkeypatch, media):
    audio = workspace / "elsewhere.m4a"
    audio.touch()
    seen = []
    monkeypatch.setattr(transcript_segmented, "find_segments", lambda *a, **k: seen.append(1) or [])

    run_cli(monkeypatch, workspace / "video.ass", "--audio", audio)

    assert seen == [1]


def test_old_style_flag_value_fails_loudly(workspace, monkeypatch):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, workspace / "video.ass", "-t", "auto")
    assert exit_info.value.code == 2


# --- Reusing work ---

def test_second_run_reuses_paid_work(media, workspace, monkeypatch, fake_llms):
    run_cli(monkeypatch, workspace / "show.mp4")
    calls = len(fake_llms["main"].calls)

    run_cli(monkeypatch, workspace / "show.mp4")

    assert len(media.calls) == 1
    assert len(fake_llms["main"].calls) == calls


def test_named_subtitle_file_is_used_over_earlier_work(workspace, monkeypatch):
    source = workspace / "video.ass"
    run_cli(monkeypatch, source, "-f")

    sub_utils.write(make_subs([(0, 5000, "edited")]), source)
    run_cli(monkeypatch, source, "-f")

    assert texts(workspace / "video.fixed.ass") == ["edited"]


def test_named_transcript_is_always_reconverted(media, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.mp4", "-c")
    transcript = workspace / "show.json"
    data = {"text": "new", "segments": [{"text": "new", "start": 0.0, "end": 2.0}],
            "words": [{"word": "new", "start": 0.0, "end": 2.0}]}
    transcript.write_text(json.dumps(data), encoding="utf-8")

    run_cli(monkeypatch, transcript, "-c")

    assert texts(workspace / "show.baberu" / "raw.ass") == ["new"]


def test_languages_are_kept_apart(workspace, monkeypatch):
    source, context = workspace / "video.ass", workspace / "ctx.txt"
    run_cli(monkeypatch, source, "-t", "--context", context, "-p")
    run_cli(monkeypatch, source, "-t", "--context", context, "-p", "--lang-to", "fr")

    job = workspace / "video.baberu"
    assert (job / "en" / "padded.ass").exists() and (job / "fr" / "padded.ass").exists()
    assert (workspace / "video.en.ass").exists() and (workspace / "video.fr.ass").exists()


def test_changed_model_stops_before_spending(workspace, monkeypatch, capsys, fake_llms):
    source, context = workspace / "video.ass", workspace / "ctx.txt"
    run_cli(monkeypatch, source, "-t", "--context", context)
    capsys.readouterr()
    fake_llms["other"] = FakeLLM(translate_all)

    error = fails(monkeypatch, capsys, source, "-t", "--context", context, "--model", "other")

    assert "--redo translate" in error
    assert fake_llms["other"].calls == []


def test_changed_source_stops_before_spending(workspace, monkeypatch, capsys, fake_llms):
    source, context = workspace / "video.ass", workspace / "ctx.txt"
    run_cli(monkeypatch, source, "-t", "--context", context)
    calls = len(fake_llms["main"].calls)
    sub_utils.write(numbered_subs(3), source)
    capsys.readouterr()

    assert "--redo translate" in fails(monkeypatch, capsys, source, "-t", "--context", context)
    assert len(fake_llms["main"].calls) == calls


def test_redo_translates_again(workspace, monkeypatch, fake_llms):
    source, context = workspace / "video.ass", workspace / "ctx.txt"
    run_cli(monkeypatch, source, "-t", "--context", context)
    calls = len(fake_llms["main"].calls)

    run_cli(monkeypatch, source, "-t", "--context", context, "--redo", "translate")

    assert len(fake_llms["main"].calls) > calls


def test_redo_implies_the_stage(workspace, monkeypatch, fake_llms):
    source = workspace / "video.ass"
    run_cli(monkeypatch, source, "-f", "--redo", "translate")

    assert (workspace / "video.baberu" / "en" / "translated.ass").exists()


def test_partial_translation_from_other_source_stops(workspace, monkeypatch, capsys, fake_llms):
    source, context = workspace / "video.ass", workspace / "ctx.txt"

    def fail_second_batch(prompt, count=[]):
        count.append(1)
        if len(count) > 1:
            raise RuntimeError("network down")
        return translate_all(prompt)

    fake_llms["main"] = FakeLLM(fail_second_batch)
    sub_utils.write(numbered_subs(120), source)
    fails(monkeypatch, capsys, source, "-t", "--context", context)
    partial = workspace / "video.baberu" / "en" / "partial.txt"
    assert partial.exists()

    sub_utils.write(make_subs([(0, 1000, "different")]), source)

    assert "--redo translate" in fails(monkeypatch, capsys, source, "-t", "--context", context)
    assert partial.exists()


def test_finished_translation_clears_its_partial(workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")

    assert not (workspace / "video.baberu" / "en" / "partial.txt").exists()


def test_line_ranges_are_kept_apart(workspace, monkeypatch):
    source, context = workspace / "video.ass", workspace / "ctx.txt"
    run_cli(monkeypatch, source, "-t", "--context", context, "--lines", "1-2")
    run_cli(monkeypatch, source, "-t", "--context", context, "--lines", "3-4")

    assert texts(workspace / "video.en.L3-4.ass")[2:4] == ["T:line2", "T:line3"]
    assert (workspace / "video.baberu" / "lines-1-2" / "en" / "translated.ass").exists()


def test_clean_removes_the_work_folder(workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "-f", "--clean")

    assert (workspace / "video.fixed.ass").exists()
    assert not (workspace / "video.baberu").exists()


# --- Output and overwriting ---

def test_directory_holds_everything(media, workspace, monkeypatch):
    out = workspace / "out"

    run_cli(monkeypatch, workspace / "show.mp4", "-d", out)

    assert (out / "show.en.ass").exists() and (out / "show.baberu").is_dir()
    assert not (workspace / "show.en.ass").exists()


def test_output_names_the_result(workspace, monkeypatch):
    target = workspace / "final.srt"

    run_cli(monkeypatch, workspace / "video.ass", "-f", "-o", target)

    assert target.exists() and not (workspace / "video.fixed.ass").exists()


def test_output_may_be_a_folder(workspace, monkeypatch):
    folder = workspace / "results"
    folder.mkdir()

    run_cli(monkeypatch, workspace / "video.ass", "-f", "-o", folder)

    assert (folder / "video.fixed.ass").exists()


def test_output_with_wrong_type_is_refused(workspace, monkeypatch, capsys):
    assert "-o" in fails(monkeypatch, capsys, workspace / "video.ass", "-f", "-o", workspace / "final.mp4")


def test_never_overwrites_someone_elses_file(workspace, monkeypatch, capsys):
    target = workspace / "final.ass"
    target.write_text("precious", encoding="utf-8")

    assert "--overwrite" in fails(monkeypatch, capsys, workspace / "video.ass", "-f", "-o", target)
    assert target.read_text("utf-8") == "precious"


def test_foreign_result_is_refused_before_paying(media, workspace, monkeypatch, capsys):
    (workspace / "show.en.ass").write_text("mine", encoding="utf-8")

    assert "--overwrite" in fails(monkeypatch, capsys, workspace / "show.mp4")
    assert media.calls == []


def test_overwrite_flag_allows_it(workspace, monkeypatch):
    target = workspace / "final.ass"
    target.write_text("old", encoding="utf-8")

    run_cli(monkeypatch, workspace / "video.ass", "-f", "-o", target, "--overwrite")

    assert "line0" in target.read_text("utf-8")


def test_own_result_is_replaced_quietly(workspace, monkeypatch):
    source = workspace / "video.ass"
    run_cli(monkeypatch, source, "-f")
    sub_utils.write(make_subs([(0, 5000, "changed")]), source)

    run_cli(monkeypatch, source, "-f")

    assert texts(workspace / "video.fixed.ass") == ["changed"]


def test_hand_edited_result_is_protected(workspace, monkeypatch, capsys):
    source = workspace / "video.ass"
    run_cli(monkeypatch, source, "-f")
    result = workspace / "video.fixed.ass"
    result.write_text(result.read_text("utf-8") + "\n; my edit", encoding="utf-8")
    sub_utils.write(make_subs([(0, 5000, "changed")]), source)
    capsys.readouterr()

    assert "--overwrite" in fails(monkeypatch, capsys, source, "-f")
    assert "my edit" in result.read_text("utf-8")


def test_identical_rerun_changes_nothing(workspace, monkeypatch):
    source = workspace / "video.ass"
    run_cli(monkeypatch, source, "-f")
    before = (workspace / "video.fixed.ass").read_bytes()

    run_cli(monkeypatch, source, "-f")

    assert (workspace / "video.fixed.ass").read_bytes() == before


def test_input_file_is_never_replaced(workspace, monkeypatch, capsys):
    source = workspace / "video.ass"
    before = source.read_bytes()

    fails(monkeypatch, capsys, source, "-f", "-o", source)

    assert source.read_bytes() == before


# --- Hardcoding ---

def test_hardcode_goes_in_the_output_directory(workspace, monkeypatch):
    video, out = workspace / "clip.mp4", workspace / "out"
    video.touch()
    calls = []
    monkeypatch.setattr(av_utils, "hardcode_subtitles", lambda v, s, o: calls.append(o))

    run_cli(monkeypatch, workspace / "video.ass", "--video", video, "-d", out)

    assert calls == [out / "video.subbed.mp4"]


def test_hardcode_finds_video_beside_subtitles(workspace, monkeypatch):
    (workspace / "video.mp4").touch()
    calls = []
    monkeypatch.setattr(av_utils, "hardcode_subtitles", lambda v, s, o: calls.append(v))

    run_cli(monkeypatch, workspace / "video.ass", "--hardcode")

    assert calls == [workspace / "video.mp4"]


def test_hardcode_uses_this_runs_result(media, workspace, monkeypatch):
    seen = []
    monkeypatch.setattr(av_utils, "hardcode_subtitles", lambda v, s, o: seen.append(s))

    run_cli(monkeypatch, workspace / "show.mp4", "-A", "--hardcode")

    assert seen == [workspace / "show.en.ass"]


def test_hardcode_after_an_earlier_run_uses_its_result(media, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.mp4")
    seen = []
    monkeypatch.setattr(av_utils, "hardcode_subtitles", lambda v, s, o: seen.append(s))

    run_cli(monkeypatch, workspace / "show.mp4", "--hardcode")

    assert seen == [workspace / "show.en.ass"]


def test_hardcode_without_subtitles_is_refused(media, workspace, monkeypatch, capsys):
    assert "subtitle" in fails(monkeypatch, capsys, workspace / "show.mp4", "--hardcode").lower()


def test_existing_video_needs_overwrite(workspace, monkeypatch, capsys):
    (workspace / "video.mp4").touch()
    (workspace / "video.subbed.mp4").touch()

    assert "--overwrite" in fails(monkeypatch, capsys, workspace / "video.ass", "--hardcode")


def test_hardcode_cannot_replace_its_own_input(workspace, monkeypatch, capsys):
    video = workspace / "clip.mp4"
    video.touch()

    fails(monkeypatch, capsys, workspace / "video.ass", "--video", video, "-o", video, "--overwrite")


@pytest.fixture
def real_clip(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    folder = tmp_path / "my [media] 『フォルダ』"
    folder.mkdir()
    video = folder / "クリップ one.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
         "-f", "lavfi", "-i", "anullsrc=d=1", "-map", "1:a", "-map", "0:v", "-shortest", str(video)],
        check=True,
    )
    sub_utils.write(make_subs([(0, 900, "hi")]), folder / "subs.ass")
    return folder


def test_real_hardcode_with_relative_paths(real_clip, monkeypatch):
    monkeypatch.chdir(real_clip)

    av_utils.hardcode_subtitles(Path("クリップ one.mp4"), Path("subs.ass"), Path("out.mp4"))

    assert (real_clip / "out.mp4").stat().st_size > 0
    assert sorted(p.name for p in real_clip.iterdir()) == ["out.mp4", "subs.ass", "クリップ one.mp4"]


def test_real_hardcode_with_absolute_paths(real_clip, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    av_utils.hardcode_subtitles(real_clip / "クリップ one.mp4", real_clip / "subs.ass", real_clip / "out.mp4")

    assert (real_clip / "out.mp4").stat().st_size > 0


def test_real_extraction_into_a_work_folder(real_clip):
    output = real_clip / "work" / "audio.m4a"

    av_utils.extract_audio(real_clip / "クリップ one.mp4", output)

    assert output.stat().st_size > 0


# --- Errors and safety ---

def test_missing_input_is_a_plain_error(workspace, monkeypatch, capsys):
    assert "does not exist" in fails(monkeypatch, capsys, workspace / "nope.mp4")


def test_unknown_type_lists_what_is_supported(workspace, monkeypatch, capsys):
    odd = workspace / "notes.xyz"
    odd.touch()

    assert ".mkv" in fails(monkeypatch, capsys, odd)


def test_missing_config_is_a_plain_error(workspace, monkeypatch, capsys):
    assert "does not exist" in fails(monkeypatch, capsys, workspace / "video.ass", "--config", workspace / "no.yaml")


def test_missing_api_key_is_found_before_paid_transcription(media, workspace, monkeypatch, capsys):
    def no_key(model_name, system_prompt="", **options):
        raise ValueError("API key not found. Set the X_API_KEY environment variable.")

    monkeypatch.setattr(AIToolFactory, "get_llm_provider", staticmethod(no_key))

    assert "X_API_KEY" in fails(monkeypatch, capsys, workspace / "show.mp4")
    assert media.calls == []


def test_missing_ffmpeg_is_found_first(media, workspace, monkeypatch, capsys):
    monkeypatch.setattr("baberu.pipeline.shutil.which", lambda name: None)

    assert "ffmpeg" in fails(monkeypatch, capsys, workspace / "show.mp4")
    assert media.calls == []


def test_failure_mid_run_is_a_plain_error(workspace, monkeypatch, capsys, fake_llms):
    def broken(prompt):
        raise RuntimeError("service exploded")

    fake_llms["main"] = FakeLLM(broken)

    assert "service exploded" in fails(monkeypatch, capsys, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")


def test_interrupt_exits_130(workspace, monkeypatch, capsys, fake_llms):
    def interrupted(prompt):
        raise KeyboardInterrupt

    fake_llms["main"] = FakeLLM(interrupted)

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")

    assert exit_info.value.code == 130
    assert "resume" in capsys.readouterr().err


def test_debug_shows_the_traceback(workspace, monkeypatch, fake_llms):
    def broken(prompt):
        raise RuntimeError("service exploded")

    fake_llms["main"] = FakeLLM(broken)

    with pytest.raises(RuntimeError):
        run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt", "--debug")


def test_atomic_path_leaves_nothing_on_failure(tmp_path):
    target = tmp_path / "sub" / "file.ogg"

    with pytest.raises(RuntimeError):
        with atomic_path(target) as temporary:
            temporary.write_bytes(b"half")
            raise RuntimeError

    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_atomic_path_publishes_on_success(tmp_path):
    target = tmp_path / "file.ogg"

    with atomic_path(target) as temporary:
        assert temporary.suffix == ".ogg"
        temporary.write_bytes(b"whole")

    assert target.read_bytes() == b"whole"
    assert [p.name for p in tmp_path.iterdir()] == ["file.ogg"]


def test_context_prompt_is_skipped_without_a_terminal(workspace, monkeypatch):
    from baberu.setup import config_setup

    settings = config_setup.load_config()
    settings["translation"]["confirm_auto_context"] = True
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)

    def no_input(*args):
        raise AssertionError("should not prompt")

    monkeypatch.setattr("builtins.input", no_input)

    run_cli(monkeypatch, workspace / "video.ass", "-t")

    assert (workspace / "video.en.ass").exists()


def test_dry_run_writes_nothing_and_spends_nothing(media, workspace, monkeypatch, capsys, fake_llms):
    before = sorted(p.name for p in workspace.iterdir())

    run_cli(monkeypatch, workspace / "show.mp4", "--dry-run")

    assert sorted(p.name for p in workspace.iterdir()) == before
    assert media.calls == [] and fake_llms["main"].calls == []
    output = capsys.readouterr()
    assert "transcribe" in output.err and "translate" in output.err and "paid" in output.err


def test_dry_run_shows_reuse(media, workspace, monkeypatch, capsys):
    run_cli(monkeypatch, workspace / "show.mp4")
    capsys.readouterr()

    run_cli(monkeypatch, workspace / "show.mp4", "--dry-run")

    assert "reuse" in capsys.readouterr().err


# --- File types and job names ---

@pytest.mark.parametrize(
    "name,kind",
    [("a.m4v", FileKind.VIDEO), ("a.MPG", FileKind.VIDEO), ("a.m2ts", FileKind.VIDEO), ("a.mka", FileKind.AUDIO),
     ("a.wma", FileKind.AUDIO), ("a.srt", FileKind.SUBTITLE), ("a.json", FileKind.TRANSCRIPT),
     ("a.txt", FileKind.TEXT), ("a.png", FileKind.IMAGE), ("a.xyz", FileKind.UNKNOWN), ("a", FileKind.UNKNOWN)],
)
def test_kind_of(name, kind):
    assert kind_of(Path(name)) == kind


def locate(path: Path, output_dir: Path | None = None) -> Job:
    return Job.locate(path, output_dir or path.parent, lang_from="ja", lang_to="en", legacy_name=legacy.job_name(path))


@pytest.mark.parametrize(
    "name",
    ["foo.raw.mkv", "Ep.1.en.mp4", "title .webm", "【とても長い日本語の題名】架空の動画です" * 4 + ".mp4", "a [b] (c).mp4"],
)
def test_media_job_is_the_exact_file_stem(name, tmp_path):
    path = tmp_path / name
    path.touch()

    job = locate(path)

    assert job.name == path.stem
    assert job.folder == tmp_path / f"{path.stem}.baberu"


def test_subtitle_named_like_an_old_stage_is_left_alone(tmp_path):
    path = tmp_path / "foo.raw.ass"
    path.touch()

    assert locate(path).name == "foo.raw"


def test_old_style_subtitle_with_siblings_joins_its_job(tmp_path):
    (tmp_path / "video.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "video.raw.ass"
    path.touch()

    assert locate(path).name == "video"


def test_file_inside_a_work_folder_belongs_to_that_job(tmp_path):
    folder = tmp_path / "show.baberu" / "en"
    folder.mkdir(parents=True)
    path = folder / "translated.ass"
    path.touch()

    job = locate(path, folder)

    assert job.name == "show" and job.folder == tmp_path / "show.baberu"
    assert enclosing_job(path) == tmp_path / "show.baberu"


def test_working_from_a_file_in_the_work_folder(workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "-f")
    fixed = workspace / "video.baberu" / "fixed.ass"

    run_cli(monkeypatch, fixed, "-p")

    assert (workspace / "video.padded.ass").exists()


def test_published_result_belongs_to_its_job(workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")

    assert locate(workspace / "video.en.ass").name == "video"


def test_result_as_input_gets_a_new_name(workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")
    before = (workspace / "video.en.ass").read_bytes()

    run_cli(monkeypatch, workspace / "video.en.ass", "-p")

    assert (workspace / "video.en.padded.ass").exists()
    assert (workspace / "video.en.ass").read_bytes() == before


def test_two_sources_cannot_share_a_job(media, workspace, monkeypatch, capsys):
    run_cli(monkeypatch, workspace / "show.mp4", "-c")
    (workspace / "show.mkv").touch()
    capsys.readouterr()

    assert "show.mp4" in fails(monkeypatch, capsys, workspace / "show.mkv", "-c")


def test_download_uses_the_real_file_path(monkeypatch, tmp_path):
    captured = {}

    class FakeYDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            return {"requested_downloads": [{"filepath": str(tmp_path / "Title [abc].mkv")}]}

    monkeypatch.setattr(av_utils.yt_dlp, "YoutubeDL", FakeYDL)

    assert av_utils.download("https://x/y", tmp_path) == tmp_path / "Title [abc].mkv"
    assert captured["outtmpl"] == "%(title).120B [%(id)s].%(ext)s"
    assert captured["paths"] == {"home": str(tmp_path)}


# --- Existing jobs ---

def make_old_job(folder: Path, name: str = "show") -> None:
    (folder / f"{name}.json").write_text(
        '{"text": "hi", "segments": [{"text": "hi", "start": 0, "end": 1}], "words": [{"word": "hi", "start": 0, "end": 1}]}',
        encoding="utf-8",
    )
    (folder / f"{name}.context.txt").write_text("notes", encoding="utf-8")
    for stage in ("raw", "2pass", "fixed", "en", "padded"):
        sub_utils.write(make_subs([(0, 5000, f"old {stage}")]), folder / f"{name}.{stage}.ass")


def test_old_job_resumes_without_paying(media, workspace, monkeypatch, fake_llms):
    make_old_job(workspace)

    run_cli(monkeypatch, workspace / "show.mp4")

    assert media.calls == [] and fake_llms["main"].calls == []
    assert texts(workspace / "show.en.ass") == ["old en"]
    assert (workspace / "show.raw.ass").exists(), "originals stay"


def test_old_job_found_when_using_a_new_directory(media, workspace, monkeypatch, fake_llms):
    make_old_job(workspace)

    run_cli(monkeypatch, workspace / "show.mp4", "-d", workspace / "out")

    assert media.calls == [] and fake_llms["main"].calls == []


def test_old_partial_translation_is_adopted(workspace, monkeypatch, fake_llms):
    (workspace / "video.partial.en.txt").write_text("kept0\nkept1", encoding="utf-8")

    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")

    assert texts(workspace / "video.en.ass")[:3] == ["kept0", "kept1", "T:line2"]


def test_dry_run_does_not_adopt(media, workspace, monkeypatch, capsys):
    make_old_job(workspace)

    run_cli(monkeypatch, workspace / "show.mp4", "--dry-run")

    assert not (workspace / "show.baberu").exists()
    assert "reuse" in capsys.readouterr().err
