"""Video frames sent with translation batches: scenes, picking, caching, prompts and the pipeline stage."""
import copy
import json

import ffmpeg
import pytest

from baberu.providers.llm.base import Image, prompt_parts
from baberu.settings import TranslationSettings
from baberu.subtitling import frames, sub_translation, translation_prompts
from baberu.subtitling.frames import Scene
from conftest import FakeLLM, is_retry, make_subs, numbered_entries, numbered_subs, run_cli, translate_all, \
    translate_or_describe


def images_in(prompt) -> int:
    return sum(isinstance(part, Image) for part in prompt_parts(prompt))


# --- Scenes ---

def test_scenes_cover_the_video_from_start_to_end():
    scenes = frames.scenes_from_cuts([4.0, 9.0], 12.0, min_scene_sec=1.0, max_scene_sec=60.0)

    assert scenes == [Scene(0, 4000), Scene(4000, 9000), Scene(9000, 12000)]


def test_a_short_scene_joins_the_one_before():
    scenes = frames.scenes_from_cuts([4.0, 4.5, 9.0], 12.0, min_scene_sec=1.0, max_scene_sec=60.0)

    assert scenes == [Scene(0, 4500), Scene(4500, 9000), Scene(9000, 12000)]


def test_a_short_first_scene_joins_the_one_after():
    scenes = frames.scenes_from_cuts([0.5, 6.0], 12.0, min_scene_sec=1.0, max_scene_sec=60.0)

    assert scenes == [Scene(0, 6000), Scene(6000, 12000)]


def test_a_long_scene_is_split_evenly():
    scenes = frames.scenes_from_cuts([], 50.0, min_scene_sec=1.0, max_scene_sec=20.0)

    assert scenes == [Scene(0, 16667), Scene(16667, 33333), Scene(33333, 50000)]


def test_cuts_outside_the_video_are_ignored():
    assert frames.scenes_from_cuts([0.0, -1.0, 15.0], 10.0, 1.0, 60.0) == [Scene(0, 10000)]
    assert frames.scenes_from_cuts([1.0], 0.0, 1.0, 60.0) == []


# --- Picking scenes for events ---

def test_pick_keeps_only_scenes_under_the_events():
    scenes = [Scene(0, 2000), Scene(2000, 4000), Scene(4000, 6000)]
    events = make_subs([(2500, 3000, "a")]).events

    assert frames.pick(scenes, events, limit=10) == [Scene(2000, 4000)]


def test_pick_over_the_limit_keeps_the_longest_in_time_order():
    scenes = [Scene(0, 1000), Scene(1000, 5000), Scene(5000, 6000), Scene(6000, 9000)]
    events = make_subs([(0, 9000, "all")]).events

    assert frames.pick(scenes, events, limit=2) == [Scene(1000, 5000), Scene(6000, 9000)]


# --- Grabbing frames ---

@pytest.fixture
def grabbed(monkeypatch):
    """Stands in for ffmpeg's frame grab, writing the time into the file. Lists each grab."""
    times = []

    def grab(video, seconds, height, output):
        times.append(seconds)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"jpeg {seconds}".encode())
        return output

    monkeypatch.setattr("baberu.tools.av_utils.grab_frame", grab)
    return times


def test_frames_are_grabbed_once_and_kept(grabbed, tmp_path):
    source = frames.FrameSource(tmp_path / "v.mp4", [Scene(0, 4000)], tmp_path / "frames", 480, 10)
    events = make_subs([(0, 1000, "a")]).events

    first = source.for_events(events)
    second = source.for_events(events)

    assert grabbed == [2.0]
    assert first[0].image.data == second[0].image.data == b"jpeg 2.0"
    assert (tmp_path / "frames" / "2000-480p.jpg").exists()


def test_a_frame_ffmpeg_cannot_grab_is_left_out(monkeypatch, tmp_path):
    def fail(*args):
        raise ffmpeg.Error("ffmpeg", b"", b"bad")

    monkeypatch.setattr("baberu.tools.av_utils.grab_frame", fail)
    source = frames.FrameSource(tmp_path / "v.mp4", [Scene(0, 4000)], tmp_path / "frames", 480, 10)

    assert source.for_events(make_subs([(0, 1000, "a")]).events) == []


# --- Prompts ---

def test_frames_follow_the_prompt_marked_with_their_entries():
    events = make_subs([(0, 1000, "a"), (1000, 2000, "b"), (2500, 3000, "c")]).events
    shown = [frames.Frame(Scene(0, 2400), Image(b"1")), frames.Frame(Scene(2400, 65000), Image(b"2"))]

    prompt = translation_prompts.with_frames("text", shown, events)

    assert prompt == ["text", "\n\nFrames from the video, each marked with the entries it shows:",
                      "\nFrame 0:00-0:02, entries 1-2:", Image(b"1"),
                      "\nFrame 0:02-1:05, entry 3:", Image(b"2")]


def test_no_frames_leaves_the_prompt_as_it_is():
    assert translation_prompts.with_frames("text", [], numbered_subs(2).events) == "text"


def test_the_system_prompt_mentions_frames_only_when_they_are_sent():
    assert "Frames" not in translation_prompts.system_prompt("ja", "en")
    assert "Frames from the video" in translation_prompts.system_prompt("ja", "en", frames=True)


# --- Translating with frames ---

def two_second_scenes(count: int) -> list[Scene]:
    return [Scene(i * 2000, i * 2000 + 2000) for i in range(count)]


def translate_with(llms, subs, tmp_path, source, **options):
    return sub_translation.translate(
        sub_file=subs, output_file=tmp_path / "partial.txt", context_prompt="ctx", llm=llms["main"],
        fallback_llm=llms.get("backup"), lang_from="ja", lang_to="en", options=TranslationSettings(**options),
        report_file=tmp_path / "translation.json", frames=source)


def test_frames_go_with_the_kept_lines_only(fake_llms, grabbed, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all, images=True)
    source = frames.FrameSource(tmp_path / "v.mp4", two_second_scenes(60), tmp_path / "frames", 480, 100)

    result, report = translate_with(fake_llms, numbered_subs(120), tmp_path, source)

    first = fake_llms["main"].calls[0]
    assert len(numbered_entries(first)) == 60      # 50 lines plus 10 lookahead
    assert images_in(first) == 50                  # one scene under each kept line
    assert first[2] == "\nFrame 0:00-0:02, entry 1:"
    assert report["batches"][0]["attempts"][0]["frames"] == 50
    assert result == [f"T:line{i}" for i in range(120)]


def test_a_targeted_retry_sends_only_its_entries_frames(fake_llms, grabbed, tmp_path):
    def skips_entry_2(prompt):
        if is_retry(prompt):
            return translate_all(prompt)
        return "\n".join(f"{n}. T:{t}" for n, t in numbered_entries(prompt) if n != 2)

    fake_llms["main"] = FakeLLM(skips_entry_2, images=True)
    source = frames.FrameSource(tmp_path / "v.mp4", two_second_scenes(4), tmp_path / "frames", 480, 100)

    translate_with(fake_llms, numbered_subs(4), tmp_path, source, discard_lines=0)

    retry = fake_llms["main"].calls[1]
    assert images_in(retry) == 1
    assert "\nFrame 0:02-0:04, entry 2:" in retry


def test_a_model_without_images_gets_plain_text(fake_llms, grabbed, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all, images=False)
    source = frames.FrameSource(tmp_path / "v.mp4", two_second_scenes(4), tmp_path / "frames", 480, 100)

    translate_with(fake_llms, numbered_subs(4), tmp_path, source)

    assert isinstance(fake_llms["main"].calls[0], str)
    assert grabbed == []


def test_a_fallback_without_images_gets_plain_text(fake_llms, grabbed, tmp_path):
    fake_llms["main"] = FakeLLM(lambda prompt: RuntimeError("down"), images=True)
    fake_llms["backup"] = FakeLLM(translate_all, model="backup", images=False)
    source = frames.FrameSource(tmp_path / "v.mp4", two_second_scenes(4), tmp_path / "frames", 480, 100)

    result, _ = translate_with(fake_llms, numbered_subs(4), tmp_path, source)

    assert images_in(fake_llms["main"].calls[0]) == 4
    assert isinstance(fake_llms["backup"].calls[0], str)
    assert result == [f"T:line{i}" for i in range(4)]


# --- The frames stage ---

@pytest.fixture
def video_job(workspace, monkeypatch, fake_llms, grabbed):
    """workspace with frames on, a 480p video.mp4 beside video.ass, and ffmpeg's scene search stubbed."""
    from baberu.setup import config_setup

    config = copy.deepcopy(config_setup.load_config())
    config["frames"]["enabled"] = True
    monkeypatch.setattr("baberu.setup.config_setup.load_config", lambda *a: config)
    monkeypatch.setattr("baberu.pipeline.shutil.which", lambda name: "ffmpeg")
    monkeypatch.setattr("baberu.tools.av_utils.video_height", lambda path: 480)
    monkeypatch.setattr("baberu.tools.av_utils.duration_seconds", lambda path: 12.0)
    searches = []

    def scene_cuts(video, threshold):
        searches.append(video.name)
        return [4.0, 8.0]

    monkeypatch.setattr("baberu.tools.av_utils.scene_cuts", scene_cuts)
    fake_llms["main"] = FakeLLM(translate_or_describe, images=True)
    (workspace / "video.mp4").write_bytes(b"video")
    return searches


def translate_cli(monkeypatch, workspace, *extra):
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt", *extra)


def test_translation_sends_one_frame_per_scene(video_job, workspace, monkeypatch, fake_llms):
    translate_cli(monkeypatch, workspace)

    prompt = fake_llms["main"].calls[0]
    assert images_in(prompt) == 3
    assert video_job == ["video.mp4"]
    saved = json.loads((workspace / "video.baberu" / "scenes.json").read_text(encoding="utf-8"))
    assert saved == {"duration_sec": 12.0, "cuts": [4.0, 8.0]}


def test_scenes_are_reused_and_found_again_on_redo(video_job, workspace, monkeypatch):
    translate_cli(monkeypatch, workspace)
    translate_cli(monkeypatch, workspace, "--redo", "translate")
    assert video_job == ["video.mp4"]

    translate_cli(monkeypatch, workspace, "--redo", "translate", "--redo", "frames")
    assert video_job == ["video.mp4", "video.mp4"]


def test_a_small_video_is_translated_without_frames(video_job, workspace, monkeypatch, fake_llms, caplog):
    monkeypatch.setattr("baberu.tools.av_utils.video_height", lambda path: 144)

    translate_cli(monkeypatch, workspace)

    assert isinstance(fake_llms["main"].calls[0], str)
    assert video_job == []
    assert "video.mp4 is 144p, under 360p" in caplog.text


def test_a_model_without_images_is_translated_without_frames(video_job, workspace, monkeypatch, fake_llms, caplog):
    fake_llms["main"] = FakeLLM(translate_or_describe, images=False)

    translate_cli(monkeypatch, workspace)

    assert isinstance(fake_llms["main"].calls[0], str)
    assert video_job == []
    assert "doesn't take images" in caplog.text


def test_the_dry_run_shows_the_frames_stage(video_job, workspace, monkeypatch, capsys):
    translate_cli(monkeypatch, workspace, "--dry-run")

    plan = capsys.readouterr().err
    assert "frames" in plan and "480p, up to 24 a batch" in plan
    assert video_job == []
