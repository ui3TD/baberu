from baberu.subtitling import sub_translation
from conftest import FakeLLM, numbered_subs, translate_all
from test_translation import run_translate


def test_leading_ellipsis_does_not_destroy_the_next_line():
    assert sub_translation._clean_ellipses(["...foo", "bar", "baz"]) == ["foo", "bar", "baz"]


def test_matching_ellipses_across_lines_are_removed():
    assert sub_translation._clean_ellipses(["foo...", "...bar"]) == ["foo", "bar"]


def test_fallback_model_takes_over_after_retries_run_out(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda prompt: "wrong count")
    fake_llms["backup"] = FakeLLM(translate_all)

    result = run_translate(numbered_subs(3), tmp_path / "partial.txt", translate_retries=2)

    assert len(fake_llms["main"].calls) == 2
    assert result == ["T:line0", "T:line1", "T:line2"]


def test_fallback_is_only_for_the_failing_batch(fake_llms, tmp_path):
    outcomes = iter(["wrong count"])
    fake_llms["main"] = FakeLLM(lambda p: next(outcomes, None) or translate_all(p))
    fake_llms["backup"] = FakeLLM(translate_all)

    run_translate(numbered_subs(4), tmp_path / "partial.txt", batch_lines=2, discard_lines=0, translate_retries=1)

    assert len(fake_llms["backup"].calls) == 1
    assert len(fake_llms["main"].calls) == 2


def test_partial_file_keeps_lines_after_a_segment(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)
    partial = tmp_path / "partial.txt"

    run_translate(numbered_subs(6), partial, segment=[1, 2])

    assert partial.read_text("utf-8").split("\n") == [
        "line0", "T:line1", "T:line2", "line3", "line4", "line5",
    ]


def test_partial_from_different_source_is_not_reused(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)
    partial = tmp_path / "partial.txt"
    run_translate(numbered_subs(3), partial)
    partial.write_text("stale0\nstale1", encoding="utf-8")
    changed = numbered_subs(3)
    changed.events[0].text = "different"

    result = run_translate(changed, partial)

    assert result == ["T:different", "T:line1", "T:line2"]


def test_partial_with_matching_source_is_resumed(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)
    partial = tmp_path / "partial.txt"
    run_translate(numbered_subs(3), partial)
    partial.write_text("kept0\nkept1", encoding="utf-8")

    result = run_translate(numbered_subs(3), partial)

    assert result == ["kept0", "kept1", "T:line2"]


def test_partial_without_a_source_record_still_resumes(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)
    partial = tmp_path / "partial.txt"
    partial.write_text("kept0", encoding="utf-8")

    result = run_translate(numbered_subs(2), partial)

    assert result == ["kept0", "T:line1"]
