import json

from pysubs2 import SSAFile

from baberu.providers.llm.base import EMPTY, REFUSED, TRUNCATED, Completion
from baberu.settings import TranslationSettings
from baberu.subtitling import sub_translation, translation_prompts
from baberu.subtitling.types import LineRange
from conftest import FakeLLM, batch_entries, is_retry, make_subs, numbered_entries, numbered_subs, translate_all


def run_translate_with_report(llms, subs: SSAFile, partial, line_range=None, report_file=None, **options):
    return sub_translation.translate(
        sub_file=subs,
        output_file=partial,
        context_prompt="ctx",
        llm=llms["main"],
        fallback_llm=llms.get("backup"),
        lang_from="ja",
        lang_to="en",
        options=TranslationSettings(**options),
        line_range=line_range,
        report_file=report_file,
    )


def run_translate(llms, subs: SSAFile, partial, line_range=None, **options) -> list[str]:
    return run_translate_with_report(llms, subs, partial, line_range, **options)[0]


def test_translates_every_line_in_order_across_batches(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)

    result = run_translate(fake_llms, numbered_subs(120), tmp_path / "partial.txt")

    assert result == [f"T:line{i}" for i in range(120)]
    assert len(fake_llms["main"].calls) == 3


def test_lookahead_lines_are_sent_but_discarded(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)

    run_translate(fake_llms, numbered_subs(120), tmp_path / "partial.txt")

    first_batch = batch_entries(fake_llms["main"].calls[0])
    assert len(first_batch) == 60  # 50 lines plus 10 lookahead


def test_unnumbered_response_with_the_exact_count_is_read_by_position(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda p: "\n".join(f"T:{e}" for e in batch_entries(p)))

    result = run_translate(fake_llms, numbered_subs(5), tmp_path / "partial.txt")

    assert result == [f"T:line{i}" for i in range(5)]


def test_resumes_from_partial_file(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)
    partial = tmp_path / "partial.txt"
    partial.write_text("\n".join(f"old{i}" for i in range(50)), encoding="utf-8")

    result = run_translate(fake_llms, numbered_subs(60), partial)

    assert result[:50] == [f"old{i}" for i in range(50)]
    assert result[50:] == [f"T:line{i}" for i in range(50, 60)]
    assert len(fake_llms["main"].calls) == 1


def test_short_response_is_retried_then_left_as_placeholders(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda prompt: "1. only one line")

    result, report = run_translate_with_report(fake_llms, numbered_subs(3), tmp_path / "partial.txt", translate_retries=2)

    assert len(fake_llms["main"].calls) == 2
    assert result == ["only one line", "[Translation missing]", "[Translation missing]"]
    assert sub_translation.unresolved_lines(report) == {2: "missing", 3: "missing"}


def test_segment_translates_only_requested_lines(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)

    result = run_translate(fake_llms, numbered_subs(10), tmp_path / "partial.txt", line_range=LineRange(3, 5))

    assert result == ["line0", "line1", "line2", "T:line3", "T:line4", "T:line5"]


def test_segment_never_shows_source_lines_as_translations(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)

    run_translate(fake_llms, numbered_subs(10), tmp_path / "partial.txt", line_range=LineRange(3, 5))

    assert "translated entries" not in fake_llms["main"].calls[0]


def test_a_merge_mid_batch_leaves_a_placeholder_and_shifts_nothing(fake_llms, tmp_path):
    def merges_entry_3(prompt):
        if is_retry(prompt):
            return "nothing useful"
        return "\n".join(f"{n}. T:{t}" for n, t in numbered_entries(prompt) if n != 3)

    fake_llms["main"] = FakeLLM(merges_entry_3)

    result = run_translate(fake_llms, numbered_subs(6), tmp_path / "partial.txt", discard_lines=0)

    assert result == ["T:line0", "T:line1", "[Translation missing]", "T:line3", "T:line4", "T:line5"]


def test_a_gap_only_in_the_lookahead_needs_no_retry(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda p: "\n".join(f"{n}. T:{t}" for n, t in numbered_entries(p) if n <= 2))

    result = run_translate(fake_llms, numbered_subs(4), tmp_path / "partial.txt", batch_lines=2, discard_lines=2)

    assert result[:2] == ["T:line0", "T:line1"]
    assert not any(is_retry(call) for call in fake_llms["main"].calls)


def test_targeted_retry_asks_only_for_the_bad_entries_and_merges_them(fake_llms, tmp_path):
    def skips_entry_2(prompt):
        if is_retry(prompt):
            return translate_all(prompt)
        return "\n".join(f"{n}. T:{t}" for n, t in numbered_entries(prompt) if n != 2)

    fake_llms["main"] = FakeLLM(skips_entry_2)

    result, report = run_translate_with_report(fake_llms, numbered_subs(4), tmp_path / "partial.txt", discard_lines=0)

    retry = fake_llms["main"].calls[1]
    assert numbered_entries(retry) == [(2, "line1")]
    assert "ctx" in retry and "1. T:line0" in retry  # glossary and accepted lines are kept
    assert result == [f"T:line{i}" for i in range(4)]
    assert sub_translation.unresolved_lines(report) == {}


def test_untranslated_line_is_retried(fake_llms, tmp_path):
    def copies_entry_1(prompt):
        if is_retry(prompt):
            return translate_all(prompt)
        return "\n".join(f"{n}. {t}" if n == 1 else f"{n}. T:{t}" for n, t in numbered_entries(prompt))

    fake_llms["main"] = FakeLLM(copies_entry_1)

    result = run_translate(fake_llms, numbered_subs(3), tmp_path / "partial.txt", discard_lines=0)

    assert result == ["T:line0", "T:line1", "T:line2"]


def test_a_line_returned_unchanged_twice_is_taken_as_meant(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda p: "\n".join(f"{n}. {t}" if t == "OK" else f"{n}. T:{t}" for n, t in numbered_entries(p)))
    subs = make_subs([(0, 1000, "line0"), (1000, 2000, "OK")])

    result, report = run_translate_with_report(fake_llms, subs, tmp_path / "partial.txt", discard_lines=0)

    assert result == ["T:line0", "OK"]
    assert len(fake_llms["main"].calls) == 2
    assert sub_translation.unresolved_lines(report) == {}


def test_truncated_response_splits_the_batch(fake_llms, tmp_path):
    def truncates_big_requests(prompt):
        if len(numbered_entries(prompt)) > 2:
            return Completion("1. T:line0\n2. T:li", TRUNCATED)
        return translate_all(prompt)

    fake_llms["main"] = FakeLLM(truncates_big_requests)

    result = run_translate(fake_llms, numbered_subs(4), tmp_path / "partial.txt", discard_lines=0)

    assert result == [f"T:line{i}" for i in range(4)]
    assert [len(numbered_entries(c)) for c in fake_llms["main"].calls] == [4, 2, 2]


def test_truncation_first_drops_the_lookahead(fake_llms, tmp_path):
    def truncates_with_lookahead(prompt):
        if len(numbered_entries(prompt)) > 2:
            return Completion("", TRUNCATED)
        return translate_all(prompt)

    fake_llms["main"] = FakeLLM(truncates_with_lookahead)

    result = run_translate(fake_llms, numbered_subs(4), tmp_path / "partial.txt", batch_lines=2, discard_lines=2)

    assert result == [f"T:line{i}" for i in range(4)]


def test_refusal_goes_to_the_fallback(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda p: Completion("", REFUSED))
    fake_llms["backup"] = FakeLLM(translate_all, model="backup")

    result, report = run_translate_with_report(fake_llms, numbered_subs(3), tmp_path / "partial.txt")

    assert len(fake_llms["main"].calls) == 1
    assert result == ["T:line0", "T:line1", "T:line2"]
    statuses = [(a["model"], a["status"]) for a in report["batches"][0]["attempts"]]
    assert statuses == [("fake-model", REFUSED), ("backup", "complete")]


def test_an_error_goes_to_the_fallback(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda p: RuntimeError("bad request"))
    fake_llms["backup"] = FakeLLM(translate_all, model="backup")

    result = run_translate(fake_llms, numbered_subs(3), tmp_path / "partial.txt")

    assert result == ["T:line0", "T:line1", "T:line2"]


def test_an_error_with_no_model_left_is_raised(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda p: RuntimeError("bad request"))

    try:
        run_translate(fake_llms, numbered_subs(3), tmp_path / "partial.txt")
    except RuntimeError as error:
        assert "bad request" in str(error)
    else:
        raise AssertionError("expected the error to be raised")


def test_empty_response_with_no_model_left_leaves_placeholders(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda p: Completion("", EMPTY))

    result = run_translate(fake_llms, numbered_subs(2), tmp_path / "partial.txt")

    assert result == ["[Translation missing]"] * 2
    assert len(fake_llms["main"].calls) == 1


def test_report_is_saved_and_kept_when_resuming(fake_llms, tmp_path):
    partial, report_file = tmp_path / "partial.txt", tmp_path / "translation.json"
    calls = []

    def first_batch_misses_line_2(prompt):
        calls.append(prompt)
        if len(calls) <= 3:
            return "\n".join(f"{n}. T:{t}" for n, t in numbered_entries(prompt) if t != "line1")
        raise KeyboardInterrupt

    fake_llms["main"] = FakeLLM(first_batch_misses_line_2)
    try:
        run_translate_with_report(fake_llms, numbered_subs(4), partial, report_file=report_file,
                                  batch_lines=2, discard_lines=0)
    except KeyboardInterrupt:
        pass
    assert sub_translation.unresolved_lines(json.loads(report_file.read_text("utf-8"))) == {2: "missing"}

    fake_llms["main"] = FakeLLM(translate_all)
    result, report = run_translate_with_report(fake_llms, numbered_subs(4), partial, report_file=report_file,
                                               batch_lines=2, discard_lines=0)

    assert result == ["T:line0", "[Translation missing]", "T:line2", "T:line3"]
    assert [b["lines"] for b in report["batches"]] == [[1, 2], [3, 4]]
    assert sub_translation.unresolved_lines(report) == {2: "missing"}


def test_partial_ending_in_an_empty_line_resumes_without_a_shift(fake_llms, tmp_path):
    partial = tmp_path / "partial.txt"
    sub_translation.write_lines(["T:a", ""], partial)
    fake_llms["main"] = FakeLLM(translate_all)

    result = run_translate(fake_llms, numbered_subs(3), partial)

    assert result == ["T:a", "", "T:line2"]


def test_prompt_sends_line_breaks_as_a_single_escape():
    subs = make_subs([(0, 1000, r"one\Ntwo"), (1000, 2000, "three\nfour"), (2000, 3000, r"five\nsix")])

    prompt = translation_prompts.translate_prompt([], "ctx", subs.events, "ja", "en", 10)

    assert "1. one\\Ntwo\n2. three\\Nfour\n3. five\\Nsix" in prompt
    assert "\\\\N" not in prompt
