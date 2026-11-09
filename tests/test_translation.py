from pysubs2 import SSAFile

from baberu.subtitling import sub_translation
from conftest import FakeLLM, batch_entries, numbered_subs, translate_all


def run_translate(subs: SSAFile, partial, **overrides) -> list[str]:
    args = dict(
        sub_file=subs,
        output_file=partial,
        context_prompt="ctx",
        model="main",
        fallback_model="backup",
        lang_from="ja",
        lang_to="en",
        batch_lines=50,
        discard_lines=10,
    )
    args.update(overrides)
    return sub_translation.translate(**args)


def test_translates_every_line_in_order_across_batches(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)

    result = run_translate(numbered_subs(120), tmp_path / "partial.txt")

    assert result == [f"T:line{i}" for i in range(120)]
    assert len(fake_llms["main"].calls) == 3


def test_lookahead_lines_are_sent_but_discarded(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)

    run_translate(numbered_subs(120), tmp_path / "partial.txt")

    first_batch = batch_entries(fake_llms["main"].calls[0])
    assert len(first_batch) == 60  # 50 lines plus 10 lookahead


def test_numbered_responses_are_unnumbered(fake_llms, tmp_path):
    def numbered(prompt):
        return "\n".join(f"{i + 1}. T:{e}" for i, e in enumerate(batch_entries(prompt)))

    fake_llms["main"] = FakeLLM(numbered)

    result = run_translate(numbered_subs(5), tmp_path / "partial.txt")

    assert result == [f"T:line{i}" for i in range(5)]


def test_resumes_from_partial_file(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)
    partial = tmp_path / "partial.txt"
    partial.write_text("\n".join(f"old{i}" for i in range(50)), encoding="utf-8")

    result = run_translate(numbered_subs(60), partial)

    assert result[:50] == [f"old{i}" for i in range(50)]
    assert result[50:] == [f"T:line{i}" for i in range(50, 60)]
    assert len(fake_llms["main"].calls) == 1


def test_short_response_is_retried_then_padded(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(lambda prompt: "only one line")

    result = run_translate(numbered_subs(3), tmp_path / "partial.txt", translate_retries=2, fallback_model="main")

    assert len(fake_llms["main"].calls) == 2
    assert result == ["only one line", "[Translation missing]", "[Translation missing]"]


def test_segment_translates_only_requested_lines(fake_llms, tmp_path):
    fake_llms["main"] = FakeLLM(translate_all)

    result = run_translate(numbered_subs(10), tmp_path / "partial.txt", segment=[3, 5])

    assert result == ["line0", "line1", "line2", "T:line3", "T:line4", "T:line5"]


def test_force_line_count_merges_excess():
    targets = numbered_subs(2).events

    assert sub_translation._force_line_count(["a", "b", "c"], targets) == ["a", "b\\Nc"]
