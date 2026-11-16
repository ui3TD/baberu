"""Translation as the pipeline runs it: the report, preflight checks and resuming."""
import json
import logging

import pytest
from pydantic import ValidationError

from baberu.providers.llm.base import Completion
from baberu.settings import TranslationSettings
from baberu.setup import config_setup
from baberu.subtitling import sub_utils
from baberu.subtitling.types import LineRange
from conftest import FakeLLM, make_subs, numbered_entries, numbered_subs, run_cli, texts, translate_all
from test_file_handling import fails


def skips_line_3(prompt):
    return "\n".join(f"{n}. T:{t}" for n, t in numbered_entries(prompt) if t != "line2")


def test_unresolved_lines_are_reported_and_logged(workspace, monkeypatch, fake_llms, caplog):
    fake_llms["main"] = FakeLLM(skips_line_3)

    with caplog.at_level(logging.WARNING):
        run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")

    report = json.loads((workspace / "video.baberu" / "en" / "translation.json").read_text("utf-8"))
    assert report["batches"][0]["unresolved"] == {"3": "missing"}
    assert "translate: 1 lines unresolved (3)" in caplog.text
    assert texts(workspace / "video.baberu" / "en" / "translated.ass")[2] == "[Translation missing]"


def test_redo_translate_clears_the_report(workspace, monkeypatch, fake_llms):
    fake_llms["main"] = FakeLLM(skips_line_3)
    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")
    fake_llms["main"] = FakeLLM(translate_all)

    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt", "--redo", "translate")

    report = json.loads((workspace / "video.baberu" / "en" / "translation.json").read_text("utf-8"))
    assert [b["unresolved"] for b in report["batches"]] == [{}]


def test_empty_generated_context_stops_the_run(workspace, monkeypatch, capsys, fake_llms):
    fake_llms["main"] = FakeLLM(lambda p: translate_all(p) if numbered_entries(p) else Completion("", "empty"))

    assert "no context" in fails(monkeypatch, capsys, workspace / "video.ass", "-t")
    assert not (workspace / "video.baberu" / "en" / "context.txt").exists()


def test_websearch_model_without_grounding_is_refused_before_spending(workspace, monkeypatch, capsys, fake_llms):
    fake_llms["main"].supports_grounding = False

    assert "can't search the web" in fails(monkeypatch, capsys, workspace / "video.ass", "-t")
    assert fake_llms["main"].calls == []


def test_context_file_skips_the_grounding_check(workspace, monkeypatch, fake_llms):
    fake_llms["main"].supports_grounding = False

    run_cli(monkeypatch, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")


@pytest.mark.parametrize("setting", ["batch_lines", "translate_retries"])
def test_settings_below_one_are_rejected(setting):
    with pytest.raises(ValidationError):
        TranslationSettings(**{setting: 0})


def test_bad_config_is_a_plain_error(workspace, monkeypatch, capsys):
    config_setup.load_config()["translation"]["batch_lines"] = 0

    assert "batch_lines" in fails(monkeypatch, capsys, workspace / "video.ass", "-t", "--context", workspace / "ctx.txt")


def test_resuming_a_partial_with_another_model_warns(workspace, monkeypatch, capsys, fake_llms, caplog):
    source, context = workspace / "video.ass", workspace / "ctx.txt"
    sub_utils.write(numbered_subs(4), source)
    config_setup.load_config()["translation"].update(batch_lines=2, discard_lines=0)

    def interrupted_after_first_batch(prompt, count=[]):
        count.append(1)
        if len(count) > 1:
            raise RuntimeError("network down")
        return translate_all(prompt)

    fake_llms["main"] = FakeLLM(interrupted_after_first_batch)
    fails(monkeypatch, capsys, source, "-t", "--context", context)
    fake_llms["other"] = FakeLLM(translate_all, model="other")

    with caplog.at_level(logging.WARNING):
        run_cli(monkeypatch, source, "-t", "--context", context, "--model", "other")

    assert "made with main; the rest will use other" in caplog.text
    assert len(fake_llms["other"].calls) == 1


def test_markdown_is_converted_only_inside_the_range():
    subs = make_subs([(0, 1, "*a*"), (1, 2, "*b*"), (2, 3, "*c*")])

    sub_utils.md_to_ass(subs, LineRange(1, 1))

    assert [e.text for e in subs.events] == ["*a*", r"{\i1}b{\i0}", "*c*"]
