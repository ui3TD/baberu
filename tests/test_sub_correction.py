from baberu.settings import MistimedSegmentSettings, PaddingSettings
from baberu.subtitling import sub_correction, sub_utils
from baberu.subtitling.types import LineRange
from conftest import make_subs


def test_remove_empty_drops_blank_lines():
    subs = make_subs([(0, 1000, "a"), (1000, 2000, "  "), (2000, 3000, "b")])

    assert [e.text for e in sub_correction.remove_empty(subs).events] == ["a", "b"]


def test_remove_empty_respects_segment():
    subs = make_subs([(0, 1, ""), (1, 2, ""), (2, 3, "c")])

    assert len(sub_correction.remove_empty(subs, LineRange(1, 2)).events) == 2


def test_short_lines_are_detected_as_a_group():
    rows = [(0, 5000, "ok")] + [(5000 + i, 5001 + i, f"s{i}") for i in range(5)] + [(6000, 20000, "ok")]

    mistimed = sub_correction.find_mistimed_lines(make_subs(rows), 0.1, MistimedSegmentSettings())

    assert {1, 2, 3, 4, 5} <= mistimed


def test_fix_redistributes_time_by_text_length():
    rows = [(0, 4000, "aaaa")] + [(4000 + i, 4001 + i, "bb") for i in range(4)] + [(9000, 30000, "c" * 99)]

    subs = sub_correction.fix_mistimed_lines(make_subs(rows), 0.1, MistimedSegmentSettings())

    durations = [e.end - e.start for e in subs.events[:5]]
    assert durations[0] > durations[1]
    assert subs.events[0].start == 0


def test_timing_standards_extend_short_lines_without_overlap():
    subs = make_subs([(0, 100, "x" * 30), (1000, 5000, "next")])

    sub_correction.apply_timing_standards(subs, PaddingSettings(max_lead_out_sec=5, max_cps=15, min_sec=1))

    assert subs.events[0].end == 1000


def test_splice_replaces_an_inclusive_range():
    subs = make_subs([(0, 1, "a"), (1, 2, "b"), (2, 3, "c")])

    sub_utils.splice(subs, LineRange(0, 1), make_subs([(0, 2, "new")]))

    assert [e.text for e in subs.events] == ["new", "c"]


def test_md_to_ass_converts_emphasis():
    subs = make_subs([(0, 1, "*i* **b**")])

    assert sub_utils.md_to_ass(subs).events[0].text == r"{\i1}i{\i0} {\b1}b{\b0}"
