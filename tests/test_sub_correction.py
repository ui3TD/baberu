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


def test_join_text_spaces_only_where_the_script_does():
    assert sub_utils.join_text("はい", "そうです") == "はいそうです"
    assert sub_utils.join_text("OK", "です") == "OKです"
    assert sub_utils.join_text("ですね。", "OK") == "ですね。OK"
    assert sub_utils.join_text("Yes", "sure") == "Yes sure"
    assert sub_utils.join_text("Yes ", "sure") == "Yes sure"
    assert sub_utils.join_text("네", "알겠어요") == "네 알겠어요"
    assert sub_utils.join_text("", "x") == "x"


def test_merge_forward_joins_japanese_without_spaces():
    subs = make_subs([(0, 1, "はい"), (1, 2, "そう"), (2, 3, "です"), (3, 4, "ね"), (4, 3000, "わかりました")])

    assert sub_correction._merge_forward(subs, 4, [0, 1, 2, 3], 25)

    assert subs.events[4].text == "はいそうですねわかりました"
    assert subs.events[4].start == 0


def test_merge_forward_refuses_when_too_fast():
    subs = make_subs([(0, 1, "a" * 20), (1, 2, "b" * 20), (2, 100, "c")])

    assert not sub_correction._merge_forward(subs, 2, [0, 1], 25)
    assert subs.events[2].text == "c"


def test_padding_ignores_formatting_tags():
    plain = make_subs([(0, 100, "x" * 30), (10000, 11000, "next")])
    tagged = make_subs([(0, 100, r"{\i1}" + "x" * 30 + r"{\i0}"), (10000, 11000, "next")])
    padding = PaddingSettings(max_lead_out_sec=5, max_cps=15, min_sec=1)

    sub_correction.apply_timing_standards(plain, padding)
    sub_correction.apply_timing_standards(tagged, padding)

    assert tagged.events[0].end == plain.events[0].end == 2000


def test_redistribute_uses_the_gap_and_keeps_the_previous_line():
    subs = make_subs([(0, 2000, "aaaa")] + [(5000 + i, 5001 + i, "bb") for i in range(4)])

    assert sub_correction._redistribute_backward(subs, 0, [1, 2, 3, 4], 25)

    assert (subs.events[0].start, subs.events[0].end) == (0, 2000)
    assert subs.events[1].start == 2000 and subs.events[4].end == 5004
    assert all(e.end - e.start > 500 for e in subs.events[1:5])


def test_redistribute_shares_with_the_previous_line_when_the_gap_is_too_short():
    subs = make_subs([(0, 4000, "aaaa")] + [(4000 + i, 4001 + i, "bb") for i in range(4)])

    assert sub_correction._redistribute_backward(subs, 0, [1, 2, 3, 4], 25)

    assert subs.events[0].start == 0 and subs.events[0].end < 4000
    assert subs.events[4].end == 4004


def test_redistribute_leaves_lines_with_no_room_to_say_them():
    rows = [(0, 100, "a" * 20)] + [(100 + i, 101 + i, "b" * 20) for i in range(4)]
    subs = make_subs(rows)

    assert not sub_correction._redistribute_backward(subs, 0, [1, 2, 3, 4], 25)

    assert [(e.start, e.end) for e in subs.events] == [(s, e) for s, e, _ in rows]


def test_splice_replaces_an_inclusive_range():
    subs = make_subs([(0, 1, "a"), (1, 2, "b"), (2, 3, "c")])

    sub_utils.splice(subs, LineRange(0, 1), make_subs([(0, 2, "new")]))

    assert [e.text for e in subs.events] == ["new", "c"]


def test_md_to_ass_converts_emphasis():
    subs = make_subs([(0, 1, "*i* **b**")])

    assert sub_utils.md_to_ass(subs).events[0].text == r"{\i1}i{\i0} {\b1}b{\b0}"
