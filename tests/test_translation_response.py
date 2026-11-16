from baberu import constants
from baberu.subtitling.translation_response import parse_response, problems
from conftest import make_subs


def test_blank_lines_between_entries_are_ignored():
    assert parse_response("1. a\n\n2. b\n\n\n3. c", 3) == ["a", "b", "c"]


def test_preamble_before_the_first_entry_is_dropped():
    assert parse_response("Here you go:\n1. a\n2. b", 2) == ["a", "b"]


def test_code_fences_and_thinking_are_dropped():
    text = "<think>\n1. not this\n</think>\n```text\n1. a\n2. b\n```"
    assert parse_response(text, 2) == ["a", "b"]


def test_other_numbering_styles_are_read():
    assert parse_response("1) a\n2: b\n3. c", 3) == ["a", "b", "c"]


def test_a_missing_number_stays_empty_in_its_place():
    assert parse_response("1. a\n3. c", 3) == ["a", None, "c"]


def test_a_repeated_number_keeps_the_first():
    assert parse_response("1. a\n2. b\n2. again\n3. c", 3) == ["a", "b", "c"]


def test_unnumbered_line_after_an_entry_is_its_second_line():
    assert parse_response("1. a\nmore\n2. b", 2) == ["a\\Nmore", "b"]


def test_unnumbered_response_with_the_exact_count_is_read_by_position():
    assert parse_response("a\nb", 2) == ["a", "b"]


def test_unnumbered_response_with_the_wrong_count_is_all_missing():
    assert parse_response("a\nb\nc", 2) == [None, None]


def test_a_number_inside_text_is_not_an_entry():
    assert parse_response("1. a\n1.5 million\n2. b", 2) == ["a\\N1.5 million", "b"]


def test_numbers_out_of_range_are_ignored():
    assert parse_response("1. a\n2. b\n3. extra", 2) == ["a", "b"]


def test_entries_are_stripped_of_spaces_and_the_continue_flag():
    assert parse_response(f"1.   a{constants.CONTINUE_FLAG}  \n2. b", 2) == ["a", "b"]


def test_matching_ellipses_are_removed_after_parsing():
    assert parse_response("1. so...\n2. ...then", 2) == ["so", "then"]


def test_problems_only_count_the_kept_lines():
    batch = make_subs([(0, 1, "あ"), (1, 2, "い"), (2, 3, "う"), (3, 4, "え")]).events

    found = problems(["A", "", "う", None], batch, kept=3)

    assert found == {1: "empty", 2: "untranslated"}


def test_problems_find_missing_lines():
    batch = make_subs([(0, 1, "あ"), (1, 2, "い")]).events

    assert problems([None, "I"], batch, kept=2) == {0: "missing"}


def test_unchanged_lines_are_fine_without_letters_or_in_the_same_language():
    batch = make_subs([(0, 1, "123"), (1, 2, "hello")]).events

    assert problems(["123", "hello"], batch, kept=2, same_language=True) == {}
    assert problems(["123", "T"], batch, kept=2) == {}
