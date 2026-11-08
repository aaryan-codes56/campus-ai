"""Citation checking on a stream."""

import pytest

from app.generation.citations import CitationFilter


def run(parts: list[str], sources: int = 3) -> tuple[str, list[int]]:
    citation_filter = CitationFilter(sources)
    output = "".join(citation_filter.feed(part) for part in parts) + citation_filter.finish()
    return output, citation_filter.cited


def test_plain_text_passes_through() -> None:
    assert run(["Hello ", "world."]) == ("Hello world.", [])


def test_valid_marker_is_kept_and_recorded() -> None:
    assert run(["Fees are due [2]."]) == ("Fees are due [2].", [2])


@pytest.mark.parametrize(
    "parts",
    [["Due [", "2]."], ["Due [2", "]."], ["Due ", "[", "2", "]", "."], ["D", "ue [2]", "."]],
)
def test_marker_split_across_chunks_is_reassembled(parts: list[str]) -> None:
    assert run(parts) == ("Due [2].", [2])


def test_text_is_held_back_only_while_a_marker_is_possible() -> None:
    citation_filter = CitationFilter(3)
    assert citation_filter.feed("Due [") == "Due "
    assert citation_filter.feed("2] soon") == "[2] soon"


def test_made_up_source_number_is_removed() -> None:
    assert run(["Due [9] soon."]) == ("Due  soon.", [])


def test_only_invalid_numbers_are_dropped_from_a_group() -> None:
    assert run(["Due [1, 9, 3]."]) == ("Due [1][3].", [1, 3])


@pytest.mark.parametrize("marker", ["[1, 2]", "[1;2]", "[Source 1, Source 2]", "[source 1, 2]"])
def test_marker_variants_are_normalised(marker: str) -> None:
    assert run([f"Due {marker}."]) == ("Due [1][2].", [1, 2])


def test_zero_is_not_a_source() -> None:
    assert run(["Due [0]."]) == ("Due .", [])


def test_citations_are_recorded_once_in_first_use_order() -> None:
    assert run(["A [2] B [1] C [2]."])[1] == [2, 1]


@pytest.mark.parametrize("text", ["See [AMENDED] here.", "An array[index] here.", "Open [ bracket"])
def test_other_brackets_are_not_touched(text: str) -> None:
    assert run([text]) == (text, [])


def test_unfinished_marker_at_the_end_is_released() -> None:
    assert run(["Due [1"]) == ("Due [1", [])


def test_long_bracketed_text_is_not_held_forever() -> None:
    citation_filter = CitationFilter(3)
    long_text = "[" + "a" * 60
    assert citation_filter.feed(long_text) == long_text


def test_with_no_sources_every_marker_is_removed() -> None:
    assert run(["Due [1]."], sources=0) == ("Due .", [])
