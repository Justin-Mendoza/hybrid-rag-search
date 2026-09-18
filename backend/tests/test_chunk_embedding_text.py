import pytest

from hybrid_rag_search.chunking.embedding_text import build_embedding_text


def test_adds_markdown_heading_hierarchy_to_content() -> None:
    assert build_embedding_text(
        content_text="Employees may work remotely.",
        heading_path=("Employee Handbook", "Remote Work"),
    ) == ("# Employee Handbook\n## Remote Work\n\nEmployees may work remotely.")


def test_plain_text_without_headings_is_unchanged() -> None:
    content = "Employees may work remotely."
    assert build_embedding_text(content_text=content, heading_path=()) == content


def test_removes_leading_leaf_heading_only_from_embedding_text() -> None:
    content = "Remote Work\n\nEmployees may work remotely."
    assert build_embedding_text(
        content_text=content,
        heading_path=("Employee Handbook", "Remote Work"),
        leading_heading_block="Remote Work",
    ) == ("# Employee Handbook\n## Remote Work\n\nEmployees may work remotely.")
    assert content == "Remote Work\n\nEmployees may work remotely."


def test_heading_only_chunk_still_has_embedding_text() -> None:
    assert (
        build_embedding_text(
            content_text="Remote Work",
            heading_path=("Employee Handbook", "Remote Work"),
            leading_heading_block="Remote Work",
        )
        == "# Employee Handbook\n## Remote Work"
    )


@pytest.mark.parametrize("content", ["", " ", 42])
def test_rejects_invalid_content(content: object) -> None:
    with pytest.raises(ValueError, match="content text"):
        build_embedding_text(content_text=content, heading_path=())  # type: ignore[arg-type]


@pytest.mark.parametrize("headings", [["Heading"], ("",), (42,)])
def test_rejects_invalid_heading_path(headings: object) -> None:
    with pytest.raises(ValueError, match="heading path"):
        build_embedding_text(  # type: ignore[arg-type]
            content_text="Content",
            heading_path=headings,
        )


def test_rejects_mismatched_leading_heading() -> None:
    with pytest.raises(ValueError, match="match the leaf"):
        build_embedding_text(
            content_text="Another heading\n\nContent",
            heading_path=("Expected heading",),
            leading_heading_block="Another heading",
        )


def test_rejects_heading_not_at_start_of_content() -> None:
    with pytest.raises(ValueError, match="first in the content"):
        build_embedding_text(
            content_text="Introduction\n\nRemote Work",
            heading_path=("Remote Work",),
            leading_heading_block="Remote Work",
        )
