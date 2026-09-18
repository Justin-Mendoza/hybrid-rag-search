"""Deterministic heading enrichment for chunk embedding input."""


def build_embedding_text(
    *,
    content_text: str,
    heading_path: tuple[str, ...],
    leading_heading_block: str | None = None,
) -> str:
    """Add Markdown heading context without embedding the leading heading twice."""

    if not isinstance(content_text, str) or not content_text.strip():
        raise ValueError("Embedding content text must be a nonblank string")
    if not isinstance(heading_path, tuple) or any(
        not isinstance(heading, str) or not heading.strip() for heading in heading_path
    ):
        raise ValueError("Embedding heading path must be a tuple of nonblank strings")

    body = content_text
    if leading_heading_block is not None:
        if not heading_path or leading_heading_block != heading_path[-1]:
            raise ValueError("Leading heading block must match the leaf heading")
        if content_text == leading_heading_block:
            body = ""
        elif content_text.startswith(f"{leading_heading_block}\n\n"):
            body = content_text[len(leading_heading_block) + 2 :]
        else:
            raise ValueError("Leading heading block must be first in the content text")

    heading_text = "\n".join(
        f"{'#' * level} {heading}" for level, heading in enumerate(heading_path, start=1)
    )
    return "\n\n".join(part for part in (heading_text, body) if part)
