"""Shared deterministic text normalization for document parsers."""


def normalize_paragraphs(text: str) -> tuple[str, ...]:
    """Normalize line endings/trailing spaces and split on blank lines."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip(" \t")
        if line.strip():
            lines.append(line)
        elif lines:
            paragraphs.append("\n".join(lines))
            lines = []
    if lines:
        paragraphs.append("\n".join(lines))
    return tuple(paragraphs)
