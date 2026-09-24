"""Shared metadata normalization and OpenSearch retrieval filters."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


def normalize_filter_text(value: str, label: str) -> str:
    """Return the canonical exact-match representation for supported metadata."""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = " ".join(value.split()).casefold()
    if not normalized:
        raise ValueError(f"{label} must be nonblank")
    return normalized


def parse_source_date(value: str, label: str = "Source date") -> datetime:
    """Parse one timezone-aware ISO-8601 value and normalize it to UTC."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank ISO-8601 string")
    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(
            candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
        )
    except ValueError as error:
        raise ValueError(f"{label} must be a valid ISO-8601 date-time") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def format_source_date(value: datetime) -> str:
    """Serialize a timezone-aware date-time in the index's canonical UTC form."""

    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Source date boundary must be a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class IndexedSourceMetadata:
    """Original document metadata plus its supported typed filter values."""

    original: dict[str, object]
    source_type: str | None
    author: str | None
    source_date: str | None

    @classmethod
    def from_document(cls, metadata: dict[str, object]) -> "IndexedSourceMetadata":
        if not isinstance(metadata, dict):
            raise ValueError("Document source metadata must be an object")
        source_type = _optional_text(metadata, "source_type", "Source type")
        author = _optional_text(metadata, "author", "Author")
        raw_date = metadata.get("source_date")
        source_date = None
        if raw_date is not None:
            if not isinstance(raw_date, str):
                raise ValueError("Source date must be a string")
            source_date = format_source_date(parse_source_date(raw_date))
        return cls(dict(metadata), source_type, author, source_date)

    def indexed_fields(self) -> dict[str, str]:
        return {
            key: value
            for key, value in (
                ("source_type", self.source_type),
                ("author", self.author),
                ("source_date", self.source_date),
            )
            if value is not None
        }


def _optional_text(metadata: dict[str, object], key: str, label: str) -> str | None:
    value = metadata.get(key)
    return None if value is None else normalize_filter_text(value, label)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RetrievalFilters:
    """One validated scope translated identically for every retrieval path."""

    tenant_id: UUID
    collection_id: UUID | None = None
    source_type: str | None = None
    author: str | None = None
    source_date_from: datetime | None = None
    source_date_to: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, UUID):
            raise ValueError("Retrieval tenant ID must be a UUID")
        if self.collection_id is not None and not isinstance(self.collection_id, UUID):
            raise ValueError("Retrieval collection ID must be a UUID when present")
        for field, label in (("source_type", "Source type"), ("author", "Author")):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, normalize_filter_text(value, label))
        for field in ("source_date_from", "source_date_to"):
            value = getattr(self, field)
            if value is not None:
                canonical = format_source_date(value)
                object.__setattr__(self, field, parse_source_date(canonical))
        if (
            self.source_date_from is not None
            and self.source_date_to is not None
            and self.source_date_from >= self.source_date_to
        ):
            raise ValueError("Source date start must be earlier than source date end")

    def clauses(self) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = [
            {"term": {"tenant_id": str(self.tenant_id)}},
            {"term": {"visibility": "ready"}},
        ]
        if self.collection_id is not None:
            clauses.append({"term": {"collection_id": str(self.collection_id)}})
        if self.source_type is not None:
            clauses.append({"term": {"source_type": self.source_type}})
        if self.author is not None:
            clauses.append({"term": {"author": self.author}})
        bounds: dict[str, str] = {}
        if self.source_date_from is not None:
            bounds["gte"] = format_source_date(self.source_date_from)
        if self.source_date_to is not None:
            bounds["lt"] = format_source_date(self.source_date_to)
        if bounds:
            clauses.append({"range": {"source_date": bounds}})
        return clauses

    def trace_values(self) -> dict[str, str]:
        values = {
            "tenant_id": str(self.tenant_id),
            "visibility": "ready",
        }
        if self.collection_id is not None:
            values["collection_id"] = str(self.collection_id)
        if self.source_type is not None:
            values["source_type"] = self.source_type
        if self.author is not None:
            values["author"] = self.author
        if self.source_date_from is not None:
            values["source_date_from"] = format_source_date(self.source_date_from)
        if self.source_date_to is not None:
            values["source_date_to"] = format_source_date(self.source_date_to)
        return values
