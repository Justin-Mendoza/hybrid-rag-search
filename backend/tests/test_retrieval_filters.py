from datetime import UTC, datetime
from uuid import uuid4

import pytest

from hybrid_rag_search.retrieval_filters import (
    IndexedSourceMetadata,
    RetrievalFilters,
    parse_source_date,
)


def test_indexed_metadata_preserves_original_and_normalizes_supported_fields() -> None:
    original: dict[str, object] = {
        "source_type": " Email ",
        "author": " Ada   Lovelace ",
        "source_date": "2026-09-24T09:30:00-04:00",
        "thread_id": "thread-7",
    }
    metadata = IndexedSourceMetadata.from_document(original)

    assert metadata.original == original
    assert metadata.original is not original
    assert metadata.indexed_fields() == {
        "source_type": "email",
        "author": "ada lovelace",
        "source_date": "2026-09-24T13:30:00Z",
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {"source_type": 7},
        {"author": "  "},
        {"source_date": "yesterday"},
        {"source_date": "2026-09-24T09:30:00"},
    ],
)
def test_indexed_metadata_rejects_invalid_supported_values(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        IndexedSourceMetadata.from_document(metadata)


def test_metadata_and_date_container_types_are_validated() -> None:
    with pytest.raises(ValueError, match="object"):
        IndexedSourceMetadata.from_document([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="string"):
        IndexedSourceMetadata.from_document({"source_date": 7})
    with pytest.raises(ValueError, match="nonblank"):
        parse_source_date("")


def test_filters_build_exact_and_half_open_date_clauses() -> None:
    tenant_id, collection_id = uuid4(), uuid4()
    filters = RetrievalFilters(
        tenant_id,
        collection_id,
        " Email ",
        " Ada  Lovelace ",
        datetime(2026, 9, 1, tzinfo=UTC),
        datetime(2026, 10, 1, tzinfo=UTC),
    )

    assert filters.clauses() == [
        {"term": {"tenant_id": str(tenant_id)}},
        {"term": {"visibility": "ready"}},
        {"term": {"collection_id": str(collection_id)}},
        {"term": {"source_type": "email"}},
        {"term": {"author": "ada lovelace"}},
        {
            "range": {
                "source_date": {
                    "gte": "2026-09-01T00:00:00Z",
                    "lt": "2026-10-01T00:00:00Z",
                }
            }
        },
    ]
    assert filters.trace_values()["source_date_to"] == "2026-10-01T00:00:00Z"


def test_filters_validate_scope_and_boundaries() -> None:
    tenant_id = uuid4()
    with pytest.raises(ValueError, match="tenant"):
        RetrievalFilters("tenant")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="collection"):
        RetrievalFilters(tenant_id, "collection")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="earlier"):
        RetrievalFilters(
            tenant_id,
            source_date_from=datetime(2026, 10, 1, tzinfo=UTC),
            source_date_to=datetime(2026, 9, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="timezone"):
        RetrievalFilters(tenant_id, source_date_from=datetime(2026, 9, 1))
    assert parse_source_date("2026-09-24T09:30:00-04:00") == datetime(
        2026, 9, 24, 13, 30, tzinfo=UTC
    )
