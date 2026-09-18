from typing import Any
from uuid import UUID

import pytest

from hybrid_rag_search.chunking.identity import DocumentContentIdentity

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
COLLECTION_ID = UUID("22222222-2222-2222-2222-222222222222")
CONTENT_HASH = "a" * 64


def identity(
    *,
    tenant_id: UUID = TENANT_ID,
    collection_id: UUID = COLLECTION_ID,
    content_hash: str = CONTENT_HASH,
) -> DocumentContentIdentity:
    return DocumentContentIdentity(tenant_id, collection_id, content_hash)


def test_document_id_is_stable_and_namespaced() -> None:
    first = identity()
    assert first == identity()
    assert first.document_id == identity().document_id
    assert first.document_id == (
        "doc_35644f8353602df6cf1cd0ee0631383957353d6d2d60a2855380501b7f3e7e09"
    )
    assert len(first.document_id) == 68


def test_document_id_changes_with_tenant_scope() -> None:
    other = identity(tenant_id=UUID("33333333-3333-3333-3333-333333333333"))
    assert other.document_id != identity().document_id


def test_document_id_changes_with_collection_scope() -> None:
    other = identity(collection_id=UUID("33333333-3333-3333-3333-333333333333"))
    assert other.document_id != identity().document_id


def test_document_id_changes_with_original_content() -> None:
    assert identity(content_hash="b" * 64).document_id != identity().document_id


@pytest.mark.parametrize("tenant_id", [str(TENANT_ID), None, 42])
def test_tenant_id_must_be_uuid(tenant_id: Any) -> None:
    with pytest.raises(ValueError, match="tenant ID"):
        DocumentContentIdentity(tenant_id, COLLECTION_ID, CONTENT_HASH)


@pytest.mark.parametrize("collection_id", [str(COLLECTION_ID), None, 42])
def test_collection_id_must_be_uuid(collection_id: Any) -> None:
    with pytest.raises(ValueError, match="collection ID"):
        DocumentContentIdentity(TENANT_ID, collection_id, CONTENT_HASH)


@pytest.mark.parametrize(
    "content_hash",
    ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, None, 42],
)
def test_content_hash_must_be_canonical_sha256(content_hash: Any) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        DocumentContentIdentity(TENANT_ID, COLLECTION_ID, content_hash)
