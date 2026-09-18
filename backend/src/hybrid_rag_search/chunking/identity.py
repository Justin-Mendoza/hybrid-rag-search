"""Stable, scoped identities for rebuildable search content."""

import hashlib
import re
from dataclasses import dataclass
from uuid import UUID

IDENTITY_VERSION = "1"


def stable_digest(namespace: str, *values: str) -> str:
    """Hash unambiguous length-prefixed UTF-8 fields in a versioned namespace."""

    digest = hashlib.sha256()
    digest.update(f"hybrid-rag-search/{namespace}/v{IDENTITY_VERSION}\0".encode())
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class DocumentContentIdentity:
    """Authorization scope plus the exact original-file content hash."""

    tenant_id: UUID
    collection_id: UUID
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, UUID):
            raise ValueError("Document content tenant ID must be a UUID")
        if not isinstance(self.collection_id, UUID):
            raise ValueError("Document content collection ID must be a UUID")
        if not isinstance(self.content_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.content_sha256
        ):
            raise ValueError("Document content hash must be a lowercase SHA-256 hex digest")

    @property
    def document_id(self) -> str:
        digest = stable_digest(
            "document",
            str(self.tenant_id),
            str(self.collection_id),
            self.content_sha256,
        )
        return f"doc_{digest}"
