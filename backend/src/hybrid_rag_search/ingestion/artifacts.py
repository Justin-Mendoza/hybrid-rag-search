"""Immutable, checksum-verified ingestion checkpoint artifacts."""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import UUID

from hybrid_rag_search.chunking.identity import stable_digest


class ArtifactKind(StrEnum):
    PARSED = "parsed"
    CHUNKS = "chunks"
    EMBEDDINGS = "embeddings"


class ArtifactConflictError(RuntimeError):
    """A deterministic artifact key already contains different bytes."""


class ArtifactIntegrityError(RuntimeError):
    """Stored artifact bytes no longer match their durable reference."""


@dataclass(frozen=True)
class ArtifactRef:
    key: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not _valid_key(self.key):
            raise ValueError("Artifact reference key is invalid")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Artifact reference hash must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("Artifact reference size must be a nonnegative integer")

    def checkpoint_value(self) -> dict[str, object]:
        return {"key": self.key, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_checkpoint(cls, value: object) -> "ArtifactRef":
        if not isinstance(value, dict) or set(value) != {"key", "sha256", "size_bytes"}:
            raise ValueError("Artifact checkpoint reference is invalid")
        try:
            return cls(
                key=value["key"],
                sha256=value["sha256"],
                size_bytes=value["size_bytes"],
            )
        except ValueError:
            raise ValueError("Artifact checkpoint reference is invalid") from None


class ArtifactStorage(Protocol):
    def save(self, key: str, content: bytes) -> ArtifactRef: ...

    def lookup(self, key: str) -> ArtifactRef | None: ...

    def fetch(self, reference: ArtifactRef) -> bytes: ...

    def delete(self, reference: ArtifactRef) -> None: ...


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def artifact_identity(kind: ArtifactKind, *values: str) -> str:
    if not isinstance(kind, ArtifactKind):
        raise ValueError("Artifact kind must be a supported ingestion stage")
    return stable_digest(f"ingestion-artifact/{kind.value}", *values)


def artifact_key(
    tenant_id: UUID,
    job_id: UUID,
    kind: ArtifactKind,
    identity: str,
) -> str:
    if not isinstance(tenant_id, UUID) or not isinstance(job_id, UUID):
        raise ValueError("Artifact tenant and job IDs must be UUIDs")
    if not isinstance(kind, ArtifactKind):
        raise ValueError("Artifact kind must be a supported ingestion stage")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("Artifact identity must be a lowercase SHA-256 digest")
    return f"{tenant_id}/{job_id}/{kind.value}/{identity}.json"


def _valid_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    match = re.fullmatch(
        r"([0-9a-f-]{36})/([0-9a-f-]{36})/(parsed|chunks|embeddings)/([0-9a-f]{64})\.json",
        key,
    )
    if match is None:
        return False
    try:
        return str(UUID(match[1])) == match[1] and str(UUID(match[2])) == match[2]
    except ValueError:
        return False


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class LocalArtifactStorage:
    """Publish immutable artifacts atomically beneath an application-owned root."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    def _path(self, key: str) -> Path:
        if not _valid_key(key):
            raise ValueError("Invalid artifact key")
        path = self.root / key
        current = path.parent
        while current != self.root:
            if current.is_symlink():
                raise ValueError("Symlinks are not allowed in artifact keys")
            current = current.parent
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in artifact keys")
        return path

    def _existing(self, path: Path, expected: ArtifactRef) -> ArtifactRef:
        content = path.read_bytes()
        if len(content) != expected.size_bytes or _sha256(content) != expected.sha256:
            raise ArtifactConflictError("Artifact key already contains different bytes")
        return expected

    def save(self, key: str, content: bytes) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise ValueError("Artifact content must be bytes")
        path = self._path(key)
        reference = ArtifactRef(key, _sha256(content), len(content))
        if path.exists():
            return self._existing(path, reference)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._path(key)
        with tempfile.NamedTemporaryFile(dir=path.parent) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            try:
                os.link(temporary.name, path)
            except FileExistsError:
                return self._existing(path, reference)
        return reference

    def lookup(self, key: str) -> ArtifactRef | None:
        """Describe an existing deterministic artifact after an uncheckpointed write."""

        path = self._path(key)
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return None
        return ArtifactRef(key, _sha256(content), len(content))

    def fetch(self, reference: ArtifactRef) -> bytes:
        if not isinstance(reference, ArtifactRef):
            raise ValueError("Artifact fetch requires an ArtifactRef")
        content = self._path(reference.key).read_bytes()
        if len(content) != reference.size_bytes or _sha256(content) != reference.sha256:
            raise ArtifactIntegrityError("Artifact checksum or size does not match its reference")
        return content

    def delete(self, reference: ArtifactRef) -> None:
        if not isinstance(reference, ArtifactRef):
            raise ValueError("Artifact delete requires an ArtifactRef")
        self._path(reference.key).unlink(missing_ok=True)
