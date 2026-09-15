"""Replaceable storage for immutable originals; keys never contain upload filenames."""

import os
import re
import tempfile
from pathlib import Path
from typing import Protocol
from uuid import UUID


def storage_key(tenant_id: UUID, document_id: UUID) -> str:
    return f"{tenant_id}/{document_id}"


class FileStorage(Protocol):
    def save(self, key: str, content: bytes) -> None: ...

    def fetch(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


class LocalFileStorage:
    """Atomic, create-only writes under a trusted, application-owned directory.

    Reject symlink components and noncanonical keys. The root and its ancestors
    must not be writable by untrusted local processes (including during calls).
    """

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f-]{36}/[0-9a-f-]{36}", key):
            raise ValueError("Invalid storage key")
        if any(str(UUID(part)) != part for part in key.split("/")):
            raise ValueError("Storage key must contain canonical UUIDs")
        path = self.root / key
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("Symlinks are not allowed in storage keys")
        return path

    def save(self, key: str, content: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(mode=0o700, exist_ok=True)
        # Publish only a complete, synced file, without overwriting an original.
        with tempfile.NamedTemporaryFile(dir=path.parent) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.link(temporary.name, path)

    def fetch(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
