from pathlib import Path
from uuid import uuid4

import pytest

from hybrid_rag_search.ingestion.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactKind,
    ArtifactRef,
    LocalArtifactStorage,
    artifact_identity,
    artifact_key,
    canonical_json_bytes,
)


def reference_key(kind: ArtifactKind = ArtifactKind.CHUNKS) -> str:
    return artifact_key(uuid4(), uuid4(), kind, artifact_identity(kind, "input"))


def test_canonical_json_and_identity_are_deterministic() -> None:
    first = canonical_json_bytes({"b": [2, 1], "a": "café"})
    second = canonical_json_bytes({"a": "café", "b": [2, 1]})
    assert first == second == b'{"a":"caf\xc3\xa9","b":[2,1]}'
    assert artifact_identity(ArtifactKind.CHUNKS, "doc", "config") == artifact_identity(
        ArtifactKind.CHUNKS, "doc", "config"
    )
    assert artifact_identity(ArtifactKind.CHUNKS, "doc") != artifact_identity(
        ArtifactKind.EMBEDDINGS, "doc"
    )


def test_artifact_key_is_scoped_and_canonical() -> None:
    tenant_id, job_id = uuid4(), uuid4()
    identity = "a" * 64
    assert artifact_key(tenant_id, job_id, ArtifactKind.PARSED, identity) == (
        f"{tenant_id}/{job_id}/parsed/{identity}.json"
    )


def test_save_is_create_only_and_reuses_identical_bytes(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path)
    key = reference_key()
    content = canonical_json_bytes({"chunks": ["one", "two"]})
    first = storage.save(key, content)
    second = storage.save(key, content)
    assert first == second
    assert storage.fetch(first) == content
    assert first.checkpoint_value() == {
        "key": key,
        "sha256": first.sha256,
        "size_bytes": len(content),
    }
    assert ArtifactRef.from_checkpoint(first.checkpoint_value()) == first
    assert len(list((storage.root / key).parent.iterdir())) == 1


@pytest.mark.parametrize(
    "value",
    [
        None,
        "reference",
        {},
        {"key": "missing-fields"},
        {"key": reference_key(), "sha256": "a" * 64, "size_bytes": 1, "extra": True},
        {"key": "invalid", "sha256": "a" * 64, "size_bytes": 1},
    ],
)
def test_checkpoint_reference_requires_exact_valid_shape(value: object) -> None:
    with pytest.raises(ValueError, match="checkpoint reference"):
        ArtifactRef.from_checkpoint(value)


def test_lookup_recovers_reference_for_existing_artifact(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path)
    key = reference_key()
    saved = storage.save(key, b"recoverable")
    recovered = storage.lookup(key)
    assert recovered == saved
    assert recovered is not None
    assert storage.fetch(recovered) == b"recoverable"


def test_lookup_returns_none_when_artifact_is_absent(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path)
    assert storage.lookup(reference_key()) is None


def test_same_key_with_different_bytes_is_a_conflict(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path)
    key = reference_key()
    storage.save(key, b"first")
    with pytest.raises(ArtifactConflictError, match="different bytes"):
        storage.save(key, b"second")


def test_fetch_detects_tampering(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path)
    reference = storage.save(reference_key(), b"original")
    (storage.root / reference.key).write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="checksum or size"):
        storage.fetch(reference)


def test_delete_is_idempotent(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path)
    reference = storage.save(reference_key(), b"artifact")
    storage.delete(reference)
    storage.delete(reference)
    with pytest.raises(FileNotFoundError):
        storage.fetch(reference)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("key", "../escape", "key"),
        ("key", 42, "key"),
        ("sha256", "bad", "hash"),
        ("size_bytes", -1, "size"),
        ("size_bytes", True, "size"),
    ],
)
def test_reference_validation(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "key": reference_key(),
        "sha256": "a" * 64,
        "size_bytes": 1,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ArtifactRef(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        ("tenant", uuid4(), ArtifactKind.PARSED, "a" * 64),
        (uuid4(), "job", ArtifactKind.PARSED, "a" * 64),
        (uuid4(), uuid4(), "parsed", "a" * 64),
        (uuid4(), uuid4(), ArtifactKind.PARSED, "bad"),
    ],
)
def test_key_rejects_invalid_parts(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        artifact_key(*arguments)  # type: ignore[arg-type]


def test_identity_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        artifact_identity("parsed", "input")  # type: ignore[arg-type]


def test_operations_require_bytes_and_references(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path)
    with pytest.raises(ValueError, match="bytes"):
        storage.save(reference_key(), "text")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ArtifactRef"):
        storage.fetch("key")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ArtifactRef"):
        storage.delete("key")  # type: ignore[arg-type]


def test_storage_rejects_invalid_key_even_with_valid_bytes(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path)
    with pytest.raises(ValueError, match="Invalid artifact key"):
        storage.save("../escape", b"content")
    with pytest.raises(ValueError, match="Invalid artifact key"):
        storage.lookup("../escape")


def test_reference_rejects_uuid_shaped_but_invalid_key() -> None:
    key = f"{'-' * 36}/{uuid4()}/parsed/{'a' * 64}.json"
    with pytest.raises(ValueError, match="key"):
        ArtifactRef(key, "b" * 64, 1)


def test_rejects_symlinked_artifact_path(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path / "artifacts")
    key = reference_key()
    tenant_directory = storage.root / key.split("/")[0]
    outside = tmp_path / "outside"
    outside.mkdir()
    tenant_directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        storage.save(key, b"secret")
    assert list(outside.iterdir()) == []


def test_rejects_symlinked_artifact_file(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(tmp_path / "artifacts")
    key = reference_key()
    path = storage.root / key
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="Symlinks"):
        storage.save(key, b"secret")
    assert outside.read_bytes() == b"outside"


def test_concurrent_identical_publish_is_reused(tmp_path: Path, monkeypatch) -> None:
    storage = LocalArtifactStorage(tmp_path)
    key = reference_key()
    content = b"artifact"

    def concurrent_link(source: str, destination: Path) -> None:
        destination.write_bytes(Path(source).read_bytes())
        raise FileExistsError

    monkeypatch.setattr("hybrid_rag_search.ingestion.artifacts.os.link", concurrent_link)
    reference = storage.save(key, content)
    assert storage.fetch(reference) == content
    assert len(list((storage.root / key).parent.iterdir())) == 1
