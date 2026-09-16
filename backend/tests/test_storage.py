from pathlib import Path
from uuid import uuid4

import pytest

from hybrid_rag_search.storage import LocalFileStorage, storage_key


def test_immutable_save_fetch_delete(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "originals")
    key = storage_key(uuid4(), uuid4())
    storage.save(key, b"original\x00bytes")
    assert storage.fetch(key) == b"original\x00bytes"
    with pytest.raises(FileExistsError):
        storage.save(key, b"replacement")
    assert storage.fetch(key) == b"original\x00bytes"
    assert len(list((storage.root / key).parent.iterdir())) == 1
    storage.delete(key)
    storage.delete(key)
    with pytest.raises(FileNotFoundError):
        storage.fetch(key)


@pytest.mark.parametrize(
    "key", ["../escape", "/tmp/escape", "a/b", "", "a\\b", "-" * 36 + "/" + "-" * 36]
)
@pytest.mark.parametrize("operation", ["save", "fetch", "delete"])
def test_rejects_invalid_keys(tmp_path: Path, key: str, operation: str) -> None:
    storage = LocalFileStorage(tmp_path)
    with pytest.raises(ValueError):
        if operation == "save":
            storage.save(key, b"bad")
        else:
            getattr(storage, operation)(key)


@pytest.mark.parametrize("symlink_parent", [True, False])
def test_rejects_symlinks(tmp_path: Path, symlink_parent: bool) -> None:
    storage = LocalFileStorage(tmp_path / "originals")
    key = storage_key(uuid4(), uuid4())
    target = tmp_path / "outside"
    target.mkdir()
    path = storage.root / key
    if symlink_parent:
        path.parent.symlink_to(target, target_is_directory=True)
    else:
        path.parent.mkdir()
        path.symlink_to(target / "secret")
    for operation in (
        lambda: storage.save(key, b"bad"),
        lambda: storage.fetch(key),
        lambda: storage.delete(key),
    ):
        with pytest.raises(ValueError, match="Symlinks"):
            operation()
    assert list(target.iterdir()) == []


def test_failed_publish_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = LocalFileStorage(tmp_path)
    key = storage_key(uuid4(), uuid4())

    def fail(*_args: object) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr("hybrid_rag_search.storage.os.link", fail)
    with pytest.raises(OSError):
        storage.save(key, b"data")
    assert list((tmp_path / key).parent.iterdir()) == []


def test_noncanonical_uuid_key(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        storage.fetch(f"{'a' * 32}----/{uuid4()}")
