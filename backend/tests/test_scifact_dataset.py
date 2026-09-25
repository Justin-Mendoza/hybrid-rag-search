from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_rag_search.evaluation import scifact
from hybrid_rag_search.evaluation.datasets import load_dataset


def _archive(path: Path) -> bytes:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "scifact/corpus.jsonl",
            '{"_id":"20","title":"B","text":"second","metadata":{}}\n'
            '{"_id":"10","title":"A","text":"first","metadata":{}}\n',
        )
        archive.writestr(
            "scifact/queries.jsonl",
            '{"_id":"2","text":"unused","metadata":{}}\n{"_id":"1","text":"claim","metadata":{}}\n',
        )
        archive.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\n1\t10\t1\n")
    return path.read_bytes()


def test_converts_archive_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "scifact.zip"
    data = _archive(archive)
    monkeypatch.setattr(scifact, "SCIFACT_ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())

    first = scifact.convert_archive(archive, tmp_path / "first")
    second = scifact.convert_archive(archive, tmp_path / "second")
    loaded = load_dataset(first)

    assert loaded == load_dataset(second)
    assert [record.id for record in loaded.corpus] == ["scifact-doc-10", "scifact-doc-20"]
    assert [query.id for query in loaded.queries] == ["scifact-query-1"]
    assert loaded.judgments[0].relevance_grade == 3
    assert "version=beir-2021-03-19-test-v1" in scifact._summary(first)


def test_rejects_wrong_archive_hash(tmp_path: Path) -> None:
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        scifact.convert_archive(archive, tmp_path / "output")


def test_download_verifies_and_moves_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"archive"
    monkeypatch.setattr(scifact, "SCIFACT_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())

    response = SimpleNamespace(content=payload, raise_for_status=lambda: None)
    monkeypatch.setattr(scifact.httpx, "get", lambda *_args, **_kwargs: response)
    destination = tmp_path / "nested/scifact.zip"
    assert scifact.download_archive(destination) == destination
    assert destination.read_bytes() == payload


def test_download_rejects_wrong_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(content=b"wrong", raise_for_status=lambda: None)
    monkeypatch.setattr(scifact.httpx, "get", lambda *_args, **_kwargs: response)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        scifact.download_archive(tmp_path / "scifact.zip")
    assert not (tmp_path / "scifact.partial").exists()


def test_main_download_and_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "fixture.zip"
    data = _archive(archive)
    monkeypatch.setattr(scifact, "SCIFACT_ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    output = tmp_path / "dataset"

    def fake_download(destination: Path) -> Path:
        shutil.copy(archive, destination)
        return destination

    monkeypatch.setattr(scifact, "download_archive", fake_download)
    monkeypatch.setattr("sys.argv", ["scifact", "download", "--output", str(output)])
    scifact.main()
    assert "corpus=2 queries=1 judgments=1" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["scifact", "validate", "--output", str(output)])
    scifact.main()
    result = capsys.readouterr().out
    assert "dataset=scifact" in result
    assert "dataset_hash=" in result
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["archive_sha256"] == hashlib.sha256(data).hexdigest()
