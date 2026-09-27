from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from hybrid_rag_search.evaluation.datasets import (
    CorpusRecord,
    DatasetManifest,
    DatasetValidationError,
    load_dataset,
)

ROOT = Path(__file__).parents[2]
SYNTHETIC = ROOT / "datasets/synthetic-workspace/v1"


def _copy_dataset(tmp_path: Path) -> Path:
    destination = tmp_path / "dataset"
    shutil.copytree(SYNTHETIC, destination)
    return destination


def _rehash(directory: Path, filename: str) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    next(item for item in manifest["files"] if item["path"] == filename)["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))


def test_loads_synthetic_dataset_deterministically() -> None:
    first = load_dataset(SYNTHETIC)
    second = load_dataset(SYNTHETIC)

    assert first == second
    assert first.manifest.version == "1"
    assert len(first.corpus) == 8
    assert len(first.queries) == 12
    assert {query.category for query in first.queries} == {
        "exact-match",
        "paraphrase",
        "stale-version",
        "conflicting-evidence",
    }
    assert first.corpus_hash == "f88d309e8d848c75580779b6bfaf1920fc112d64e097532bad62040be2685951"
    assert len(first.dataset_hash) == 64


def test_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    directory = _copy_dataset(tmp_path)
    with (directory / "corpus.jsonl").open("a") as output:
        output.write(" ")

    with pytest.raises(DatasetValidationError, match="SHA-256 mismatch"):
        load_dataset(directory)


@pytest.mark.parametrize(
    ("filename", "replacement", "message"),
    [
        ("queries.jsonl", "not-json\n", "queries.jsonl:1"),
        ("queries.jsonl", "[]\n", "expected an object"),
        ("queries.jsonl", "\n", "blank lines"),
        (
            "judgments.jsonl",
            '{"query_id":"missing","target_type":"document","target_id":"benefits-guide","relevance_grade":3}\n',
            "unknown query",
        ),
        (
            "judgments.jsonl",
            '{"query_id":"q-exact-policy-id","target_type":"document","target_id":"missing","relevance_grade":3}\n',
            "unknown target",
        ),
        (
            "judgments.jsonl",
            '{"query_id":"q-exact-policy-id","target_type":"chunk","target_id":"benefits-guide","relevance_grade":3}\n',
            "target type mismatch",
        ),
    ],
)
def test_rejects_malformed_references(
    tmp_path: Path, filename: str, replacement: str, message: str
) -> None:
    directory = _copy_dataset(tmp_path)
    (directory / filename).write_text(replacement)
    _rehash(directory, filename)

    with pytest.raises(DatasetValidationError, match=message):
        load_dataset(directory)


def test_rejects_duplicate_and_unjudged_queries(tmp_path: Path) -> None:
    directory = _copy_dataset(tmp_path)
    query = (directory / "queries.jsonl").read_text().splitlines()[0]
    (directory / "queries.jsonl").write_text(query + "\n" + query + "\n")
    _rehash(directory, "queries.jsonl")
    with pytest.raises(DatasetValidationError, match="duplicate query IDs"):
        load_dataset(directory)

    shutil.copy(SYNTHETIC / "queries.jsonl", directory / "queries.jsonl")
    with (directory / "queries.jsonl").open("a") as output:
        output.write('{"id":"unjudged","text":"No judgment exists"}\n')
    _rehash(directory, "queries.jsonl")
    with pytest.raises(DatasetValidationError, match="queries without judgments"):
        load_dataset(directory)


def test_rejects_query_without_positive_judgment(tmp_path: Path) -> None:
    directory = _copy_dataset(tmp_path)
    (directory / "queries.jsonl").write_text('{"id":"only-zero","text":"Question"}\n')
    (directory / "judgments.jsonl").write_text(
        '{"query_id":"only-zero","target_type":"document","target_id":"benefits-guide","relevance_grade":0}\n'
    )
    _rehash(directory, "queries.jsonl")
    _rehash(directory, "judgments.jsonl")

    with pytest.raises(DatasetValidationError, match="without a positive judgment"):
        load_dataset(directory)


def test_rejects_invalid_manifest_and_missing_file(tmp_path: Path) -> None:
    directory = _copy_dataset(tmp_path)
    (directory / "manifest.json").write_text("{}")
    with pytest.raises(DatasetValidationError, match="invalid manifest"):
        load_dataset(directory)

    directory = _copy_dataset(tmp_path / "second")
    (directory / "corpus.jsonl").unlink()
    with pytest.raises(DatasetValidationError, match="cannot read corpus.jsonl"):
        load_dataset(directory)


def test_contract_rejects_wrong_manifest_files_and_parent_shapes() -> None:
    manifest = json.loads((SYNTHETIC / "manifest.json").read_text())
    manifest["files"] = manifest["files"][:2]
    with pytest.raises(ValidationError, match="files must contain"):
        DatasetManifest.model_validate(manifest)

    with pytest.raises(ValidationError, match="chunk records require document_id"):
        CorpusRecord(id="orphan", type="chunk", text="text")
    with pytest.raises(ValidationError, match="document records cannot have document_id"):
        CorpusRecord(id="document", type="document", document_id="parent", text="text")
