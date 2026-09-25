"""Strict, deterministic loading for evaluation-only datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")]


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset violates its contract."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HasId(Protocol):
    id: str


class DatasetFile(StrictModel):
    path: Literal["corpus.jsonl", "queries.jsonl", "judgments.jsonl"]
    sha256: Sha256


class DatasetSource(StrictModel):
    name: str = Field(min_length=1)
    url: str | None = None
    archive_sha256: Sha256 | None = None
    license: str | None = None


class DatasetManifest(StrictModel):
    schema_version: Literal["evaluation-dataset-v1"]
    dataset_id: Identifier
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    purpose: Literal["evaluation"]
    description: str = Field(min_length=1)
    source: DatasetSource
    files: tuple[DatasetFile, ...]

    @model_validator(mode="after")
    def require_all_files_once(self) -> DatasetManifest:
        paths = [item.path for item in self.files]
        expected = {"corpus.jsonl", "queries.jsonl", "judgments.jsonl"}
        if set(paths) != expected or len(paths) != len(expected):
            raise ValueError("files must contain corpus, queries, and judgments exactly once")
        return self


class CorpusRecord(StrictModel):
    id: Identifier
    type: Literal["document", "chunk"]
    document_id: Identifier | None = None
    title: str | None = None
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parent(self) -> CorpusRecord:
        if self.type == "chunk" and self.document_id is None:
            raise ValueError("chunk records require document_id")
        if self.type == "document" and self.document_id is not None:
            raise ValueError("document records cannot have document_id")
        return self


class QueryRecord(StrictModel):
    id: Identifier
    text: str = Field(min_length=1)
    category: Identifier | None = None
    expected_answer: str | None = None
    key_facts: tuple[str, ...] = ()
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class JudgmentRecord(StrictModel):
    query_id: Identifier
    target_type: Literal["document", "chunk"]
    target_id: Identifier
    relevance_grade: int = Field(ge=0, le=3)


class EvaluationDataset(StrictModel):
    manifest: DatasetManifest
    corpus: tuple[CorpusRecord, ...]
    queries: tuple[QueryRecord, ...]
    judgments: tuple[JudgmentRecord, ...]
    corpus_hash: Sha256
    dataset_hash: Sha256


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl[ModelT: StrictModel](path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise DatasetValidationError(f"{path.name}:{line_number}: blank lines are not allowed")
        try:
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise DatasetValidationError(f"{path.name}:{line_number}: expected an object")
            records.append(model.model_validate(value))
        except (json.JSONDecodeError, ValidationError) as error:
            raise DatasetValidationError(f"{path.name}:{line_number}: {error}") from error
    return tuple(records)


def _unique_ids(records: Sequence[HasId], *, label: str) -> None:
    ids = [record.id for record in records]
    duplicates = sorted({identifier for identifier in ids if ids.count(identifier) > 1})
    if duplicates:
        raise DatasetValidationError(f"duplicate {label} IDs: {', '.join(duplicates)}")


def load_dataset(directory: Path) -> EvaluationDataset:
    """Load and validate a four-file dataset directory without mutating it."""
    manifest_path = directory / "manifest.json"
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise DatasetValidationError(f"invalid manifest: {error}") from error

    file_bytes: dict[str, bytes] = {}
    for entry in manifest.files:
        path = directory / entry.path
        try:
            data = path.read_bytes()
        except OSError as error:
            raise DatasetValidationError(f"cannot read {entry.path}: {error}") from error
        actual = _sha256(data)
        if actual != entry.sha256:
            raise DatasetValidationError(
                f"{entry.path} SHA-256 mismatch: expected {entry.sha256}, got {actual}"
            )
        file_bytes[entry.path] = data

    corpus = tuple(_read_jsonl(directory / "corpus.jsonl", CorpusRecord))
    queries = tuple(_read_jsonl(directory / "queries.jsonl", QueryRecord))
    judgments = tuple(_read_jsonl(directory / "judgments.jsonl", JudgmentRecord))
    _unique_ids(corpus, label="corpus")
    _unique_ids(queries, label="query")

    corpus_by_id = {record.id: record for record in corpus}
    query_ids = {query.id for query in queries}
    judged_queries: set[str] = set()
    positive_queries: set[str] = set()
    for judgment in judgments:
        if judgment.query_id not in query_ids:
            raise DatasetValidationError(f"judgment references unknown query {judgment.query_id}")
        target = corpus_by_id.get(judgment.target_id)
        if target is None:
            raise DatasetValidationError(f"judgment references unknown target {judgment.target_id}")
        if target.type != judgment.target_type:
            raise DatasetValidationError(
                f"judgment target type mismatch for {judgment.target_id}: "
                f"expected {target.type}, got {judgment.target_type}"
            )
        judged_queries.add(judgment.query_id)
        if judgment.relevance_grade > 0:
            positive_queries.add(judgment.query_id)

    missing = sorted(query_ids - judged_queries)
    if missing:
        raise DatasetValidationError(f"queries without judgments: {', '.join(missing)}")
    no_positive = sorted(query_ids - positive_queries)
    if no_positive:
        raise DatasetValidationError(
            f"queries without a positive judgment: {', '.join(no_positive)}"
        )

    digest_input = "\n".join(
        [manifest.schema_version, manifest.dataset_id, manifest.version]
        + [f"{name}:{_sha256(data)}" for name, data in sorted(file_bytes.items())]
    ).encode()
    return EvaluationDataset(
        manifest=manifest,
        corpus=corpus,
        queries=queries,
        judgments=judgments,
        corpus_hash=_sha256(file_bytes["corpus.jsonl"]),
        dataset_hash=_sha256(digest_input),
    )
