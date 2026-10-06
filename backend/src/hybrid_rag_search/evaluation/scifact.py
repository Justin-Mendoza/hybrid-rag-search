"""Download and convert the pinned BEIR SciFact test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx

from hybrid_rag_search.evaluation.datasets import StrictModel, load_dataset

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_ARCHIVE_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
SCIFACT_VERSION = "beir-2021-03-19-test-v1"
DEFAULT_OUTPUT = Path("datasets/.cache/scifact") / SCIFACT_VERSION
SUBSET_OUTPUT = Path("datasets/scifact-subset/v1")


def build_subset(source: Path = DEFAULT_OUTPUT, output: Path = SUBSET_OUTPUT) -> Path:
    """Select 30 queries and 500 documents by stable salted SHA-256 ordering."""
    dataset = load_dataset(source)

    def order(identifier: str) -> str:
        return _sha256(f"scifact-subset-v1:{identifier}".encode())

    if len(dataset.queries) < 30 or len(dataset.corpus) < 500:
        raise ValueError("SciFact subset requires at least 30 queries and 500 documents")
    queries = sorted(dataset.queries, key=lambda record: order(record.id))[:30]
    query_ids = {record.id for record in queries}
    judgments = [record for record in dataset.judgments if record.query_id in query_ids]
    selected = {record.target_id for record in judgments}
    if len(selected) > 500:
        raise ValueError("Selected judgments require more than 500 documents")
    for record in sorted(dataset.corpus, key=lambda record: order(record.id)):
        if len(selected) == 500:
            break
        selected.add(record.id)
    corpus = [record for record in dataset.corpus if record.id in selected]
    groups: dict[str, list[StrictModel]] = {
        "corpus.jsonl": list(corpus),
        "queries.jsonl": list(queries),
        "judgments.jsonl": list(judgments),
    }
    rendered = {
        name: b"".join(
            _json_line(record)
            for record in sorted(
                [record.model_dump(mode="json") for record in records],
                key=lambda record: (
                    record.get("id", "") or f"{record['query_id']}:{record['target_id']}"
                ),
            )
        )
        for name, records in groups.items()
    }
    manifest = dataset.manifest.model_dump(mode="json")
    manifest.update(
        dataset_id="scifact-subset",
        name="SciFact pipeline subset",
        version="scifact-subset-v1",
        description=(
            "Pipeline baseline, not full BEIR SciFact. "
            f"Parent dataset hash: {dataset.dataset_hash}. "
            "30 lowest SHA-256(scifact-subset-v1:query ID) queries; all their judgments; "
            "fill to 500 documents by the same salted hash ordering; output sorted by ID."
        ),
        files=[{"path": name, "sha256": _sha256(data)} for name, data in sorted(rendered.items())],
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, data in rendered.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    load_dataset(output)
    return output


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def download_archive(destination: Path) -> Path:
    """Download the official archive and reject content that differs from the pin."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    try:
        response = httpx.get(SCIFACT_URL, follow_redirects=True, timeout=60)
        response.raise_for_status()
        temporary.write_bytes(response.content)
        actual = _sha256(temporary.read_bytes())
        if actual != SCIFACT_ARCHIVE_SHA256:
            raise ValueError(
                f"SciFact archive SHA-256 mismatch: expected {SCIFACT_ARCHIVE_SHA256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def convert_archive(archive: Path, output: Path) -> Path:
    """Convert BEIR SciFact's test split into the shared four-file contract."""
    archive_bytes = archive.read_bytes()
    actual = _sha256(archive_bytes)
    if actual != SCIFACT_ARCHIVE_SHA256:
        raise ValueError(
            f"SciFact archive SHA-256 mismatch: expected {SCIFACT_ARCHIVE_SHA256}, got {actual}"
        )

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
        qrels_text = source.read("scifact/qrels/test.tsv").decode("utf-8")
        qrels = list(csv.DictReader(io.StringIO(qrels_text), delimiter="\t"))
        relevant_query_ids = {row["query-id"] for row in qrels}

        corpus_records: list[dict[str, Any]] = []
        for line in source.read("scifact/corpus.jsonl").decode("utf-8").splitlines():
            source_record = json.loads(line)
            corpus_records.append(
                {
                    "id": f"scifact-doc-{source_record['_id']}",
                    "metadata": source_record.get("metadata", {}),
                    "text": source_record["text"],
                    "title": source_record.get("title") or None,
                    "type": "document",
                }
            )

        query_records: list[dict[str, Any]] = []
        for line in source.read("scifact/queries.jsonl").decode("utf-8").splitlines():
            source_record = json.loads(line)
            if source_record["_id"] in relevant_query_ids:
                query_records.append(
                    {"id": f"scifact-query-{source_record['_id']}", "text": source_record["text"]}
                )

    corpus_records.sort(key=lambda record: record["id"])
    query_records.sort(key=lambda record: record["id"])
    judgment_records = sorted(
        (
            {
                "query_id": f"scifact-query-{row['query-id']}",
                "relevance_grade": 3 if int(row["score"]) > 0 else 0,
                "target_id": f"scifact-doc-{row['corpus-id']}",
                "target_type": "document",
            }
            for row in qrels
        ),
        key=lambda record: (record["query_id"], record["target_id"]),
    )

    output.mkdir(parents=True, exist_ok=True)
    rendered = {
        "corpus.jsonl": b"".join(_json_line(record) for record in corpus_records),
        "queries.jsonl": b"".join(_json_line(record) for record in query_records),
        "judgments.jsonl": b"".join(_json_line(record) for record in judgment_records),
    }
    for name, data in rendered.items():
        (output / name).write_bytes(data)

    manifest = {
        "schema_version": "evaluation-dataset-v1",
        "dataset_id": "scifact",
        "name": "BEIR SciFact test split",
        "version": SCIFACT_VERSION,
        "purpose": "evaluation",
        "description": "Public scientific claim retrieval benchmark converted from BEIR SciFact.",
        "source": {
            "name": "BEIR SciFact",
            "url": SCIFACT_URL,
            "archive_sha256": SCIFACT_ARCHIVE_SHA256,
            "license": "CC BY-NC 2.0",
        },
        "files": [
            {"path": name, "sha256": _sha256(data)} for name, data in sorted(rendered.items())
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _summary(directory: Path) -> str:
    dataset = load_dataset(directory)
    return (
        f"dataset={dataset.manifest.dataset_id} version={dataset.manifest.version} "
        f"corpus={len(dataset.corpus)} queries={len(dataset.queries)} "
        f"judgments={len(dataset.judgments)} corpus_hash={dataset.corpus_hash} "
        f"dataset_hash={dataset.dataset_hash}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "validate", "subset"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output = args.output or (SUBSET_OUTPUT if args.command == "subset" else DEFAULT_OUTPUT)

    if args.command == "download":
        archive = args.output.parent / "scifact.zip"
        download_archive(archive)
        convert_archive(archive, args.output)
    elif args.command == "subset":
        build_subset(args.source, args.output)
    print(_summary(args.output))


if __name__ == "__main__":  # pragma: no cover - exercised through main() tests
    main()
