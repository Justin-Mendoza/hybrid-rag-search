import hashlib
import sys
from pathlib import Path

import pytest

from hybrid_rag_search.evaluation import scifact
from hybrid_rag_search.evaluation.corpus import selected_dataset
from hybrid_rag_search.evaluation.datasets import load_dataset


def test_subset_is_repeatable_and_preserves_all_selected_judgments(tmp_path: Path) -> None:
    source = Path("datasets/scifact-subset/v1")
    first = scifact.build_subset(source, tmp_path / "first")
    second = scifact.build_subset(source, tmp_path / "second")
    dataset = load_dataset(first)
    assert dataset == load_dataset(second)
    assert len(dataset.corpus) == 500
    assert len(dataset.queries) == 30
    assert dataset.judgments == load_dataset(source).judgments
    assert {item.target_id for item in dataset.judgments} <= {item.id for item in dataset.corpus}
    assert dataset.manifest.source.license == "CC BY-NC 2.0"


def test_subset_rejects_small_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 30"):
        scifact.build_subset(Path("datasets/synthetic-workspace/v1"), tmp_path)


def test_subset_rejects_too_many_required_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = load_dataset(Path("datasets/scifact-subset/v1"))
    judgments = tuple(
        dataset.judgments[0].model_copy(update={"target_id": f"doc-{number}"})
        for number in range(501)
    )
    monkeypatch.setattr(
        scifact, "load_dataset", lambda _: dataset.model_copy(update={"judgments": judgments})
    )
    with pytest.raises(ValueError, match="more than 500"):
        scifact.build_subset(output=tmp_path)


def test_subset_cli_and_default_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scifact, "SUBSET_OUTPUT", tmp_path / "subset")
    monkeypatch.setattr(
        sys, "argv", ["scifact", "subset", "--source", "datasets/scifact-subset/v1"]
    )
    scifact.main()
    assert len(load_dataset(tmp_path / "subset").corpus) == 500


def test_registered_subset_remains_available_to_evaluation_runner() -> None:
    assert len(selected_dataset("scifact-subset").corpus) == 500


def test_selection_ignores_input_order_and_retains_complete_judgments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = load_dataset(Path("datasets/scifact-subset/v1"))
    corpus = tuple(parent.corpus[0].model_copy(update={"id": f"doc-{i}"}) for i in range(600))
    queries = tuple(parent.queries[0].model_copy(update={"id": f"query-{i}"}) for i in range(40))
    judgments = tuple(
        parent.judgments[0].model_copy(update={"query_id": query.id, "target_id": f"doc-{i}"})
        for i, query in enumerate(queries)
    )
    source = parent.model_copy(
        update={"corpus": corpus, "queries": queries, "judgments": judgments}
    )
    original_loader = scifact.load_dataset
    source_path = tmp_path / "source"

    def loader(path: Path):
        return source if path == source_path else original_loader(path)

    monkeypatch.setattr(scifact, "load_dataset", loader)
    first = load_dataset(scifact.build_subset(source_path, tmp_path / "first"))
    source = source.model_copy(
        update={
            "corpus": tuple(reversed(corpus)),
            "queries": tuple(reversed(queries)),
            "judgments": tuple(reversed(judgments)),
        }
    )
    second = load_dataset(scifact.build_subset(source_path, tmp_path / "second"))
    assert first == second
    expected_queries = sorted(
        queries,
        key=lambda query: hashlib.sha256(f"scifact-subset-v1:{query.id}".encode()).hexdigest(),
    )[:30]
    assert {query.id for query in first.queries} == {query.id for query in expected_queries}
    assert set(first.judgments) == {
        item for item in judgments if item.query_id in {query.id for query in expected_queries}
    }
