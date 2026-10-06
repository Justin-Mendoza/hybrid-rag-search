import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hybrid_rag_search.config import Settings
from hybrid_rag_search.evaluation import runner
from hybrid_rag_search.evaluation.corpus import evaluation_scope, selected_dataset
from hybrid_rag_search.evaluation.datasets import EvaluationDataset
from hybrid_rag_search.evaluation.runner import (
    compare_reports,
    configuration,
    evaluate,
    save_report,
)
from hybrid_rag_search.evaluation.search import MODES, SearchMode, benchmark_settings


def test_deterministic_regression_report(tmp_path: Path) -> None:
    import asyncio

    dataset = selected_dataset("synthetic-workspace")

    async def search(
        dataset: EvaluationDataset, query: str, mode: SearchMode, filters: dict[str, Any]
    ) -> dict[str, Any]:
        key = next(item.id for item in dataset.queries if item.text == query)
        positives = [
            item.target_id
            for item in dataset.judgments
            if item.query_id == key and item.relevance_grade > 0
        ]
        ranking = (["unjudged"] if mode == "bm25" else []) + positives
        return {
            "documents": [{"document_id": value} for value in ranking],
            "chunks": [],
            "trace": {"adapter": "fake"},
        }

    first = asyncio.run(evaluate(dataset, search))
    second = asyncio.run(evaluate(dataset, search))
    for report in (first, second):
        report["metadata"] = {
            "dataset": {"dataset_id": "fixture"},
            "configuration": {},
            "pipeline_version": "fixture",
            "index": "fixture",
            "models": {},
            "adapters": {"embedding": "fake", "rerank": "fake"},
        }
    assert first["aggregate"]["bm25"]["mrr@10"] == 0.5
    assert first["aggregate"]["hybrid_rerank"]["mrr@10"] == 1
    assert all(first["aggregate"][mode]["recall@50"] == 1 for mode in MODES)
    assert compare_reports(first, second)["max_absolute_per_query_metric_delta"] == 0
    save_report(first, tmp_path)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["queries"][0]["modes"]["dense"]["delta_vs_bm25"]["mrr@10"] == 0.5
    assert "Per-query differences" in (tmp_path / "report.md").read_text()
    changed = copy.deepcopy(second)
    changed["metadata"]["models"] = {"embedding": "other"}
    with pytest.raises(ValueError, match="models"):
        compare_reports(first, changed)
    changed = copy.deepcopy(second)
    changed["queries"][0]["query_id"] = "other"
    with pytest.raises(ValueError, match="query sets"):
        compare_reports(first, changed)
    changed_dataset = dataset.model_copy(
        update={"judgments": (*dataset.judgments, dataset.judgments[0])}
    )
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(evaluate(changed_dataset, search))


def test_scope_is_stable_isolated_and_independent_of_judgments() -> None:
    dataset = selected_dataset("synthetic-workspace")
    settings = Settings(_env_file=None)
    scope = evaluation_scope(dataset, settings)
    assert scope == evaluation_scope(
        dataset.model_copy(update={"dataset_hash": "a" * 64}), settings
    )
    assert scope != evaluation_scope(dataset.model_copy(update={"corpus_hash": "b" * 64}), settings)
    assert scope != evaluation_scope(
        dataset, settings.model_copy(update={"opensearch_index_schema_version": "chunks-v4"})
    )
    assert len(scope.document_ids) == 8
    with pytest.raises(ValueError, match="Unknown"):
        selected_dataset("../notes.txt")
    chunk = dataset.corpus[0].model_copy(update={"type": "chunk"})
    with pytest.raises(ValueError, match="document corpora"):
        evaluation_scope(dataset.model_copy(update={"corpus": (chunk,)}), settings)
    configured = benchmark_settings(settings)
    assert configured.rerank_candidate_limit == configured.rerank_result_limit == 100
    assert "cohere_api_key" not in configuration(configured)
    assert settings.rerank_candidate_limit == 20


@pytest.mark.anyio
async def test_baseline_records_reproduction_metadata_without_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    settings = Settings(_env_file=None, cohere_api_key="never-write-this")
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(runner.platform, "platform", lambda: "test-system")
    monkeypatch.setattr(runner.platform, "processor", lambda: "test-processor")
    monkeypatch.setattr(runner, "verify_prepared", AsyncMock(return_value=8))
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        runner.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            **kwargs,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"evaluation-index": {}})
            ),
        ),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda args, **kwargs: "revision\n" if "rev-parse" in args else "changed.py\n",
    )
    search = MagicMock()
    search.__aenter__.return_value = search
    monkeypatch.setattr(runner, "EvaluationSearch", lambda _: search)
    metric_names = ("recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10")
    result = {
        "aggregate": {mode: dict.fromkeys(metric_names, 1) for mode in MODES},
        "queries": [
            {
                "query_id": "q",
                "modes": {
                    mode: {
                        "delta_vs_bm25": dict.fromkeys(metric_names, 0),
                        "search": {"trace": {"fallback_reason": "rate_limited"}},
                    }
                    for mode in MODES
                },
            }
        ],
    }
    monkeypatch.setattr(runner, "evaluate", AsyncMock(return_value=result))
    await runner.run_baseline("synthetic-workspace", tmp_path, "warm; provider unknown", 7)
    saved = (tmp_path / "report.json").read_text()
    assert "never-write-this" not in saved
    metadata = json.loads(saved)["metadata"]
    assert metadata["rerank_fallback_count"] == 1
    assert metadata["code_revision"] == "revision"
    assert metadata["tracked_worktree_dirty"] is True
    assert metadata["cache_state"] == "warm; provider unknown"
    assert metadata["adapters"]["embedding"] == "cohere"


@pytest.mark.anyio
async def test_evaluation_pacing_is_explicit_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = selected_dataset("synthetic-workspace")
    dataset = dataset.model_copy(
        update={"queries": dataset.queries[:1], "judgments": dataset.judgments[:3]}
    )

    async def search(*args, **kwargs):
        return {"documents": [], "chunks": []}

    sleep = AsyncMock()
    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    await evaluate(dataset, search, query_delay=7)
    sleep.assert_awaited_once_with(7)
    with pytest.raises(ValueError, match="nonnegative"):
        await evaluate(dataset, search, query_delay=-1)
