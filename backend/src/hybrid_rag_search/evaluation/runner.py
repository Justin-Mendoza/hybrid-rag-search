"""Run retrieval benchmarks and save inspectable, reproducible file reports."""

import argparse
import asyncio
import json
import os
import platform
import subprocess
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from fastapi.encoders import jsonable_encoder

from hybrid_rag_search.config import Settings, get_settings
from hybrid_rag_search.evaluation.corpus import (
    DATASETS,
    evaluation_index_settings,
    evaluation_scope,
    prepare_corpus,
    selected_dataset,
    verify_prepared,
)
from hybrid_rag_search.evaluation.datasets import EvaluationDataset
from hybrid_rag_search.evaluation.metrics import score_ranking
from hybrid_rag_search.evaluation.search import (
    MODES,
    EvaluationSearch,
    SearchMode,
    benchmark_settings,
)

SearchFunction = Callable[
    [EvaluationDataset, str, SearchMode, dict[str, Any]], Awaitable[dict[str, Any]]
]


async def evaluate(
    dataset: EvaluationDataset, search: SearchFunction, *, query_delay: float = 0
) -> dict[str, Any]:
    if query_delay < 0:
        raise ValueError("Query delay must be nonnegative")
    judgments: dict[str, dict[str, int]] = {query.id: {} for query in dataset.queries}
    for item in dataset.judgments:
        targets = judgments[item.query_id]
        if item.target_type != "document" or item.target_id in targets:
            raise ValueError("Evaluation requires unique document judgments per query")
        targets[item.target_id] = item.relevance_grade
    rows: list[dict[str, Any]] = []
    for query in dataset.queries:
        if query_delay:
            await asyncio.sleep(query_delay)
        modes: dict[str, Any] = {}
        for mode in MODES:
            started = perf_counter()
            result = await search(dataset, query.text, mode, query.metadata_filters)
            # Corpus text is already versioned. Keep every ranked chunk reference
            # without repeating source bytes in hundreds of benchmark queries.
            result["chunks"] = [
                {key: chunk[key] for key in ("chunk_id", "evaluation_document_id", "rank", "score")}
                for chunk in result["chunks"]
            ]
            ranking = [item["document_id"] for item in result["documents"]]
            modes[mode] = {
                "metrics": score_ranking(ranking, judgments[query.id]),
                "elapsed_ms": (perf_counter() - started) * 1000,
                "chunk_count": len(result["chunks"]),
                "document_count": len(ranking),
                "search": result,
            }
        reference = modes["bm25"]["metrics"]
        for mode in MODES:
            modes[mode]["delta_vs_bm25"] = {
                key: value - reference[key] for key, value in modes[mode]["metrics"].items()
            }
        rows.append(
            {
                "query_id": query.id,
                "query": query.text,
                "category": query.category,
                "judgments": judgments[query.id],
                "modes": modes,
            }
        )
        print(f"Evaluated {len(rows)}/{len(dataset.queries)} queries", flush=True)
    aggregate = {
        mode: {
            key: sum(row["modes"][mode]["metrics"][key] for row in rows) / len(rows)
            for key in rows[0]["modes"][mode]["metrics"]
        }
        for mode in MODES
    }
    return {"aggregate": aggregate, "queries": rows}


def markdown_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        f"# Retrieval baseline: {metadata['dataset']['dataset_id']}",
        "",
        "## Reproduction metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Aggregate scores",
        "",
        "| Mode | Recall@5 | Recall@10 | Recall@50 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    metric_names = ("recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10")
    for mode, metrics in report["aggregate"].items():
        lines.append(
            f"| {mode} | " + " | ".join(f"{metrics[key]:.6f}" for key in metric_names) + " |"
        )
    lines.extend(
        [
            "",
            "## Per-query differences against BM25",
            "",
            "| Query | Mode | Δ Recall@5 | Δ Recall@10 | Δ Recall@50 | Δ MRR@10 | Δ nDCG@10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["queries"]:
        for mode in MODES[1:]:
            values = [row["modes"][mode]["delta_vs_bm25"][key] for key in metric_names]
            lines.append(
                f"| {row['query_id']} | {mode} | "
                + " | ".join(f"{value:+.6f}" for value in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "JSON contains winning chunks, complete chunk rankings, timings, and traces.",
            "Unjudged documents earn zero credit; their relevance is unknown.",
            "",
        ]
    )
    return "\n".join(lines)


def save_report(report: dict[str, Any], output: Path) -> None:
    rendered = jsonable_encoder(report)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(markdown_report(rendered))


def compare_reports(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Require comparable inputs; report maximum observed per-query and aggregate drift."""
    for key in ("dataset", "configuration", "pipeline_version", "index", "models", "adapters"):
        if first["metadata"][key] != second["metadata"][key]:
            raise ValueError(f"Reports differ in {key}; cannot establish repeatability")
    if [row["query_id"] for row in first["queries"]] != [
        row["query_id"] for row in second["queries"]
    ]:
        raise ValueError("Reports have different query sets")
    per_query = max(
        abs(a["modes"][mode]["metrics"][key] - b["modes"][mode]["metrics"][key])
        for a, b in zip(first["queries"], second["queries"], strict=True)
        for mode in MODES
        for key in a["modes"][mode]["metrics"]
    )
    aggregate = max(
        abs(first["aggregate"][mode][key] - second["aggregate"][mode][key])
        for mode in MODES
        for key in first["aggregate"][mode]
    )
    return {
        "max_absolute_per_query_metric_delta": per_query,
        "max_absolute_aggregate_metric_delta": aggregate,
        "latency": "descriptive; no equality requirement",
    }


async def run_baseline(name: str, output: Path, cache_state: str, query_delay: float = 0) -> None:
    settings = benchmark_settings(get_settings())
    dataset = selected_dataset(name)
    settings = evaluation_index_settings(dataset, settings)
    chunks = await verify_prepared(dataset, settings)
    scope = evaluation_scope(dataset, settings)
    async with httpx.AsyncClient(base_url=settings.opensearch_url) as client:
        response = await client.get(f"/_alias/{settings.opensearch_read_alias}")
        response.raise_for_status()
        indexes = sorted(response.json())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "diff", "HEAD", "--name-only"], text=True).strip())
    async with EvaluationSearch(settings) as search:
        result = await evaluate(dataset, search.search, query_delay=query_delay)
    result["metadata"] = {
        "report_schema": "retrieval-evaluation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "dataset": {
            "dataset_id": dataset.manifest.dataset_id,
            "version": dataset.manifest.version,
            "dataset_hash": dataset.dataset_hash,
            "corpus_hash": dataset.corpus_hash,
            "document_count": len(dataset.corpus),
            "query_count": len(dataset.queries),
        },
        "chunk_count": chunks,
        "pipeline_version": scope.pipeline_version,
        "index": indexes,
        "tenant_id": str(scope.tenant_id),
        "collection_id": str(scope.collection_id),
        "hardware": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "cache_state": cache_state,
        "query_delay_seconds": query_delay,
        "execution": (
            "sequential queries and modes; hybrid branches concurrent; no query embedding cache"
        ),
        "models": {
            "embedding": settings.cohere_embed_model,
            "rerank": settings.cohere_rerank_model,
        },
        "adapters": {"embedding": "cohere", "rerank": "cohere", "retrieval": "opensearch"},
        "configuration": configuration(settings),
        "metric_rules": (
            "document first occurrence; positive >0; nDCG gain 2^grade-1; macro mean; unjudged=0"
        ),
        "rerank_fallback_count": sum(
            row["modes"]["hybrid_rerank"]["search"]["trace"]["fallback_reason"] is not None
            for row in result["queries"]
        ),
    }
    save_report(result, output)
    print(f"Saved {output}/report.json and report.md", flush=True)


def configuration(settings: Settings) -> dict[str, Any]:
    """Explicit allowlist prevents credentials and unrelated settings entering reports."""
    names = [name for name in type(settings).model_fields if name.startswith("opensearch_bm25_")]
    names += [
        "opensearch_dense_candidate_limit",
        "cohere_embed_dimensions",
        "rerank_candidate_limit",
        "rerank_result_limit",
        "retrieval_deadline_seconds",
        "cohere_rerank_timeout_seconds",
        "cohere_timeout_seconds",
        "opensearch_index_schema_version",
    ]
    return {**{name: getattr(settings, name) for name in names}, "rrf_rank_constant": 60}


def main() -> None:  # pragma: no cover - exercised through documented CLI commands
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("dataset", choices=list(DATASETS))
    prepare.add_argument("--document-delay", type=float, default=0)
    run = sub.add_parser("run")
    run.add_argument("dataset", choices=list(DATASETS))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--query-delay", type=float, default=0)
    run.add_argument(
        "--cache-state", required=True, help="Describe observed cache conditions honestly"
    )
    compare = sub.add_parser("compare")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    export = sub.add_parser("export")
    export.add_argument("report", type=Path)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        asyncio.run(
            prepare_corpus(
                selected_dataset(args.dataset), get_settings(), document_delay=args.document_delay
            )
        )
    elif args.command == "run":
        asyncio.run(run_baseline(args.dataset, args.output, args.cache_state, args.query_delay))
    elif args.command == "export":
        save_report(json.loads(args.report.read_text()), args.output)
    else:
        print(
            json.dumps(
                compare_reports(
                    json.loads(args.first.read_text()), json.loads(args.second.read_text())
                ),
                indent=2,
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
