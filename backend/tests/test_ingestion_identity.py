from dataclasses import replace
from typing import Any

import pytest

from hybrid_rag_search.chunking.contracts import DEFAULT_CHUNKING_CONFIG
from hybrid_rag_search.ingestion.identity import PipelineConfig


def pipeline(**changes: Any) -> PipelineConfig:
    values: dict[str, Any] = {
        "parser_name": "pypdf",
        "parser_version": "1",
        "chunk_config_version": DEFAULT_CHUNKING_CONFIG.version,
        "embedding_model": "embed-english-light-v3.0",
        "embedding_dimensions": 384,
        "embedding_input_type": "search_document",
        "embedding_type": "float",
        "index_schema_version": "chunks-v1",
    }
    values.update(changes)
    return PipelineConfig(**values)


def test_pipeline_version_is_stable() -> None:
    first = pipeline()
    assert first.version == pipeline().version
    assert first.version.startswith("pipe_")
    assert len(first.version) == 69


@pytest.mark.parametrize(
    "change",
    [
        {"parser_name": "another-parser"},
        {"parser_version": "2"},
        {"chunk_config_version": "chunkcfg_" + "b" * 64},
        {"embedding_model": "embed-v4.0"},
        {"embedding_dimensions": 1024},
        {"embedding_input_type": "classification"},
        {"embedding_type": "int8"},
        {"index_schema_version": "chunks-v2"},
    ],
)
def test_every_pipeline_input_changes_version(change: dict[str, Any]) -> None:
    assert replace(pipeline(), **change).version != pipeline().version


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parser_name", "", "parser name"),
        ("parser_version", " ", "parser version"),
        ("embedding_model", 42, "embedding model"),
        ("embedding_input_type", "", "embedding input type"),
        ("embedding_type", "", "embedding type"),
        ("index_schema_version", "", "index schema version"),
        ("chunk_config_version", "random", "chunk configuration"),
        ("embedding_dimensions", 0, "embedding dimensions"),
        ("embedding_dimensions", True, "embedding dimensions"),
    ],
)
def test_pipeline_rejects_invalid_inputs(field: str, value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pipeline(**{field: value})
