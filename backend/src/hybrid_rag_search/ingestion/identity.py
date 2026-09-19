"""Stable identity for the complete search-ingestion transformation recipe."""

import json
import re
from dataclasses import dataclass

from hybrid_rag_search.chunking.identity import stable_digest


@dataclass(frozen=True)
class PipelineConfig:
    parser_name: str
    parser_version: str
    chunk_config_version: str
    embedding_model: str
    embedding_dimensions: int
    embedding_input_type: str
    embedding_type: str
    index_schema_version: str

    def __post_init__(self) -> None:
        for label, value in (
            ("parser name", self.parser_name),
            ("parser version", self.parser_version),
            ("embedding model", self.embedding_model),
            ("embedding input type", self.embedding_input_type),
            ("embedding type", self.embedding_type),
            ("index schema version", self.index_schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Pipeline {label} must be a nonblank string")
        if not isinstance(self.chunk_config_version, str) or not re.fullmatch(
            r"chunkcfg_[0-9a-f]{64}", self.chunk_config_version
        ):
            raise ValueError("Pipeline chunk configuration must be a stable configuration identity")
        if (
            not isinstance(self.embedding_dimensions, int)
            or isinstance(self.embedding_dimensions, bool)
            or self.embedding_dimensions <= 0
        ):
            raise ValueError("Pipeline embedding dimensions must be a positive integer")

    @property
    def version(self) -> str:
        canonical = json.dumps(
            {
                "chunk_config_version": self.chunk_config_version,
                "embedding_dimensions": self.embedding_dimensions,
                "embedding_input_type": self.embedding_input_type,
                "embedding_model": self.embedding_model,
                "embedding_type": self.embedding_type,
                "index_schema_version": self.index_schema_version,
                "parser_name": self.parser_name,
                "parser_version": self.parser_version,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"pipe_{stable_digest('ingestion-pipeline', canonical)}"
