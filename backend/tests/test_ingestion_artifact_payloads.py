import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from hybrid_rag_search.chunking.contracts import Chunk, SourceSpan
from hybrid_rag_search.ingestion.artifact_payloads import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactContractError,
    ChunkEmbedding,
    EmbeddingsArtifact,
    build_embeddings_artifact,
    decode_chunks_artifact,
    decode_embeddings_artifact,
    decode_parsed_artifact,
    encode_chunks_artifact,
    encode_embeddings_artifact,
    encode_parsed_artifact,
)
from hybrid_rag_search.ingestion.artifacts import (
    ArtifactKind,
    LocalArtifactStorage,
    artifact_identity,
    artifact_key,
)
from hybrid_rag_search.parsers.contracts import ParsedBlock, ParsedDocument, SourceLocator
from hybrid_rag_search.providers.embeddings import (
    EmbeddingPurpose,
    EmbeddingResult,
)

IDENTITY = "a" * 64
DOCUMENT_ID = "doc_" + "b" * 64
CONFIG_VERSION = "chunkcfg_" + "c" * 64


def parsed_document() -> ParsedDocument:
    return ParsedDocument(
        blocks=(
            ParsedBlock("Policy", SourceLocator(1, page_number=2, heading_path=("Policy",))),
            ParsedBlock(
                "Work remotely.",
                SourceLocator(2, page_number=2, heading_path=("Policy",)),
            ),
        ),
        media_type="application/pdf",
        parser_name="pypdf",
        parser_version="1",
    )


def chunks() -> tuple[Chunk, ...]:
    locator = SourceLocator(2, page_number=2, heading_path=("Policy",))
    return (
        Chunk(
            document_id=DOCUMENT_ID,
            config_version=CONFIG_VERSION,
            order=1,
            content_text="Work remotely.",
            embedding_text="# Policy\n\nWork remotely.",
            embedding_token_count=5,
            source_spans=(SourceSpan(locator, 0, 14),),
        ),
        Chunk(
            document_id=DOCUMENT_ID,
            config_version=CONFIG_VERSION,
            order=2,
            content_text="remotely.",
            embedding_text="# Policy\n\nremotely.",
            embedding_token_count=4,
            source_spans=(SourceSpan(locator, 5, 14),),
        ),
    )


def result(**changes: Any) -> EmbeddingResult:
    values: dict[str, Any] = {
        "vectors": ((1.0, 0.0), (0.0, 1.0)),
        "model": "embed-test",
        "dimensions": 2,
        "purpose": EmbeddingPurpose.DOCUMENT,
        "adapter": "fake",
    }
    values.update(changes)
    return EmbeddingResult(**values)


def decoded_json(content: bytes) -> dict[str, Any]:
    value = json.loads(content)
    assert isinstance(value, dict)
    return value


def changed_json(content: bytes, change) -> bytes:
    value = decoded_json(content)
    change(value)
    return json.dumps(value, separators=(",", ":")).encode()


def test_parsed_artifact_round_trip_is_versioned_and_deterministic() -> None:
    document = parsed_document()
    encoded = encode_parsed_artifact(document, IDENTITY)
    assert encoded == encode_parsed_artifact(document, IDENTITY)
    assert decode_parsed_artifact(encoded, IDENTITY) == document
    assert decoded_json(encoded) == {
        "identity": IDENTITY,
        "kind": "parsed",
        "payload": {
            "blocks": [
                {
                    "locator": {
                        "block_number": 1,
                        "heading_path": ["Policy"],
                        "page_number": 2,
                    },
                    "text": "Policy",
                },
                {
                    "locator": {
                        "block_number": 2,
                        "heading_path": ["Policy"],
                        "page_number": 2,
                    },
                    "text": "Work remotely.",
                },
            ],
            "media_type": "application/pdf",
            "parser_name": "pypdf",
            "parser_version": "1",
        },
        "schema_version": ARTIFACT_SCHEMA_VERSION,
    }


def test_uncheckpointed_artifact_can_be_looked_up_and_validated(tmp_path: Path) -> None:
    tenant_id, job_id = uuid4(), uuid4()
    identity = artifact_identity(ArtifactKind.PARSED, "original-content", "parser-v1")
    key = artifact_key(tenant_id, job_id, ArtifactKind.PARSED, identity)
    storage = LocalArtifactStorage(tmp_path)
    storage.save(key, encode_parsed_artifact(parsed_document(), identity))

    recovered = storage.lookup(key)
    assert recovered is not None
    assert decode_parsed_artifact(storage.fetch(recovered), identity) == parsed_document()


def test_chunks_artifact_round_trip_recomputes_chunk_ids() -> None:
    expected = chunks()
    encoded = encode_chunks_artifact(expected, IDENTITY)
    assert decode_chunks_artifact(encoded, IDENTITY) == expected
    assert decoded_json(encoded)["payload"]["chunks"][0]["chunk_id"] == expected[0].chunk_id


def test_embeddings_artifact_pairs_vectors_with_chunk_ids() -> None:
    source_chunks = chunks()
    artifact = build_embeddings_artifact(source_chunks, result())
    encoded = encode_embeddings_artifact(artifact, IDENTITY)
    assert decode_embeddings_artifact(encoded, IDENTITY) == artifact
    assert tuple(item.chunk_id for item in artifact.embeddings) == tuple(
        chunk.chunk_id for chunk in source_chunks
    )
    assert decoded_json(encoded)["payload"]["purpose"] == "document"


@pytest.mark.parametrize(
    "decoder",
    [decode_parsed_artifact, decode_chunks_artifact, decode_embeddings_artifact],
)
def test_decoders_reject_wrong_checkpoint_identity(decoder) -> None:
    contents = {
        decode_parsed_artifact: encode_parsed_artifact(parsed_document(), IDENTITY),
        decode_chunks_artifact: encode_chunks_artifact(chunks(), IDENTITY),
        decode_embeddings_artifact: encode_embeddings_artifact(
            build_embeddings_artifact(chunks(), result()), IDENTITY
        ),
    }
    with pytest.raises(ArtifactContractError, match="identity"):
        decoder(contents[decoder], "d" * 64)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(kind="chunks"),
        lambda value: value.update(extra=True),
        lambda value: value["payload"].update(extra=True),
        lambda value: value.update(identity="invalid"),
    ],
)
def test_parsed_decoder_rejects_incompatible_envelopes(mutation) -> None:
    content = changed_json(encode_parsed_artifact(parsed_document(), IDENTITY), mutation)
    with pytest.raises(ArtifactContractError, match="Invalid parsed"):
        decode_parsed_artifact(content, IDENTITY)


@pytest.mark.parametrize("content", [b"not-json", b"\xff", "json"])
def test_decoder_rejects_non_json_bytes(content: Any) -> None:
    with pytest.raises(ArtifactContractError, match="Invalid parsed"):
        decode_parsed_artifact(content, IDENTITY)


def test_parsed_decoder_applies_domain_invariants() -> None:
    content = changed_json(
        encode_parsed_artifact(parsed_document(), IDENTITY),
        lambda value: value["payload"]["blocks"][1]["locator"].update(block_number=3),
    )
    with pytest.raises(ArtifactContractError, match="Invalid parsed"):
        decode_parsed_artifact(content, IDENTITY)


@pytest.mark.parametrize("value", ["bad", "A" * 64, 42])
def test_encoders_require_valid_identity(value: Any) -> None:
    with pytest.raises(ValueError, match="identity"):
        encode_parsed_artifact(parsed_document(), value)


def test_parsed_encoder_requires_domain_value() -> None:
    with pytest.raises(ValueError, match="ParsedDocument"):
        encode_parsed_artifact("document", IDENTITY)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [(), [], ("chunk",)],
)
def test_chunks_encoder_requires_nonempty_chunk_tuple(value: Any) -> None:
    with pytest.raises(ValueError, match="Chunks artifact"):
        encode_chunks_artifact(value, IDENTITY)


def test_chunks_encoder_requires_one_document_and_configuration() -> None:
    first, second = chunks()
    with pytest.raises(ValueError, match="one document"):
        encode_chunks_artifact((first, replace(second, document_id="doc_" + "d" * 64)), IDENTITY)
    with pytest.raises(ValueError, match="one document"):
        encode_chunks_artifact(
            (first, replace(second, config_version="chunkcfg_" + "d" * 64)), IDENTITY
        )


def test_chunks_encoder_requires_consecutive_order() -> None:
    first, second = chunks()
    with pytest.raises(ValueError, match="consecutive"):
        encode_chunks_artifact((first, replace(second, order=3)), IDENTITY)


def test_chunks_decoder_rejects_changed_chunk_identity() -> None:
    content = changed_json(
        encode_chunks_artifact(chunks(), IDENTITY),
        lambda value: value["payload"]["chunks"][0].update(chunk_id="chk_" + "d" * 64),
    )
    with pytest.raises(ArtifactContractError, match="Invalid chunks"):
        decode_chunks_artifact(content, IDENTITY)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["payload"]["chunks"][1].update(order=3),
        lambda value: value["payload"]["chunks"][0]["source_spans"][0].update(end_char=0),
        lambda value: value["payload"]["chunks"][0].update(extra=True),
    ],
)
def test_chunks_decoder_rejects_invalid_payload(mutation) -> None:
    content = changed_json(encode_chunks_artifact(chunks(), IDENTITY), mutation)
    with pytest.raises(ArtifactContractError, match="Invalid chunks"):
        decode_chunks_artifact(content, IDENTITY)


def test_embedding_builder_requires_document_result_and_matching_count() -> None:
    with pytest.raises(ValueError, match="EmbeddingResult"):
        build_embeddings_artifact(chunks(), "result")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="document-purpose"):
        build_embeddings_artifact(chunks(), result(purpose=EmbeddingPurpose.QUERY))
    with pytest.raises(ValueError, match="count"):
        build_embeddings_artifact(chunks(), result(vectors=((1.0, 0.0),)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunk_id", "bad", "chunk ID"),
        ("vector", (), "nonempty tuple"),
        ("vector", [], "nonempty tuple"),
        ("vector", (math.inf,), "finite numbers"),
        ("vector", (True,), "finite numbers"),
        ("vector", ("one",), "finite numbers"),
    ],
)
def test_chunk_embedding_validation(field: str, value: Any, message: str) -> None:
    values: dict[str, Any] = {"chunk_id": chunks()[0].chunk_id, "vector": (1.0, 0.0)}
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ChunkEmbedding(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("document_id", "bad", "document ID"),
        ("model", " ", "model"),
        ("adapter", "", "adapter"),
        ("dimensions", 0, "dimensions"),
        ("dimensions", True, "dimensions"),
        ("embeddings", (), "nonempty tuple"),
        ("embeddings", [], "nonempty tuple"),
        ("embeddings", ("item",), "ChunkEmbedding"),
    ],
)
def test_embeddings_artifact_validation(field: str, value: Any, message: str) -> None:
    item = ChunkEmbedding(chunks()[0].chunk_id, (1.0, 0.0))
    values: dict[str, Any] = {
        "document_id": DOCUMENT_ID,
        "model": "model",
        "dimensions": 2,
        "adapter": "fake",
        "embeddings": (item,),
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        EmbeddingsArtifact(**values)


def test_embeddings_artifact_rejects_duplicate_ids_and_dimension_mismatch() -> None:
    item = ChunkEmbedding(chunks()[0].chunk_id, (1.0, 0.0))
    with pytest.raises(ValueError, match="unique"):
        EmbeddingsArtifact(DOCUMENT_ID, "model", 2, "fake", (item, item))
    with pytest.raises(ValueError, match="dimensions"):
        EmbeddingsArtifact(DOCUMENT_ID, "model", 3, "fake", (item,))


def test_embeddings_encoder_requires_domain_value() -> None:
    with pytest.raises(ValueError, match="EmbeddingsArtifact"):
        encode_embeddings_artifact("artifact", IDENTITY)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["payload"].update(purpose="query"),
        lambda value: value["payload"]["embeddings"][0].update(vector=[math.inf, 0.0]),
        lambda value: value["payload"].update(dimensions=3),
        lambda value: value["payload"]["embeddings"][1].update(
            chunk_id=value["payload"]["embeddings"][0]["chunk_id"]
        ),
    ],
)
def test_embeddings_decoder_rejects_invalid_payload(mutation) -> None:
    artifact = build_embeddings_artifact(chunks(), result())
    content = changed_json(encode_embeddings_artifact(artifact, IDENTITY), mutation)
    with pytest.raises(ArtifactContractError, match="Invalid embeddings"):
        decode_embeddings_artifact(content, IDENTITY)
