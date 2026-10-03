"""Local evaluation-only HTTP surface; identity verification belongs to Day 16."""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.evaluation.corpus import selected_dataset, verify_prepared
from hybrid_rag_search.evaluation.search import EvaluationSearch, SearchMode
from hybrid_rag_search.lexical_retrieval import RetrievalError
from hybrid_rag_search.providers.errors import ProviderError

router = APIRouter()


class DebugSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["synthetic-workspace", "scifact"]
    query: str = Field(min_length=1, max_length=4096)
    mode: SearchMode


@router.post("/debug/search", tags=["evaluation"])
async def debug_search(request: DebugSearchRequest) -> dict[str, Any]:
    settings = get_settings()
    if settings.app_env == "production":
        raise HTTPException(404, "Evaluation debug endpoint is local-only")
    try:
        dataset = selected_dataset(request.dataset)
        await verify_prepared(dataset, settings)
        async with EvaluationSearch(settings) as search:
            return await search.search(dataset, request.query, request.mode, debug=True)
    except (ValueError, OSError) as error:
        raise HTTPException(409, str(error)) from error
    except (RetrievalError, ProviderError) as error:
        raise HTTPException(503, error.code) from error
