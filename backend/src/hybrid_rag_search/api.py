"""Local demo API: simulated identity selection, real database authorization."""

from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hybrid_rag_search.authorization import AccessDenied, Authorization
from hybrid_rag_search.config import get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.documents import DocumentNotFound, DocumentService, DocumentUnavailable
from hybrid_rag_search.lexical_retrieval import RetrievalError
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.search import AuthorizedSearch, SearchMode
from hybrid_rag_search.storage import LocalFileStorage

router = APIRouter(prefix="/v1")
DemoUser = Annotated[UUID | None, Header(alias="X-Demo-User-ID")]


async def authorization() -> AsyncIterator[Authorization]:
    engine = create_async_engine(async_database_url(get_settings().database_url))
    try:
        yield Authorization(async_sessionmaker(engine, expire_on_commit=False))
    finally:
        await engine.dispose()


Policy = Annotated[Authorization, Depends(authorization)]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: UUID
    query: str = Field(min_length=1, max_length=4096)
    collection_id: UUID | None = None
    mode: SearchMode = "hybrid_rerank"
    debug: bool = False


@router.get("/demo-users", tags=["demo identity"])
async def demo_users(policy: Policy) -> list[dict[str, object]]:
    return await policy.demo_users()


@router.get("/workspaces", tags=["demo identity"])
async def workspaces(policy: Policy, demo_user: DemoUser = None) -> list[dict[str, object]]:
    try:
        return await policy.workspaces(demo_user)
    except AccessDenied as error:
        raise HTTPException(error.status_code, error.detail) from None


@router.post("/search", tags=["search"])
async def search(
    request: SearchRequest, policy: Policy, demo_user: DemoUser = None
) -> dict[str, object]:
    try:
        scope = await policy.scope(demo_user, request.workspace_id, request.collection_id)
        async with AuthorizedSearch(get_settings()) as retrieval:
            return await retrieval.search(scope, request.query, request.mode, debug=request.debug)
    except AccessDenied as error:
        raise HTTPException(error.status_code, error.detail) from None
    except (RetrievalError, ProviderError) as error:
        raise HTTPException(503, error.code) from None


@router.get("/workspaces/{workspace_id}/documents/{document_id}/content", tags=["documents"])
async def document_content(
    workspace_id: UUID, document_id: UUID, policy: Policy, demo_user: DemoUser = None
) -> Response:
    try:
        await policy.document_scope(demo_user, workspace_id, document_id)
        service = DocumentService(policy.sessions, LocalFileStorage(get_settings().storage_root))
        content = await service.fetch(workspace_id, document_id)
        return Response(
            content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment"},
        )
    except AccessDenied as error:
        raise HTTPException(error.status_code, error.detail) from None
    except (DocumentNotFound, DocumentUnavailable):
        raise HTTPException(404, "Document not found") from None
