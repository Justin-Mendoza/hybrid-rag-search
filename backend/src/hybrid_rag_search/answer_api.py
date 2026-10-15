"""Read-only author-private history; streaming answer creation belongs to Day 19."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from hybrid_rag_search.answer_history import AnswerDetail, AnswerHistory, AnswerSummary
from hybrid_rag_search.api import DemoUser, Policy
from hybrid_rag_search.management_schemas import ErrorView

router = APIRouter(
    prefix="/v1/workspaces/{workspace_id}/answers",
    tags=["answer history"],
    responses={code: {"model": ErrorView} for code in (401, 403, 404, 422, 503)},
)


def history(policy: Policy) -> AnswerHistory:
    return AnswerHistory(policy)


History = Annotated[AnswerHistory, Depends(history)]


@router.get("")
async def answers(
    workspace_id: UUID,
    service: History,
    demo_user: DemoUser = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[AnswerSummary]:
    return await service.list(demo_user, workspace_id, limit=limit, offset=offset)


@router.get("/{answer_id}")
async def answer(
    workspace_id: UUID,
    answer_id: UUID,
    service: History,
    demo_user: DemoUser = None,
) -> AnswerDetail:
    return await service.get(demo_user, workspace_id, answer_id)
