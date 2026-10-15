"""Save server-produced answers and recheck ownership/source access on every read."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select

from hybrid_rag_search.answers import AnswerResult, AnswerService
from hybrid_rag_search.authorization import AccessDenied, Authorization
from hybrid_rag_search.models import SavedAnswer


class AnswerSummary(BaseModel):
    id: UUID
    question: str
    created_at: datetime
    available: bool


class AnswerDetail(AnswerSummary):
    result: AnswerResult | None


class AnswerHistory:
    def __init__(self, policy: Authorization) -> None:
        self.policy = policy

    async def answer_and_save(
        self,
        service: AnswerService,
        user_id: UUID | None,
        workspace_id: UUID,
        question: str,
        collection_id: UUID | None = None,
    ) -> UUID:
        scope = await self.policy.scope(user_id, workspace_id, collection_id)
        result = await service.answer(scope, question)
        # JSON serialization normalizes UUIDs, Decimals, source spans, and timestamps.
        row = SavedAnswer(
            tenant_id=scope.tenant_id,
            author_user_id=scope.user_id,
            question=question,
            result=result.model_dump(mode="json"),
        )
        async with self.policy.sessions.begin() as session:
            session.add(row)
            await session.flush()
            answer_id = row.id
        return answer_id

    async def _available(self, user_id: UUID, workspace_id: UUID, result: AnswerResult) -> bool:
        try:
            for document_id in {item.document_id for item in result.evidence}:
                await self.policy.document_scope(user_id, workspace_id, document_id)
        except AccessDenied:
            return False
        return True

    async def list(
        self,
        user_id: UUID | None,
        workspace_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AnswerSummary]:
        scope = await self.policy.scope(user_id, workspace_id)
        async with self.policy.sessions() as session:
            rows = await session.scalars(
                select(SavedAnswer)
                .where(
                    SavedAnswer.tenant_id == workspace_id,
                    SavedAnswer.author_user_id == scope.user_id,
                )
                .order_by(SavedAnswer.created_at.desc(), SavedAnswer.id.desc())
                .limit(limit)
                .offset(offset)
            )
            return [
                AnswerSummary(
                    id=row.id,
                    question=row.question,
                    created_at=row.created_at,
                    available=await self._available(
                        scope.user_id, workspace_id, AnswerResult.model_validate(row.result)
                    ),
                )
                for row in rows
            ]

    async def get(self, user_id: UUID | None, workspace_id: UUID, answer_id: UUID) -> AnswerDetail:
        scope = await self.policy.scope(user_id, workspace_id)
        async with self.policy.sessions() as session:
            row = await session.scalar(
                select(SavedAnswer).where(
                    SavedAnswer.id == answer_id,
                    SavedAnswer.tenant_id == workspace_id,
                    SavedAnswer.author_user_id == scope.user_id,
                )
            )
            if row is None:
                raise AccessDenied(404, "Answer not found")
            result = AnswerResult.model_validate(row.result)
            available = await self._available(scope.user_id, workspace_id, result)
            return AnswerDetail(
                id=row.id,
                question=row.question,
                created_at=row.created_at,
                available=available,
                result=result if available else None,
            )
