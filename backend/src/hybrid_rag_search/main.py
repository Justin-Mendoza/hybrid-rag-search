from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from hybrid_rag_search.config import get_settings


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Hybrid RAG Search API", version="0.1.0")

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", environment=settings.app_env)

    return app


app = create_app()
