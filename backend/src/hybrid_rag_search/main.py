from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.health import dependency_status


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dependencies: dict[str, bool]


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Hybrid RAG Search API", version="0.1.0")

    async def live() -> HealthResponse:
        return HealthResponse(status="ok", environment=settings.app_env)

    app.add_api_route("/health", live, methods=["GET"], response_model=HealthResponse)
    app.add_api_route(
        "/health/live", live, methods=["GET"], response_model=HealthResponse, tags=["operations"]
    )

    @app.get("/health/ready", response_model=ReadinessResponse, tags=["operations"])
    async def ready(response: Response) -> ReadinessResponse:
        dependencies = await dependency_status(settings)
        is_ready = all(dependencies.values())
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="ready" if is_ready else "not_ready", dependencies=dependencies
        )

    return app


app = create_app()
