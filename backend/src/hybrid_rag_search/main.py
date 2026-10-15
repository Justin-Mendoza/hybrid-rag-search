from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from hybrid_rag_search.answer_api import router as answer_router
from hybrid_rag_search.api import router as product_router
from hybrid_rag_search.api_errors import install_errors
from hybrid_rag_search.config import get_settings
from hybrid_rag_search.health import dependency_status
from hybrid_rag_search.management_api import router as management_router


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dependencies: dict[str, bool]


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Hybrid RAG Search API", version="0.1.0")
    install_errors(app)
    app.include_router(product_router)
    app.include_router(answer_router)
    app.include_router(management_router)

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
