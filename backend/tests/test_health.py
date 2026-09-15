from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from hybrid_rag_search.main import create_app


@pytest.fixture
def client() -> Generator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.mark.parametrize("path", ["/health", "/health/live"])
def test_liveness(path: str, client: TestClient) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "development"}


def test_readiness_when_dependencies_are_ready(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    async def ready_dependencies(*_: object) -> dict[str, bool]:
        return {"postgres": True, "redis": True, "opensearch": True}

    monkeypatch.setattr("hybrid_rag_search.main.dependency_status", ready_dependencies)
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {"postgres": True, "redis": True, "opensearch": True},
    }


def test_readiness_when_a_dependency_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    async def unavailable_dependency(*_: object) -> dict[str, bool]:
        return {"postgres": True, "redis": False, "opensearch": True}

    monkeypatch.setattr("hybrid_rag_search.main.dependency_status", unavailable_dependency)
    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
