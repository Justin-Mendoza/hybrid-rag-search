from fastapi.testclient import TestClient

from hybrid_rag_search.main import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "development"}
