from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get("X-Request-ID")


def test_predict_endpoint() -> None:
    request_id = "test-request-id-123"
    response = client.post(
        "/predict",
        json={"features": [1.0, 0.5, -0.2]},
        headers={"X-Request-ID": request_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {0, 1}
    assert 0.0 <= body["score"] <= 1.0
    assert body["model_version"] == "v1"
    assert body["request_id"] == request_id
    assert response.headers["X-Request-ID"] == request_id

