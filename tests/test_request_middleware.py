
import uuid

from fastapi.testclient import TestClient

from api.main import app


# ============================================================
# TEST 1: HEALTH ENDPOINT RETURNS HTTP 200
# ============================================================

def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


# ============================================================
# TEST 2: RESPONSE CONTAINS REQUEST ID
# ============================================================

def test_request_id_header_exists():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    request_id = response.headers.get("X-Request-ID")

    assert request_id is not None
    assert request_id != ""

    # Verify that request ID is a valid UUID
    uuid.UUID(request_id)


# ============================================================
# TEST 3: EACH REQUEST HAS A UNIQUE ID
# ============================================================

def test_unique_request_ids():
    client = TestClient(app)

    response_1 = client.get("/health")
    response_2 = client.get("/health")

    request_id_1 = response_1.headers.get("X-Request-ID")
    request_id_2 = response_2.headers.get("X-Request-ID")

    assert request_id_1 is not None
    assert request_id_2 is not None

    assert request_id_1 != request_id_2
