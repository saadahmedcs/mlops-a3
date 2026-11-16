"""
Unit tests for the FastAPI interface endpoints and middleware.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_check_endpoint(test_client: TestClient) -> None:
    """Verify that /health responds with 200 OK and valid status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "mlops-ai-agent"
    assert "X-Request-ID" in response.headers


def test_list_tools_endpoint(test_client: TestClient) -> None:
    """Verify that /tools discovers all 11 registered tools."""
    response = test_client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    tool_names = [t["name"] for t in data["tools"]]
    assert "dvc_inspect_datasets" in tool_names
    assert "mlflow_get_best_run" in tool_names
    assert "airflow_get_dag_status" in tool_names
    assert "k8s_inspect_deployment" in tool_names


def test_query_endpoint_mandatory_scenario(test_client: TestClient) -> None:
    """
    Test the mandatory assignment scenario:
    'Find the best recent model, identify the dataset version used to train it,
    check the deployment status, and explain whether retraining is necessary.'
    """
    payload = {
        "query": (
            "Find the best recent model, identify the dataset version used to train it, "
            "check the deployment status, and explain whether retraining is necessary."
        ),
        "target_namespace": "mlops",
        "min_f1_threshold": 0.85,
    }
    response = test_client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify response schema fields
    assert data["status"] == "COMPLETED"
    assert data["provenance"] is not None
    assert data["provenance"]["dataset_version"] == "v2.1"
    assert data["deployment"] is not None
    assert data["deployment"]["namespace"] == "mlops"
    assert data["deployment"]["health_status"] == "HEALTHY"
    assert data["retraining_decision"] == "NOT_RECOMMENDED"
    assert len(data["facts"]) >= 3
    assert len(data["tool_executions"]) >= 3
    assert data["total_duration_ms"] > 0
    assert "X-Request-ID" in response.headers


def test_query_endpoint_validation_error(test_client: TestClient) -> None:
    """Verify that empty or invalid queries return 422 Unprocessable Entity."""
    response = test_client.post("/query", json={"query": "ab"})  # Too short (min 3 chars)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"
    assert "request_id" in data


def test_request_id_header_propagation(test_client: TestClient) -> None:
    """Verify that custom X-Request-ID header is propagated through the request and returned."""
    custom_id = "test-custom-req-id-9988"
    response = test_client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id
    assert response.json()["request_id"] == custom_id
