"""
Pytest configuration and shared test fixtures.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DvcTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MlflowTools
from api.main import app


@pytest.fixture
def test_client() -> TestClient:
    """TestClient instance for testing FastAPI routes."""
    return TestClient(app)


@pytest.fixture
def mock_dvc_tools() -> DvcTools:
    return DvcTools(repo_path=".")


@pytest.fixture
def mock_mlflow_tools() -> MlflowTools:
    return MlflowTools(tracking_uri="sqlite:///:memory:")


@pytest.fixture
def mock_airflow_tools() -> AirflowTools:
    return AirflowTools()


@pytest.fixture
def mock_k8s_tools() -> KubernetesTools:
    return KubernetesTools()


@pytest.fixture
def orchestrator(
    mock_dvc_tools: DvcTools,
    mock_mlflow_tools: MlflowTools,
    mock_airflow_tools: AirflowTools,
    mock_k8s_tools: KubernetesTools,
) -> MLOpsAgentOrchestrator:
    return MLOpsAgentOrchestrator(
        dvc_tools=mock_dvc_tools,
        mlflow_tools=mock_mlflow_tools,
        airflow_tools=mock_airflow_tools,
        k8s_tools=mock_k8s_tools,
    )
