"""
Unit tests for agent Pydantic schemas and validation rules.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import (
    AgentExecutionStatus,
    AgentResponse,
    AirflowDagStatusInput,
    AirflowDagStatusOutput,
    DvcInspectInput,
    DvcInspectOutput,
    K8sModifyClusterInput,
    MetricEvidence,
    ProvenanceEvidence,
    RetrainingDecision,
    ToolExecutionStatus,
    UserQueryRequest,
)


def test_user_query_request_validation() -> None:
    """Test valid and invalid natural language request inputs."""
    # Valid payload
    req = UserQueryRequest(
        query="Find the best recent model and check deployment status.",
        target_namespace="mlops",
        min_f1_threshold=0.85,
    )
    assert req.query == "Find the best recent model and check deployment status."
    assert req.target_namespace == "mlops"
    assert req.min_f1_threshold == 0.85
    assert req.user_confirmed is False

    # Short query fails validation
    with pytest.raises(ValidationError):
        UserQueryRequest(query="a")


def test_k8s_modify_cluster_namespace_guardrail() -> None:
    """Verify that unauthorized Kubernetes namespaces are strictly rejected by schema validation."""
    # Allowed namespace 'mlops' passes
    allowed = K8sModifyClusterInput(
        action="restart",
        target_resource="prediction-api",
        namespace="mlops",
        user_confirmed=True,
    )
    assert allowed.namespace == "mlops"

    # Unauthorized namespace fails validation
    with pytest.raises(ValidationError) as exc_info:
        K8sModifyClusterInput(
            action="restart",
            target_resource="prediction-api",
            namespace="production-unauthorized",
            user_confirmed=True,
        )
    assert "is unauthorized" in str(exc_info.value)


def test_dvc_schemas() -> None:
    """Test DVC input and output serialization."""
    dvc_in = DvcInspectInput(repo_path=".")
    assert dvc_in.repo_path == "."

    dvc_out = DvcInspectOutput(
        status=ToolExecutionStatus.SUCCESS,
        current_commit="a4c92e1f40d398b1b51e03197cb244c010d98e52",
        tracked_datasets=[],
        has_uncommitted_changes=False,
        message="DVC inspection completed.",
    )
    assert dvc_out.status == ToolExecutionStatus.SUCCESS
    assert dvc_out.current_commit.startswith("a4c92e1")


def test_airflow_schemas() -> None:
    """Test Airflow DAG status schemas."""
    dag_in = AirflowDagStatusInput(dag_id="training_pipeline")
    assert dag_in.dag_id == "training_pipeline"

    dag_out = AirflowDagStatusOutput(
        status=ToolExecutionStatus.SUCCESS,
        dag_id="training_pipeline",
        state="success",
        failed_tasks=[],
        summary="All tasks passed.",
    )
    assert dag_out.state == "success"
    assert len(dag_out.failed_tasks) == 0


def test_evidence_grounded_agent_response() -> None:
    """Verify that AgentResponse correctly encapsulates all required evidence layers."""
    provenance = ProvenanceEvidence(
        dataset_name="training_data.csv",
        dataset_version="v2.1",
        git_commit="a4c92e1",
        tracked_files=["data/training_data.csv"],
        is_reproducible=True,
    )
    metric = MetricEvidence(
        model_name="customer_churn_v2",
        run_id="run_25",
        experiment_name="churn_exp",
        primary_metric="f1_score",
        primary_value=0.892,
        metrics={"f1_score": 0.892, "accuracy": 0.914},
        parameters={"model": "xgboost"},
    )

    resp = AgentResponse(
        query="Verify model status",
        request_id="req_test_123",
        status=AgentExecutionStatus.COMPLETED,
        summary="Execution successful",
        provenance=provenance,
        metrics=[metric],
        facts=["Model F1 is 0.892"],
        uncertainties=[],
        recommendations=["Continue serving current model"],
        retraining_decision=RetrainingDecision.NOT_RECOMMENDED,
        retraining_reason="Model meets production quality threshold.",
        tool_executions=[],
        total_duration_ms=45.2,
    )

    assert resp.status == AgentExecutionStatus.COMPLETED
    assert resp.provenance is not None
    assert resp.provenance.dataset_version == "v2.1"
    assert resp.metrics[0].primary_value == 0.892
    assert resp.retraining_decision == RetrainingDecision.NOT_RECOMMENDED
