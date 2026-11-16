"""
Pydantic Schemas and Data Models for the MLOps AI Agent.

Defines type-safe models for:
- User queries and conversation context
- Structured tool inputs and outputs (DVC, MLflow, Airflow, Kubernetes)
- Tool selection, execution tracking, and safety policies
- Evidence-based structured response objects (provenance, metrics, deployment, diagnosis)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Enums
# ============================================================================

class ToolExecutionStatus(str, Enum):
    """Execution status of an individual tool call."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


class AgentExecutionStatus(str, Enum):
    """Overall status of agent request resolution."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    PARTIAL = "PARTIAL"


class DiagnosisLayer(str, Enum):
    """Operational layers where faults or degradations may occur."""
    DATA = "DATA"
    MODEL = "MODEL"
    WORKFLOW = "WORKFLOW"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class RetrainingDecision(str, Enum):
    """Defensible retraining recommendation based on concrete evidence."""
    RECOMMENDED = "RECOMMENDED"
    NOT_RECOMMENDED = "NOT_RECOMMENDED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ============================================================================
# User Request and Context Models
# ============================================================================

class UserQueryRequest(BaseModel):
    """Inbound natural language query received via FastAPI."""
    query: str = Field(
        ...,
        description="Natural language query from user/engineer",
        min_length=3,
        examples=["Find the best recent model, identify the dataset version used to train it, check deployment status, and explain whether retraining is necessary."]
    )
    request_id: str | None = Field(
        None,
        description="Unique request tracing identifier. Automatically generated if omitted."
    )
    conversation_id: str | None = Field(
        None,
        description="Identifier correlating multiple conversational turns."
    )
    user_confirmed: bool = Field(
        False,
        description="Explicit user confirmation for cluster or system state mutations."
    )
    target_namespace: str = Field(
        "mlops",
        description="Approved Kubernetes namespace for cluster operations."
    )
    freshness_threshold_days: int = Field(
        30,
        description="Data/model age threshold in days before retraining is considered.",
        ge=1
    )
    min_f1_threshold: float = Field(
        0.80,
        description="Minimum F1 score required for production readiness.",
        ge=0.0,
        le=1.0
    )


# ============================================================================
# DVC Tool Schemas (Task 1: Data Management)
# ============================================================================

class DvcDatasetInfo(BaseModel):
    """Provenance details for a DVC-tracked dataset."""
    path: str = Field(..., description="Path to data file or directory")
    version_tag: str | None = Field(None, description="Semantic or release version tag (e.g. v2.1)")
    git_commit: str = Field(..., description="Associated Git commit SHA")
    remote_url: str | None = Field(None, description="Configured DVC remote storage URL")
    md5_hash: str | None = Field(None, description="DVC content hash")


class DvcInspectInput(BaseModel):
    """Input parameters for inspecting DVC-tracked datasets."""
    repo_path: str = Field(".", description="Path to git/dvc repository")
    target_path: str | None = Field(None, description="Specific dataset file or folder to inspect")


class DvcInspectOutput(BaseModel):
    """Output from DVC dataset inspection."""
    status: ToolExecutionStatus
    current_commit: str = Field(..., description="Current Git HEAD commit SHA")
    tracked_datasets: list[DvcDatasetInfo] = Field(default_factory=list)
    has_uncommitted_changes: bool = Field(False, description="Whether tracked data differs from workspace")
    message: str


class DvcDiffInput(BaseModel):
    """Input for comparing DVC dataset versions across revisions."""
    rev_a: str = Field(..., description="Baseline Git commit or tag")
    rev_b: str = Field("HEAD", description="Comparison Git commit or tag")
    target_path: str | None = Field(None, description="Specific path to compare")


class DvcDiffOutput(BaseModel):
    """Output from comparing two DVC data revisions."""
    status: ToolExecutionStatus
    has_changed: bool
    added_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    summary: str


class DvcReproduceInput(BaseModel):
    """Input for retrieving reproduction instructions or checking out a previous dataset."""
    git_commit: str = Field(..., description="Target Git commit hash to reproduce")
    target_path: str | None = Field(None, description="Optional target file/dir")


class DvcReproduceOutput(BaseModel):
    """Output containing reproducible checkout commands and provenance."""
    status: ToolExecutionStatus
    git_commit: str
    checkout_command: str
    dvc_pull_command: str
    instructions: str


# ============================================================================
# MLflow Tool Schemas (Task 2: Experiment Intelligence)
# ============================================================================

class MlflowRunSummary(BaseModel):
    """Summary of a single MLflow training run."""
    run_id: str
    experiment_id: str
    experiment_name: str
    status: str
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    artifact_uri: str | None = None
    dataset_version: str | None = Field(None, description="DVC dataset version recorded in tags/params")
    git_commit: str | None = Field(None, description="Git commit associated with training run")


class MlflowBestRunInput(BaseModel):
    """Input for finding the top-performing model run."""
    metric_name: str = Field("f1_score", description="Evaluation metric to optimize")
    higher_is_better: bool = Field(True, description="Whether higher values denote better performance")
    max_runs: int = Field(5, description="Number of recent runs to evaluate", ge=1, le=50)
    experiment_name: str | None = Field(None, description="Optional filter by experiment name")


class MlflowBestRunOutput(BaseModel):
    """Output identifying the best run with metrics and provenance references."""
    status: ToolExecutionStatus
    best_run: MlflowRunSummary | None = None
    evaluated_runs_count: int = 0
    metric_name: str
    metric_value: float | None = None
    is_production_ready: bool = Field(False, description="Whether candidate satisfies production quality gate")
    selection_rationale: str


class MlflowCompareInput(BaseModel):
    """Input for comparing multiple MLflow runs."""
    run_ids: list[str] = Field(..., min_length=2, max_length=10)
    metrics_to_compare: list[str] = Field(default_factory=lambda: ["f1_score", "accuracy", "loss"])


class MlflowCompareOutput(BaseModel):
    """Comparative analysis across multiple runs."""
    status: ToolExecutionStatus
    comparison_matrix: list[dict[str, Any]]
    recommendation: str


# ============================================================================
# Airflow Tool Schemas (Task 3: Workflow Operations)
# ============================================================================

class AirflowDagStatusInput(BaseModel):
    """Input for checking the execution status of a training DAG."""
    dag_id: str = Field("training_pipeline", description="Airflow DAG identifier")
    limit: int = Field(1, description="Number of recent DAG runs to inspect", ge=1, le=10)


class AirflowDagStatusOutput(BaseModel):
    """Execution state of the target Airflow DAG."""
    status: ToolExecutionStatus
    dag_id: str
    run_id: str | None = None
    state: str = Field(..., description="DAG run state: success, failed, running, queued")
    start_date: str | None = None
    end_date: str | None = None
    failed_tasks: list[str] = Field(default_factory=list)
    task_states: dict[str, str] = Field(default_factory=dict)
    summary: str


class AirflowTriggerDagInput(BaseModel):
    """Input for triggering an approved training DAG."""
    dag_id: str = Field("training_pipeline", description="Target DAG identifier")
    conf: dict[str, Any] = Field(default_factory=dict, description="Configuration parameters passed to DAG run")


class AirflowTriggerDagOutput(BaseModel):
    """Result of triggering a DAG run."""
    status: ToolExecutionStatus
    dag_id: str
    dag_run_id: str | None = None
    execution_date: str | None = None
    message: str


class AirflowTaskDiagnosticsInput(BaseModel):
    """Input for diagnosing a failed task in a DAG run."""
    dag_id: str = Field("training_pipeline", description="DAG identifier")
    task_id: str = Field(..., description="Failed task name (e.g. apply_model_quality_gate)")
    dag_run_id: str | None = Field(None, description="Specific run ID; latest run used if omitted")


class AirflowTaskDiagnosticsOutput(BaseModel):
    """Root-cause diagnostic for a task failure."""
    status: ToolExecutionStatus
    dag_id: str
    task_id: str
    state: str
    failure_reason: str
    log_snippet: str
    recommended_action: str


# ============================================================================
# Kubernetes Tool Schemas (Task 4: Cluster Operations)
# ============================================================================

class K8sPodInfo(BaseModel):
    """Runtime status of an individual Kubernetes Pod."""
    name: str
    namespace: str
    phase: str = Field(..., description="Pending, Running, Succeeded, Failed, Unknown")
    ready: bool
    restart_count: int
    image: str
    started_at: str | None = None


class K8sInspectDeploymentInput(BaseModel):
    """Input for inspecting application Deployments and Pods."""
    namespace: str = Field("mlops", description="Kubernetes namespace (must match approved namespace)")
    app_label: str = Field("prediction-api", description="Label selector for application components")


class K8sInspectDeploymentOutput(BaseModel):
    """Runtime health, replica counts, and events for deployment."""
    status: ToolExecutionStatus
    namespace: str
    deployment_name: str
    replicas_desired: int
    replicas_ready: int
    active_image_tag: str
    pods: list[K8sPodInfo] = Field(default_factory=list)
    healthy: bool
    restart_events_detected: int = 0
    recent_events: list[str] = Field(default_factory=list)


class K8sPodLogsInput(BaseModel):
    """Input for retrieving logs from an application Pod."""
    pod_name: str
    namespace: str = Field("mlops", description="Approved Kubernetes namespace")
    tail_lines: int = Field(50, ge=1, le=500)


class K8sPodLogsOutput(BaseModel):
    """Diagnostic log extraction from a Pod."""
    status: ToolExecutionStatus
    pod_name: str
    logs: str
    error_patterns_found: list[str] = Field(default_factory=list)


class K8sModifyClusterInput(BaseModel):
    """Safety-gated schema for mutating cluster resources."""
    action: str = Field(..., description="Action requested: restart, scale, rollback, delete")
    target_resource: str = Field(..., description="Deployment or Pod name")
    namespace: str = Field("mlops", description="Target namespace (restricted to approved namespaces)")
    user_confirmed: bool = Field(False, description="Must be True for mutation to proceed")
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        approved_namespaces = {"mlops", "default"}
        if v not in approved_namespaces:
            raise ValueError(f"Namespace '{v}' is unauthorized. Allowed namespaces: {approved_namespaces}")
        return v


class K8sModifyClusterOutput(BaseModel):
    """Result of attempting a cluster modification."""
    status: ToolExecutionStatus
    authorized: bool
    action: str
    message: str


# ============================================================================
# Tool Selection, Tracking & Evidence-Based Output Models
# ============================================================================

class ToolCallRequest(BaseModel):
    """Structured tool invocation decision formulated by the agent."""
    tool_name: str = Field(..., description="Registered name of tool to invoke")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Validated input parameters")
    justification: str = Field(..., description="Reason this tool is required for user query")


class ToolExecutionResult(BaseModel):
    """Audit record capturing tool outcome and latency."""
    tool_name: str
    status: ToolExecutionStatus
    duration_ms: float
    request_id: str
    data: dict[str, Any] | None = None
    error_message: str | None = None


class ProvenanceEvidence(BaseModel):
    """Concrete data provenance retrieved via DVC and Git."""
    dataset_name: str
    dataset_version: str
    git_commit: str
    dvc_remote: str | None = None
    tracked_files: list[str] = Field(default_factory=list)
    is_reproducible: bool = True


class MetricEvidence(BaseModel):
    """Model evaluation metrics retrieved from MLflow."""
    model_name: str
    run_id: str
    experiment_name: str
    primary_metric: str
    primary_value: float
    metrics: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, str] = Field(default_factory=dict)
    artifact_uri: str | None = None


class DeploymentEvidence(BaseModel):
    """Infrastructure and deployment health retrieved from Kubernetes."""
    service_name: str
    namespace: str
    deployed_version: str
    replicas_desired: int
    replicas_ready: int
    health_status: str
    recent_restarts: int
    unhealthy_pods: list[str] = Field(default_factory=list)


class DiagnosisEvidence(BaseModel):
    """System failure or anomaly diagnosis across MLOps layers."""
    layer: DiagnosisLayer
    summary: str
    root_cause: str
    evidence_snippets: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


# ============================================================================
# Primary Agent Response Object
# ============================================================================

class AgentResponse(BaseModel):
    """
    Evidence-grounded response conforming to the assignment specification:
    status, metrics, provenance, diagnosis, facts, uncertainties, and recommendations.
    Operational claims are grounded directly in tool results without hallucination.
    """
    query: str
    request_id: str
    status: AgentExecutionStatus
    summary: str

    # Grounded evidence objects
    provenance: ProvenanceEvidence | None = None
    metrics: list[MetricEvidence] | None = None
    deployment: DeploymentEvidence | None = None
    diagnosis: DiagnosisEvidence | None = None

    # Deliberate distinction between verified facts and uncertainties
    facts: list[str] = Field(
        default_factory=list,
        description="Verified operational facts supported by tool responses"
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Uncertainties or unverified assumptions identified during analysis"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable next steps recommended for the engineer"
    )

    # Retraining policy gate
    retraining_decision: RetrainingDecision = Field(
        RetrainingDecision.NOT_RECOMMENDED,
        description="Retraining decision grounded in freshness, quality gate, or degradation"
    )
    retraining_reason: str | None = Field(
        None,
        description="Justification for why retraining is or is not recommended"
    )

    # Audit & telemetry
    tool_executions: list[ToolExecutionResult] = Field(
        default_factory=list,
        description="Audit log of all tools executed during this request"
    )
    total_duration_ms: float = Field(
        ...,
        description="Total end-to-end request latency in milliseconds"
    )
