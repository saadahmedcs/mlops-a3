"""
FastAPI Interface for the Production MLOps AI Agent.

Provides RESTful HTTP APIs with:
- Strict Pydantic request/response validation
- Request-correlated structured logging (X-Request-ID)
- Global exception handling and security controls
- Liveness/readiness health probes for Kubernetes
- Tool discovery and query dispatch to MLOpsAgentOrchestrator
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.logger import get_request_id, logger, set_request_id
from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import (
    AgentExecutionStatus,
    AgentResponse,
    RetrainingDecision,
    ToolExecutionResult,
    ToolExecutionStatus,
    UserQueryRequest,
)

# Global orchestrator singleton
orchestrator = MLOpsAgentOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle events."""
    logger.info("Initializing MLOps AI Agent API service...")
    yield
    logger.info("Shutting down MLOps AI Agent API service...")


app = FastAPI(
    title="MLOps Production AI Agent API",
    description="Conversational and programmatic interface for inspecting and operating MLOps infrastructure (DVC, MLflow, Airflow, Kubernetes).",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for dashboard/web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_and_tracing_middleware(request: Request, call_next: Any) -> Response:
    """
    Middleware that establishes request tracing context (X-Request-ID),
    measures end-to-end HTTP request duration, and logs requests securely.
    """
    # Extract or generate unique request identifier
    req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    set_request_id(req_id)

    start_time = time.perf_counter()
    logger.info(
        f"Incoming HTTP request: {request.method} {request.url.path}",
        extra={"request_id": req_id, "status": "STARTED"},
    )

    try:
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        logger.info(
            f"Completed HTTP request: {request.method} {request.url.path} -> {response.status_code} in {duration_ms}ms",
            extra={"request_id": req_id, "status": "SUCCESS", "duration_ms": duration_ms},
        )
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            f"Unhandled exception during HTTP request {request.method} {request.url.path}: {exc}",
            extra={"request_id": req_id, "status": "FAILED", "duration_ms": duration_ms},
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred while processing the request.",
                "request_id": req_id,
            },
            headers={"X-Request-ID": req_id},
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Standardized validation error handler."""
    req_id = get_request_id()
    logger.warning(
        f"Request validation failed for {request.url.path}: {exc.errors()}",
        extra={"request_id": req_id, "status": "REJECTED"},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "The request payload failed schema validation.",
            "details": exc.errors(),
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id},
    )


@app.exception_handler(PermissionError)
async def permission_exception_handler(request: Request, exc: PermissionError) -> JSONResponse:
    """Security guardrail rejection handler."""
    req_id = get_request_id()
    logger.warning(
        f"Security guardrail rejected action: {exc}",
        extra={"request_id": req_id, "status": "REJECTED"},
    )
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={
            "error": "SecurityGuardrailRejection",
            "message": str(exc),
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id},
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Liveness and readiness probe for Kubernetes deployment and monitoring.
    Returns status 200 when service is operational.
    """
    return {
        "status": "healthy",
        "service": "mlops-ai-agent",
        "version": "1.0.0",
        "timestamp": time.time(),
        "request_id": get_request_id(),
    }


@app.post("/query", response_model=AgentResponse, tags=["Agent"])
async def process_user_query(request: UserQueryRequest) -> AgentResponse:
    """
    Primary conversational endpoint:
    Receives natural language queries, coordinates tools (DVC, MLflow, Airflow, Kubernetes),
    and returns an evidence-grounded response.
    """
    # Propagate request_id from header if present and not set in payload
    if not request.request_id:
        request.request_id = get_request_id()

    response = orchestrator.process_query(request)
    return response


@app.get("/tools", tags=["Tooling"])
async def list_available_tools() -> Dict[str, Any]:
    """
    Tool discovery endpoint exposing tool capabilities and authorization policies.
    """
    return {
        "tools": [
            {
                "name": "dvc_inspect_datasets",
                "category": "Data Management",
                "description": "Inspects DVC-tracked datasets, Git commit provenance, and working tree diffs.",
                "read_only": True,
            },
            {
                "name": "dvc_compare_versions",
                "category": "Data Management",
                "description": "Compares data file changes across Git commits or DVC revisions.",
                "read_only": True,
            },
            {
                "name": "dvc_reproduce_version",
                "category": "Data Management",
                "description": "Provides step-by-step commands to checkout and pull past dataset states.",
                "read_only": True,
            },
            {
                "name": "mlflow_get_best_run",
                "category": "Experiment Tracking",
                "description": "Identifies the best model run based on a target metric (e.g. F1) and evaluates quality gates.",
                "read_only": True,
            },
            {
                "name": "mlflow_compare_runs",
                "category": "Experiment Tracking",
                "description": "Compares multiple runs across metrics, hyperparameters, and artifacts.",
                "read_only": True,
            },
            {
                "name": "airflow_get_dag_status",
                "category": "Workflow Orchestration",
                "description": "Retrieves execution status and failed tasks for training DAGs.",
                "read_only": True,
            },
            {
                "name": "airflow_trigger_dag",
                "category": "Workflow Orchestration",
                "description": "Triggers approved model training pipeline DAG.",
                "read_only": False,
            },
            {
                "name": "airflow_diagnose_task",
                "category": "Workflow Orchestration",
                "description": "Extracts task error logs and explains failure root causes.",
                "read_only": True,
            },
            {
                "name": "k8s_inspect_deployment",
                "category": "Infrastructure",
                "description": "Inspects Kubernetes Pods, replicas, health status, and deployed image versions.",
                "read_only": True,
            },
            {
                "name": "k8s_get_pod_logs",
                "category": "Infrastructure",
                "description": "Fetches diagnostic logs from serving pods.",
                "read_only": True,
            },
            {
                "name": "k8s_modify_cluster_state",
                "category": "Infrastructure",
                "description": "Performs cluster mutations. Requires explicit human confirmation.",
                "read_only": False,
            },
        ]
    }
