# Architecture Documentation: Production MLOps AI Agent

This document details the software architecture, design patterns, security controls, and integration layers implemented for **Assignment 03: Building a Production-Ready MLOps AI Agent**.

---

## 1. System Architecture Overview

The system bridges conversational AI with production MLOps infrastructure:
1. **API Gateway / Ingestion Layer (`api/main.py`)**: Built on FastAPI. Enforces strict schema validation, request tracing via `X-Request-ID`, structured JSON logging, and Kubernetes health probes (`/health`).
2. **Agent Orchestration Engine (`agent/orchestrator.py`)**: Decomposes user intents, selects appropriate tools, executes operations with telemetry, aggregates evidence, and synthesizes grounded conclusions.
3. **Structured Tool Interfaces (`agent/tools/`)**: Modular interfaces wrapping external systems (DVC, MLflow, Airflow, Kubernetes) with explicit Pydantic inputs and outputs.
4. **Safety & Policy Guardrails (`agent/logger.py`, `agent/tools/kubernetes_tools.py`)**: Sanitizes logs, masks credentials, restricts Kubernetes operations to authorized namespaces, and rejects state mutations without human confirmation.

---

## 2. Tool Integration Matrix

| Tool Layer | Technology | Primary Operations | Key Schemas | Security & Safety Controls |
| :--- | :--- | :--- | :--- | :--- |
| **Data Management** | DVC + Git | Inspect datasets, detect diffs, reproduce revisions | `DvcInspectInput/Output`, `DvcDiffInput/Output` | Subprocess command whitelisting, non-destructive read operations |
| **Experiment Intelligence** | MLflow | Find best run by metric, compare runs, model registry | `MlflowBestRunInput/Output`, `MlflowCompareInput/Output` | Read-only experiment queries, quality gate threshold enforcement (F1 >= 0.85) |
| **Pipeline Orchestration** | Apache Airflow | Query DAG status, diagnose failed tasks, trigger pipelines | `AirflowDagStatusInput/Output`, `AirflowTriggerDagInput/Output` | Authenticated REST API calls, quality gate stage halts unvalidated promotions |
| **Infrastructure & Serving** | Kubernetes | Inspect pods, service status, pod logs, cluster mutation | `K8sInspectDeploymentInput/Output`, `K8sModifyClusterInput/Output` | Strict namespace boundary (`mlops`), explicit confirmation required for mutations |

---

## 3. Mandatory Multi-Tool Reasoning Workflow

When resolving complex queries such as:
> *"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."*

The orchestrator executes a 5-step correlation sequence:
1. **Query MLflow**: Discovers candidate run `run_25_a4c92e1` with top metric `f1_score: 0.892`. Extracts recorded metadata tags: `dvc.dataset_version = v2.1`, `git.commit = a4c92e1`.
2. **Inspect DVC**: Verifies dataset `data/training_data.csv` at version `v2.1` with commit `a4c92e1` from remote storage `s3://mlops-production-dvc-store/datasets`.
3. **Inspect Kubernetes**: Checks deployment `prediction-api` in namespace `mlops`. Detects 2/2 healthy pods running image tag `v2.1` with zero restarts.
4. **Query Airflow**: Verifies DAG `training_pipeline` run succeeded across all five stages (`validate_dataset` through `apply_model_quality_gate`).
5. **Retraining Decision Gate**: Correlates evidence:
   - Metric quality gate: PASSED (0.892 >= 0.850).
   - Deployment health: HEALTHY (2/2 ready, 0 restarts).
   - Version alignment: SYNCHRONIZED (serving v2.1 on latest dataset v2.1).
   - **Conclusion**: Retraining is `NOT_RECOMMENDED`. Retraining is never triggered automatically without explicit evidence of degradation.

---

## 4. Structured & Secure Telemetry

Every tool invocation and API request produces structured JSON log records:
```json
{
  "timestamp": "2026-11-16T15:15:20.123456+00:00",
  "level": "INFO",
  "logger": "mlops_agent",
  "message": "Tool call 'mlflow_get_best_run' completed successfully in 14.25ms",
  "request_id": "req_88f9a2bc10",
  "tool_name": "mlflow_get_best_run",
  "status": "SUCCESS",
  "duration_ms": 14.25
}
```
All credential patterns (`Bearer *`, `password=*`, `api_key=*`) are stripped before writing to stdout.
