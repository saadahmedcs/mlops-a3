# Production-Ready MLOps AI Agent (Assignment 03)

[![CI](https://github.com/your-username/assignment-03-mlops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/assignment-03-mlops-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, production-ready MLOps AI Agent engineered for **Assignment 03: Building a Production-Ready MLOps AI Agent** (Fall 2026). The agent understands natural-language queries, dynamically orchestrates tools across **DVC**, **MLflow**, **Apache Airflow**, and **Kubernetes**, collects ground-truth operational evidence, and synthesizes structured, evidence-grounded responses without hallucinating results.

---

## Architecture Overview

```
                          User / ML Engineer
                       (Natural-language query)
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │          FastAPI Interface          │
               │   • Request Validation (Pydantic)   │
               │   • Request Tracing (X-Request-ID)  │
               │   • Secret Redaction & JSON Logging │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │        MLOps AI Orchestrator        │
               │   • Intent Decomposition            │
               │   • Tool Selection & Execution      │
               │   • Safety Guardrail Policy Engine  │
               └───────┬───────┬───────┬───────┬─────┘
                       │       │       │       │
         ┌─────────────┘       │       │       └─────────────┐
         ▼                     ▼       ▼                     ▼
┌─────────────────┐ ┌───────────────┐ ┌───────────────┐ ┌─────────────────┐
│       DVC       │ │    MLflow     │ │    Airflow    │ │   Kubernetes    │
│  Data Lineage   │ │  Experiment   │ │   Pipeline    │ │   Deployment    │
│  & Provenance   │ │ Intelligence  │ │ Orchestration │ │  & Diagnostics  │
└─────────────────┘ └───────────────┘ └───────────────┘ └─────────────────┘
         │                     │       │                     │
         └─────────────┐       │       │       ┌─────────────┘
                       ▼       ▼       ▼       ▼
               ┌─────────────────────────────────────┐
               │       Evidence-Based Response       │
               │   status • metrics • provenance     │
               │     diagnosis • recommendations     │
               └─────────────────────────────────────┘
```

---

## Directory Structure

```text
assignment-03-mlops-agent/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow (linting + pytest)
├── agent/
│   ├── main.py                    # Standalone CLI entrypoint
│   ├── orchestrator.py            # Multi-tool agent orchestration engine
│   ├── logger.py                  # Secure structured JSON logger with secret masking
│   ├── schemas.py                 # Pydantic models for queries, tools, and evidence
│   └── tools/
│       ├── __init__.py            # Tool exports and registry
│       ├── dvc_tools.py           # DVC dataset inspection and provenance
│       ├── mlflow_tools.py        # MLflow run tracking, best run, and metrics
│       ├── airflow_tools.py       # Airflow DAG status, trigger, and diagnostics
│       └── kubernetes_tools.py    # K8s deployment inspection and safe mutations
├── api/
│   └── main.py                    # Production FastAPI interface and middleware
├── dags/
│   └── training_pipeline.py       # Airflow DAG (validate -> preprocess -> train -> eval -> quality gate)
├── data/
│   ├── training/
│   │   └── train.py               # Data training wrapper entrypoint
│   └── training_data.csv          # Sample DVC-tracked dataset
├── training/
│   └── train.py                   # Model training script with MLflow logging
├── kubernetes/
│   ├── configmap.yaml             # Non-sensitive configuration
│   ├── deployment.yaml            # 2 replicas, probes, security context, limits
│   └── service.yaml               # ClusterIP service definition
├── tests/
│   ├── conftest.py                # Pytest fixtures and TestClient
│   ├── test_api.py                # FastAPI route tests
│   ├── test_logging.py            # Secret masking and decorator tests
│   └── test_schemas.py            # Pydantic schema validation tests
├── architecture.md                # Detailed system architecture document
├── pyproject.toml                 # Packaging, Ruff, and pytest configuration
├── requirements.txt               # Pinned project dependencies
└── README.md                      # Project documentation
```

---

## Quickstart & Installation

### 1. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the FastAPI Service
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Interacting with the Agent

### Mandatory Integration Scenario (cURL)
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
    "target_namespace": "mlops",
    "min_f1_threshold": 0.85
  }'
```

### CLI Interface
```bash
python agent/main.py --query "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
```

---

## Safety & Security Guardrails

1. **Secret Masking**: All logs automatically scrub passwords, bearer tokens, API keys, and AWS credentials.
2. **Kubernetes Mutation Gating**: State-changing operations (scale, restart, delete) are strictly rejected unless explicit user confirmation is passed (`user_confirmed=True`).
3. **Namespace Isolation**: Operations are restricted to approved namespaces (`mlops`, `default`). Attempts to operate outside approved boundaries raise a `403 Forbidden` error.
4. **No Hallucinated Claims**: The agent grounds all facts in verified tool outputs; unverified points are explicitly categorized under `uncertainties`.
5. **Retraining Guard**: Retraining is never triggered automatically; it is only recommended when backed by concrete degradation metrics or failed quality gates.

---

## Running Automated Tests

```bash
# Run full test suite with verbose output
pytest -v tests/
```
