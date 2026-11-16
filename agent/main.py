"""
CLI Entrypoint for MLOps AI Agent.

Allows developers and engineers to execute queries against the agent locally,
inspecting the structured JSON logs and the evidence-grounded response.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path when running as a direct script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import UserQueryRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="MLOps Production AI Agent CLI")
    parser.add_argument(
        "--query",
        type=str,
        default=(
            "Find the best recent model, identify the dataset version used to train it, "
            "check the deployment status, and explain whether retraining is necessary."
        ),
        help="Natural language query for the agent",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="mlops",
        help="Target Kubernetes namespace",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Provide explicit confirmation for state modifications",
    )
    args = parser.parse_args()

    orchestrator = MLOpsAgentOrchestrator()
    request = UserQueryRequest(
        query=args.query,
        target_namespace=args.namespace,
        user_confirmed=args.confirm,
    )

    print(f"\n[+] Executing Query: {request.query}")
    print("-" * 70)
    response = orchestrator.process_query(request)
    print("\n[+] Agent Response:")
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
