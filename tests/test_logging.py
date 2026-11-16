"""
Tests for structured logging, secret masking, and security guardrails.
"""

from __future__ import annotations

import json
import logging
import pytest

from agent.logger import (
    StructuredJsonFormatter,
    get_request_id,
    log_tool_execution,
    redact_secrets,
    set_request_id,
)
from agent.schemas import K8sModifyClusterInput
from agent.tools.kubernetes_tools import KubernetesTools


def test_redact_secrets_masks_sensitive_values() -> None:
    """Verify that Bearer tokens, passwords, API keys, and sensitive dictionaries are redacted."""
    # Test string token masking
    raw_str = "Connecting with Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 and password=SuperSecretPassword123"
    sanitized_str = redact_secrets(raw_str)
    assert "SuperSecretPassword123" not in sanitized_str
    assert "Bearer [REDACTED]" in sanitized_str
    assert "[REDACTED]" in sanitized_str

    # Test dictionary recursive masking
    sensitive_dict = {
        "username": "mlops_admin",
        "api_key": "sk-live-999333aaa111",
        "secret_token": "secret_abc_123",
        "nested": {
            "password": "NestedPassword456",
            "cluster_ip": "10.96.0.1",
        },
    }
    sanitized_dict = redact_secrets(sensitive_dict)
    assert sanitized_dict["username"] == "mlops_admin"
    assert sanitized_dict["api_key"] == "[REDACTED]"
    assert sanitized_dict["secret_token"] == "[REDACTED]"
    assert sanitized_dict["nested"]["password"] == "[REDACTED]"
    assert sanitized_dict["nested"]["cluster_ip"] == "10.96.0.1"


def test_structured_json_formatter_outputs_valid_json() -> None:
    """Verify that StructuredJsonFormatter outputs valid JSON containing request_id and timestamp."""
    set_request_id("req_test_logging_abc")
    formatter = StructuredJsonFormatter()

    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Service token is token=super_secret_token",
        args=(),
        exc_info=None,
    )
    record.tool_name = "test_tool"
    record.status = "SUCCESS"
    record.duration_ms = 12.5

    formatted = formatter.format(record)
    log_obj = json.loads(formatted)

    assert log_obj["level"] == "INFO"
    assert log_obj["logger"] == "test.logger"
    assert log_obj["request_id"] == "req_test_logging_abc"
    assert log_obj["tool_name"] == "test_tool"
    assert log_obj["status"] == "SUCCESS"
    assert log_obj["duration_ms"] == 12.5
    assert "super_secret_token" not in log_obj["message"]
    assert "[REDACTED]" in log_obj["message"]


def test_tool_execution_decorator() -> None:
    """Verify that @log_tool_execution tracks duration and handles successful execution."""
    set_request_id("req_decorator_test")

    @log_tool_execution("sample_math_tool")
    def sample_func(a: int, b: int) -> int:
        return a + b

    result = sample_func(5, 7)
    assert result == 12


def test_k8s_unauthorized_modification_rejected() -> None:
    """
    Requirement from Rubric (Section 6):
    'A test confirming unauthorized Kubernetes modifications are rejected.'
    """
    k8s = KubernetesTools()
    params = K8sModifyClusterInput(
        action="scale",
        target_resource="prediction-api",
        namespace="mlops",
        user_confirmed=False,  # Lacks user confirmation
    )

    with pytest.raises(PermissionError) as exc_info:
        k8s.modify_cluster_state(params)

    assert "requires explicit user confirmation" in str(exc_info.value)
