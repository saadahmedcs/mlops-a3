"""
Structured and Secure Logging Framework for MLOps AI Agent.

Provides JSON-formatted logging, request context propagation via contextvars,
secret/credential redaction to prevent accidental token leakage,
and tool execution metrics tracking (tool name, status, duration_ms, request_id).
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypeVar, cast

# Context variable to correlate logs with an incoming request
current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="SYSTEM"
)


def set_request_id(request_id: str) -> None:
    """Set the active request ID for the current async task context."""
    current_request_id.set(request_id)


def get_request_id() -> str:
    """Retrieve the active request ID for the current context."""
    return current_request_id.get()


# Regex patterns matching sensitive information to prevent secrets from leaking into logs
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"(?i)(password|passwd|secret|api[_-]?key|token|auth[_-]?token)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?"),
    re.compile(r"(?i)(aws_access_key_id|aws_secret_access_key|aws_session_token)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?"),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----"),
]


def redact_secrets(content: Any) -> Any:
    """
    Recursively sanitize dictionaries, lists, strings, and primitives
    to redact API keys, passwords, and tokens.
    """
    if isinstance(content, str):
        masked = content
        for pattern in SENSITIVE_PATTERNS:
            if "bearer" in pattern.pattern.lower():
                masked = pattern.sub(r"\1[REDACTED]", masked)
            elif "PRIVATE KEY" in pattern.pattern:
                masked = pattern.sub("[REDACTED_PRIVATE_KEY]", masked)
            else:
                masked = pattern.sub(r"\1=[REDACTED]", masked)
        return masked
    elif isinstance(content, dict):
        sanitized: dict[str, Any] = {}
        for key, value in content.items():
            key_lower = str(key).lower()
            if any(term in key_lower for term in ("pass", "secret", "token", "key", "auth", "credential")):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = redact_secrets(value)
        return sanitized
    elif isinstance(content, list):
        return [redact_secrets(item) for item in content]
    return content


class StructuredJsonFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON logs enriched with request_id,
    timestamp in ISO 8601, logger name, and optional tool telemetry metadata.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
            "request_id": getattr(record, "request_id", get_request_id()),
        }

        # Include custom tool execution metadata if present on the record
        for attr in ("tool_name", "status", "duration_ms", "extra_info"):
            if hasattr(record, attr):
                log_payload[attr] = redact_secrets(getattr(record, attr))

        if record.exc_info:
            log_payload["exception"] = redact_secrets(self.formatException(record.exc_info))

        return json.dumps(log_payload, default=str)


def setup_logger(name: str = "mlops_agent", level: int = logging.INFO) -> logging.Logger:
    """Configure and retrieve a structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup is invoked multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# Default application logger instance
logger = setup_logger()

F = TypeVar("F", bound=Callable[..., Any])


def log_tool_execution(tool_name: str) -> Callable[[F], F]:
    """
    Decorator for tool functions to record structured telemetry:
    tool_name, status (SUCCESS, FAILED, REJECTED), duration_ms, and request_id.
    """

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                req_id = get_request_id()
                start_time = time.perf_counter()
                logger.info(
                    f"Initiating tool call '{tool_name}'",
                    extra={
                        "request_id": req_id,
                        "tool_name": tool_name,
                        "status": "STARTED",
                    },
                )
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.info(
                        f"Tool call '{tool_name}' completed successfully in {duration_ms}ms",
                        extra={
                            "request_id": req_id,
                            "tool_name": tool_name,
                            "status": "SUCCESS",
                            "duration_ms": duration_ms,
                        },
                    )
                    return result
                except Exception as exc:
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    status = "REJECTED" if "confirmation" in str(exc).lower() or "forbidden" in str(exc).lower() else "FAILED"
                    logger.error(
                        f"Tool call '{tool_name}' {status.lower()} in {duration_ms}ms: {exc}",
                        extra={
                            "request_id": req_id,
                            "tool_name": tool_name,
                            "status": status,
                            "duration_ms": duration_ms,
                        },
                    )
                    raise

            return cast(F, async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                req_id = get_request_id()
                start_time = time.perf_counter()
                logger.info(
                    f"Initiating tool call '{tool_name}'",
                    extra={
                        "request_id": req_id,
                        "tool_name": tool_name,
                        "status": "STARTED",
                    },
                )
                try:
                    result = func(*args, **kwargs)
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.info(
                        f"Tool call '{tool_name}' completed successfully in {duration_ms}ms",
                        extra={
                            "request_id": req_id,
                            "tool_name": tool_name,
                            "status": "SUCCESS",
                            "duration_ms": duration_ms,
                        },
                    )
                    return result
                except Exception as exc:
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    status = "REJECTED" if "confirmation" in str(exc).lower() or "forbidden" in str(exc).lower() else "FAILED"
                    logger.error(
                        f"Tool call '{tool_name}' {status.lower()} in {duration_ms}ms: {exc}",
                        extra={
                            "request_id": req_id,
                            "tool_name": tool_name,
                            "status": status,
                            "duration_ms": duration_ms,
                        },
                    )
                    raise

            return cast(F, sync_wrapper)

    return decorator
