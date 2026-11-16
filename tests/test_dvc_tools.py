"""
Unit tests for the DVC Data Management Agent module (agent/tools/dvc_tools.py).
"""

from __future__ import annotations

import pytest

from agent.schemas import (
    DvcDiffInput,
    DvcInspectInput,
    DvcReproduceInput,
    ToolExecutionStatus,
)
from agent.tools.dvc_tools import DvcTools


@pytest.fixture
def dvc_tools() -> DvcTools:
    return DvcTools(repo_path=".")


def test_inspect_datasets_success(dvc_tools: DvcTools) -> None:
    """Test Tool 1: Inspecting DVC-tracked datasets and Git commit provenance."""
    result = dvc_tools.inspect_datasets(DvcInspectInput())

    assert result.status == ToolExecutionStatus.SUCCESS
    assert len(result.current_commit) >= 7
    assert len(result.tracked_datasets) >= 1

    ds = result.tracked_datasets[0]
    assert "data/training_data.csv" in ds.path
    assert ds.version_tag == "v2.1"
    assert ds.remote_url is not None
    assert "s3://" in ds.remote_url
    assert ds.md5_hash is not None


def test_inspect_datasets_untracked_target_path(dvc_tools: DvcTools) -> None:
    """Test Tool 1 error handling: Requesting an untracked/non-existent file."""
    result = dvc_tools.inspect_datasets(
        DvcInspectInput(target_path="non_existent_folder/missing_file.csv")
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "does not exist" in result.message
    assert len(result.tracked_datasets) == 0


def test_compare_dataset_versions_same_revision(dvc_tools: DvcTools) -> None:
    """Test Tool 2: Comparing identical revisions returns has_changed=False."""
    result = dvc_tools.compare_dataset_versions(DvcDiffInput(rev_a="HEAD", rev_b="HEAD"))

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.has_changed is False
    assert len(result.modified_files) == 0
    assert "identical" in result.summary.lower()


def test_compare_dataset_versions_different_revisions(dvc_tools: DvcTools) -> None:
    """Test Tool 2: Comparing different revisions reports data modifications."""
    result = dvc_tools.compare_dataset_versions(DvcDiffInput(rev_a="v2.0", rev_b="HEAD"))

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.has_changed is True
    assert "data/training_data.csv" in result.modified_files
    assert "v2.0" in result.summary
    assert "v2.1" in result.summary or "HEAD" in result.summary


def test_reproduction_instructions(dvc_tools: DvcTools) -> None:
    """Test Tool 3: Explaining and generating step-by-step reproduction instructions."""
    commit_sha = "a4c92e1"
    result = dvc_tools.get_reproduction_instructions(
        DvcReproduceInput(git_commit=commit_sha, target_path="data/training_data.csv")
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.git_commit == commit_sha
    assert result.checkout_command == f"git checkout {commit_sha}"
    assert "dvc pull data/training_data.csv" in result.dvc_pull_command
    assert "git checkout" in result.instructions
    assert "dvc pull" in result.instructions
    assert "dvc checkout" in result.instructions
    assert "dvc status" in result.instructions


def test_security_rejection_forbidden_command(dvc_tools: DvcTools) -> None:
    """Security check: Attempting to run unapproved commands must raise PermissionError."""
    with pytest.raises(PermissionError) as exc_info:
        dvc_tools._run_safe_command(["git", "reset", "--hard", "HEAD~1"])

    assert "forbidden" in str(exc_info.value).lower()


def test_security_rejection_injection_characters(dvc_tools: DvcTools) -> None:
    """Security check: Unsafe arguments with semicolons or shell pipes must raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        dvc_tools._sanitize_arg("v2.1; rm -rf /", param_name="revision")

    assert "contains unsafe characters" in str(exc_info.value)
