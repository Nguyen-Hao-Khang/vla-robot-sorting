"""Native upstream source attestation: discriminate source drift from runtime trash."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _checker():
    file = Path(__file__).resolve().parents[2] / "scripts/check_native_manipulation_dependencies.py"
    spec = importlib.util.spec_from_file_location("_damh_dependency_attestation", file)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def local_pinned_repo(tmp_path):
    repo = tmp_path / "outer"
    nested = repo / "RoboTwin"
    nested.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "DAMH Test")
    _git(repo, "config", "user.email", "damh-test@example.invalid")
    tracked = nested / "envs" / "canonical.py"
    tracked.parent.mkdir()
    tracked.write_text("EXPECTED_SOURCE=1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return nested, tracked, _git(repo, "rev-parse", "HEAD")


def test_clean_git_repo_and_allowlisted_generated_cache(local_pinned_repo):
    nested, _tracked, pin = local_pinned_repo
    check = _checker()
    assert check.inspect_git_checkout(nested, pin)["source_accepted"] is True
    cache = nested / "__pycache__" / "canonical.cpython-310.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"synthetic pyc; never imported")
    runtime = nested / ".runtime" / "session-id.txt"
    runtime.parent.mkdir()
    runtime.write_text("diagnostic\n")
    result = check.inspect_git_checkout(nested, pin)
    assert result["source_accepted"] is True
    assert len(result["untracked_allowlisted"]) == 2
    assert result["tracked_modified"] == []


def test_runtime_allowlist_does_not_mask_untracked_code_or_config(local_pinned_repo):
    nested, _tracked, pin = local_pinned_repo
    check = _checker()
    for filename in (
        "bad_module.py",
        "new_config.yaml",
        "robot.urdf",
        ".runtime/bad_module.py",
        ".runtime/new_config.yaml",
        "logs/trap.py",
    ):
        file = nested / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("NOT_CANONICAL\n")
        result = check.inspect_git_checkout(nested, pin)
        assert result["source_accepted"] is False
        assert result["untracked_unrecognized"]
        file.unlink()


def test_changed_tracked_source_and_wrong_commit_are_independent_blockers(local_pinned_repo):
    nested, tracked, pin = local_pinned_repo
    check = _checker()
    tracked.write_text("CHANGED_SOURCE=1\n")
    record = check.inspect_git_checkout(nested, pin)
    assert not record["source_accepted"]
    assert record["head_matches"] is True
    assert record["tracked_modified"]
    tracked.write_text("EXPECTED_SOURCE=1\n")
    record = check.inspect_git_checkout(nested, "0" * 40)
    assert not record["head_matches"]
    assert not record["source_accepted"]


def test_full_diagnostic_reports_all_categories_without_native_execution(
    tmp_path, local_pinned_repo, capsys
):
    nested, _tracked, _pin = local_pinned_repo
    check = _checker()
    rc = check.diagnose_dependencies(nested)
    report = json.loads(capsys.readouterr().out)
    assert rc == 2  # fixture Git SHA differs from RoboTwin pin; Windows is not Linux
    assert report["native_acceptance"] is False
    assert report["controller_eligible"] is False
    assert "upstream_git_status" in report["checks"]
    assert "pinned_packages" in report["checks"]
    assert "curobo_source_git" in report["checks"]
    assert "cuda_runtime" in report["checks"]
    assert report["checks"]["upstream_git_status"]["value"]["head_matches"] is False
    assert report["all_preflight_gates_passed"] is False


def test_checker_cli_diagnostic_exits_nonzero_and_emits_json(local_pinned_repo):
    nested, _tracked, _pin = local_pinned_repo
    file = Path(__file__).resolve().parents[2] / "scripts/check_native_manipulation_dependencies.py"
    result = subprocess.run(
        [sys.executable, str(file), "--robotwin-root", str(nested), "--diagnose"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["schema"] == "damh.native-dependency-diagnostic.v1"
    assert payload["all_preflight_gates_passed"] is False
