"""Independent S1-S integration tests; never send provider requests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_robot_sorting.evaluation.gemini_budget import PersistentGeminiBudget
from vla_robot_sorting.semantics.gemini import SemanticProviderError


def _evaluator(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("run_quota_aware_gemini_evaluation")


@pytest.mark.parametrize("value", [False, True, 1.5, -1, 75.01, 0, 76])
def test_noninteger_or_out_of_range_job_ceiling_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="max_requests"):
        PersistentGeminiBudget(
            tmp_path / "budget.sqlite", job_id="test-job", max_requests=value
        )


@pytest.mark.parametrize("value", [-1, 0.5, True, float("nan"), float("inf")])
def test_budget_reservation_rejects_invalid_token_accounting(tmp_path, value):
    budget = PersistentGeminiBudget(tmp_path / "budget.sqlite", job_id="test")
    with pytest.raises(ValueError, match="input_tokens"):
        budget.reserve("gemini-3.5-flash-lite", value)
    assert budget.summary()["job_requests_reserved"] == 0


@pytest.mark.parametrize("stamp", [-1, float("nan"), float("inf"), "now"])
def test_budget_reservation_rejects_unusable_timestamps(tmp_path, stamp):
    budget = PersistentGeminiBudget(tmp_path / "budget.sqlite", job_id="test")
    with pytest.raises(ValueError, match="reservation time"):
        budget.reserve("gemini-3.5-flash-lite", 1, now=stamp)
    assert budget.summary()["job_requests_reserved"] == 0


def test_live_provider_failure_is_partial_not_success_without_actual_network(
    monkeypatch, tmp_path,
):
    evaluator = _evaluator(monkeypatch)
    observed = {"provider_attempts": 0}

    class NoNetworkTransport:
        def __init__(self, *, api_key: str):
            assert api_key == "synthetic-test-key"

        def generate_json(self, **kwargs):
            observed["provider_attempts"] += 1
            raise SemanticProviderError("synthetic transport failure")

    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(evaluator, "GeminiRESTTransport", NoNetworkTransport)
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-test-key")
    output = tmp_path / "attempts.jsonl"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "quota.py", "--mode", "live",
        "--confirm-exclusive-budget", "--model", "flash-lite",
        "--case-id", "BC-001", "--case-id", "BC-002",
        "--max-cases", "2", "--max-requests", "2",
        "--job-id", "synthetic-test", "--budget-db", str(tmp_path / "ledger.sqlite"),
        "--output-results", str(output), "--output-summary", str(summary_path),
    ])
    assert evaluator.main() == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "LIVE_PARTIAL_OR_ERROR_NOT_ACCEPTED"
    assert summary["requested_case_count"] == 2
    assert summary["completed_case_count"] == 1
    assert summary["provider_error_count"] == 1
    assert summary["stopped_early"] is True
    assert summary["request_attempts_reserved"] == 1
    assert observed["provider_attempts"] == 1
    assert "synthetic-test-key" not in summary_path.read_text(encoding="utf-8")


def test_empty_er2_selection_cannot_claim_success_or_spend_quota(monkeypatch, tmp_path):
    evaluator = _evaluator(monkeypatch)
    cases = [
        {
            "case_id": "BC-A",
            "expected_behavior": "structured_success",
            "normalized_intent_id": "A",
        },
        {
            "case_id": "BC-B",
            "expected_behavior": "structured_success",
            "normalized_intent_id": "B",
        },
    ]
    monkeypatch.setattr(evaluator, "load_corpus", lambda _: cases)
    monkeypatch.setattr(evaluator, "corpus_sha256", lambda _: "f" * 64)
    monkeypatch.setattr(
        evaluator, "result_from_dict",
        lambda record: SimpleNamespace(**record),
    )
    lite_results = tmp_path / "lite.jsonl"
    lite_results.write_text(
        "".join(
            json.dumps({
                "case_id": case["case_id"], "error": None,
                "observed_behavior": "structured_success",
                "normalized_intent_id": case["normalized_intent_id"],
            }) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "er2-empty-summary.json"
    monkeypatch.setattr(sys, "argv", [
        "quota.py", "--mode", "dry-run", "--model", "robotics-er2",
        "--max-cases", "8", "--lite-results", str(lite_results),
        "--output-summary", str(summary_path),
    ])
    assert evaluator.main() == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "NO_CASES_SELECTED_NO_REQUESTS"
    assert summary["selected_count"] == 0
    assert summary["request_attempts_reserved"] == 0
