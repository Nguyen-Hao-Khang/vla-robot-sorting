from __future__ import annotations

import sqlite3

import pytest

from vla_robot_sorting.evaluation.gemini_budget import (
    BudgetedGeminiTransport,
    BudgetExceeded,
    PersistentGeminiBudget,
)


class FakeTransport:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, **kwargs):
        self.calls += 1
        return {"ok": True}


def test_budget_persists_and_caps_job_requests(tmp_path):
    path = tmp_path / "ledger.sqlite"
    budget = PersistentGeminiBudget(path, job_id="job-1", max_requests=1)
    budget.reserve("gemini-3.5-flash-lite", 50, now=1_000_000)
    restored = PersistentGeminiBudget(path, job_id="job-1", max_requests=1)
    with pytest.raises(BudgetExceeded, match="job request budget"):
        restored.reserve("gemini-3.5-flash-lite", 50, now=1_000_001)


def test_budget_counts_request_before_transport_failure(tmp_path):
    class FailingTransport(FakeTransport):
        def generate_json(self, **kwargs):
            self.calls += 1
            raise RuntimeError("network failure")

    path = tmp_path / "ledger.sqlite"
    budget = PersistentGeminiBudget(path, job_id="failure", max_requests=1)
    transport = BudgetedGeminiTransport(FailingTransport(), budget)
    args = dict(
        model_id="gemini-3.5-flash-lite",
        system_instruction="s",
        user_prompt="p",
        response_schema={},
    )
    with pytest.raises(RuntimeError, match="network failure"):
        transport.generate_json(**args)
    with sqlite3.connect(path) as db:
        assert db.execute("select count(*) from reservations").fetchone()[0] == 1


def test_unsupported_model_fails_before_network(tmp_path):
    fake = FakeTransport()
    wrapped = BudgetedGeminiTransport(
        fake, PersistentGeminiBudget(tmp_path / "ledger.sqlite", job_id="j")
    )
    with pytest.raises(BudgetExceeded, match="unsupported model"):
        wrapped.generate_json(
            model_id="unlisted-model",
            system_instruction="",
            user_prompt="",
            response_schema={},
        )
    assert fake.calls == 0


def test_rate_limit_and_token_caps_are_fail_closed(tmp_path):
    budget = PersistentGeminiBudget(tmp_path / "ledger.sqlite", job_id="j")
    for index in range(10):
        budget.reserve("gemini-3.5-flash-lite", 1, now=1_000_000 + index)
    with pytest.raises(BudgetExceeded, match="RPM"):
        budget.reserve("gemini-3.5-flash-lite", 1, now=1_000_020)

    budget = PersistentGeminiBudget(tmp_path / "tokens.sqlite", job_id="tokens")
    budget.reserve("gemini-3.5-flash-lite", 150_000, now=1_000_000)
    with pytest.raises(BudgetExceeded, match="TPM"):
        budget.reserve("gemini-3.5-flash-lite", 1, now=1_000_001)


def test_failed_reservation_does_not_consume_slot(tmp_path):
    budget = PersistentGeminiBudget(tmp_path / "ledger.sqlite", job_id="j", max_requests=1)
    with pytest.raises(BudgetExceeded):
        budget.reserve("unknown", 1)
    budget.reserve("gemini-3.5-flash-lite", 1)
    assert budget.summary()["job_requests_reserved"] == 1
