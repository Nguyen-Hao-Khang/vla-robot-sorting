"""Persistent, fail-closed request accounting for bounded Gemini evaluations."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from vla_robot_sorting.semantics.gemini import GeminiTransport


@dataclass(frozen=True, slots=True)
class ModelQuota:
    requests_per_minute: int
    input_tokens_per_minute: int
    requests_per_day: int


MODEL_QUOTAS = {
    "gemini-3.5-flash-lite": ModelQuota(10, 150_000, 100),
    "gemini-robotics-er-2-preview": ModelQuota(2, 80_000, 8),
}


class BudgetExceeded(RuntimeError):
    """Raised before network I/O when persistent budget cannot be guaranteed."""


class PersistentGeminiBudget:
    """SQLite ledger; reservations are charged before a network attempt."""

    def __init__(self, path: str | Path, *, job_id: str, max_requests: int = 75) -> None:
        if not job_id.strip():
            raise ValueError("job_id must be non-empty")
        if type(max_requests) is not int or not 1 <= max_requests <= 75:
            raise ValueError("max_requests must be an integer between 1 and 75")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id
        self.max_requests = max_requests
        with closing(self._connect()) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reservations "
                "(job_id TEXT NOT NULL, model_id TEXT NOT NULL, reserved_at REAL NOT NULL, "
                "input_tokens INTEGER NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def summary(self) -> dict[str, Any]:
        now = time.time()
        with closing(self._connect()) as db:
            count = db.execute(
                "SELECT count(*) FROM reservations WHERE job_id=?", (self.job_id,)
            ).fetchone()[0]
            totals = {
                model: db.execute(
                    "SELECT count(*), coalesce(sum(input_tokens),0) FROM reservations "
                    "WHERE model_id=? AND reserved_at>=?",
                    (model, now - 60),
                ).fetchone()
                for model in MODEL_QUOTAS
            }
            daily = {
                model: db.execute(
                    "SELECT count(*) FROM reservations WHERE model_id=? AND reserved_at>=?",
                    (model, now - 86_400),
                ).fetchone()[0]
                for model in MODEL_QUOTAS
            }
        return {
            "job_requests_reserved": count,
            "job_request_limit": self.max_requests,
            "requests_last_60s_by_model": {k: v[0] for k, v in totals.items()},
            "input_tokens_last_60s_by_model": {k: v[1] for k, v in totals.items()},
            "requests_last_24h_by_model": daily,
        }

    def reserve(self, model_id: str, input_tokens: int, *, now: float | None = None) -> None:
        quota = MODEL_QUOTAS.get(model_id)
        if quota is None:
            raise BudgetExceeded("unsupported model; request was not sent")
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError("input_tokens must be a non-negative integer")
        stamp = time.time() if now is None else now
        if not isinstance(stamp, (int, float)) or not isfinite(stamp) or stamp < 0:
            raise ValueError("reservation time must be finite and non-negative")
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            job_count = db.execute(
                "SELECT count(*) FROM reservations WHERE job_id=?", (self.job_id,)
            ).fetchone()[0]
            minute = db.execute(
                "SELECT count(*), coalesce(sum(input_tokens),0) FROM reservations "
                "WHERE model_id=? AND reserved_at>=?",
                (model_id, stamp - 60),
            ).fetchone()
            day_count = db.execute(
                "SELECT count(*) FROM reservations WHERE model_id=? AND reserved_at>=?",
                (model_id, stamp - 86_400),
            ).fetchone()[0]
            if job_count >= self.max_requests:
                raise BudgetExceeded("job request budget exhausted; request was not sent")
            if minute[0] >= quota.requests_per_minute:
                raise BudgetExceeded("model RPM budget exhausted; request was not sent")
            if minute[1] + input_tokens > quota.input_tokens_per_minute:
                raise BudgetExceeded("model input TPM budget exhausted; request was not sent")
            if day_count >= quota.requests_per_day:
                raise BudgetExceeded("model daily request budget exhausted; request was not sent")
            db.execute(
                "INSERT INTO reservations(job_id,model_id,reserved_at,input_tokens) "
                "VALUES(?,?,?,?)",
                (self.job_id, model_id, stamp, input_tokens),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


class BudgetedGeminiTransport:
    """Reserve conservatively before each actual provider request; no retries."""

    def __init__(self, transport: GeminiTransport, budget: PersistentGeminiBudget) -> None:
        self._transport = transport
        self._budget = budget

    def generate_json(
        self,
        *,
        model_id: str,
        system_instruction: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        schema_text = json.dumps(response_schema, ensure_ascii=False, sort_keys=True)
        # UTF-8 byte count is a conservative upper bound, not a tokenizer measurement.
        token_bound = len((system_instruction + user_prompt + schema_text).encode("utf-8"))
        self._budget.reserve(model_id, token_bound)
        return self._transport.generate_json(
            model_id=model_id,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_schema=response_schema,
        )
