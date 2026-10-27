#!/usr/bin/env python3
"""Quota-limited Gemini semantic evaluation; dry-run is the safe default."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from run_bilingual_semantic_evaluation import DEFAULT_CORPUS, build_semantic_request

from vla_robot_sorting.evaluation.gemini_budget import (
    BudgetedGeminiTransport,
    PersistentGeminiBudget,
)
from vla_robot_sorting.evaluation.semantic_corpus import (
    corpus_sha256,
    error_result,
    load_corpus,
    result_from_dict,
    result_from_semantic,
    result_to_dict,
    score_semantic_results,
)
from vla_robot_sorting.semantics.gemini import (
    GeminiModelConfig,
    GeminiRESTTransport,
    GeminiSemanticReasoner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", choices=("flash-lite", "robotics-er2"), default="flash-lite")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=66)
    parser.add_argument("--max-requests", type=int, default=75)
    parser.add_argument("--job-id", default="s1s-default")
    parser.add_argument(
        "--budget-db", type=Path, default=Path("experiments/results/s1s_budget.sqlite")
    )
    parser.add_argument("--output-results", type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--lite-results", type=Path)
    parser.add_argument("--confirm-exclusive-budget", action="store_true")
    args = parser.parse_args()

    if args.max_cases < 1 or args.max_cases > args.max_requests:
        parser.error("--max-cases must be between 1 and --max-requests")
    model_id = (
        "gemini-3.5-flash-lite" if args.model == "flash-lite" else "gemini-robotics-er-2-preview"
    )
    if args.model == "robotics-er2" and args.max_cases > 8:
        parser.error("Robotics ER2 comparison is capped at 8 selected cases")
    cases = load_corpus(args.corpus)
    selected_ids = set(args.case_id)
    selected = [case for case in cases if not selected_ids or str(case["case_id"]) in selected_ids]
    if args.model == "robotics-er2":
        if args.case_id:
            parser.error("do not combine --case-id with Lite-based ER2 selection")
        if args.lite_results is None:
            parser.error("--lite-results is required for Robotics ER2 comparison selection")
        lite_records = [
            result_from_dict(json.loads(line))
            for line in args.lite_results.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gold = {str(case["case_id"]): case for case in cases}
        if any(record.case_id not in gold for record in lite_records):
            parser.error("Lite results contain case IDs absent from the corpus")
        lite_ids = [record.case_id for record in lite_records]
        if len(set(lite_ids)) != len(lite_ids) or set(lite_ids) != set(gold):
            parser.error("Lite results must contain exactly one result for every corpus case")

        def lite_priority(record):
            case = gold[record.case_id]
            if record.error is not None:
                return 0
            if record.observed_behavior != case["expected_behavior"]:
                return 1
            if (
                record.observed_behavior == "structured_success"
                and record.normalized_intent_id != case.get("normalized_intent_id")
            ):
                return 2
            return 3

        candidates = [record for record in lite_records if lite_priority(record) < 3]
        selected_ids = {record.case_id for record in sorted(candidates, key=lite_priority)[:8]}
        selected = [case for case in cases if str(case["case_id"]) in selected_ids]
        selected = selected[: args.max_cases]
    else:
        selected = selected[: args.max_cases]
    if args.case_id and len(selected) != len(selected_ids):
        parser.error("one or more --case-id values are absent from the corpus")
    summary = {
        "mode": args.mode,
        "corpus_path": str(args.corpus),
        "corpus_sha256": corpus_sha256(args.corpus),
        "selected_case_ids": [str(case["case_id"]) for case in selected],
        "selected_count": len(selected),
        "model_id": model_id,
        "max_requests": args.max_requests,
        "selection_basis": (
            "Lite errors/behavior/intent failures"
            if args.model == "robotics-er2"
            else "corpus order or explicit case IDs"
        ),
        "request_attempts_reserved": 0,
        "credential_status": "not_checked_in_dry_run",
        "claim_boundary": (
            "Semantic structured-output evaluation with synthetic corpus scene-cardinality "
            "fixture; no visual grounding or manipulation claim."
        ),
    }
    if not selected:
        summary["status"] = "NO_CASES_SELECTED_NO_REQUESTS"
        _emit(summary, args.output_summary)
        return 2 if args.mode == "live" else 0
    if args.mode == "dry-run":
        summary["status"] = "DRY_RUN_ONLY"
    else:
        if not args.confirm_exclusive_budget:
            summary.update(
                status="LIVE_BLOCKED_SHARED_BUDGET_UNCONFIRMED", credential_status="not_checked"
            )
            _emit(summary, args.output_summary)
            return 2
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            summary.update(
                status="LIVE_BLOCKED_NO_SECURE_CREDENTIAL", credential_status="unavailable"
            )
            _emit(summary, args.output_summary)
            return 2
        if args.output_results is None:
            parser.error("--output-results is required in live mode")
        budget = PersistentGeminiBudget(
            args.budget_db, job_id=args.job_id, max_requests=args.max_requests
        )
        reserved_before = budget.summary()["job_requests_reserved"]
        raw_transport = GeminiRESTTransport(api_key=api_key)
        transport = BudgetedGeminiTransport(raw_transport, budget)
        config = GeminiModelConfig(
            primary_model=model_id,
            fallback_model=(
                "gemini-robotics-er-2-preview"
                if model_id == "gemini-3.5-flash-lite"
                else "gemini-3.5-flash-lite"
            ),
        )
        reasoner = GeminiSemanticReasoner(transport, config=config, fallback_enabled=False)
        outputs = []
        for case in selected:
            started = time.perf_counter()
            try:
                result = reasoner.reason(build_semantic_request(case))
                outputs.append(
                    result_from_semantic(case, result, latency_s=time.perf_counter() - started)
                )
            except Exception as exc:  # keep request contents/provider bodies out of evidence
                outputs.append(
                    error_result(
                        case, latency_s=time.perf_counter() - started, error=type(exc).__name__
                    )
                )
                if type(exc).__name__ in {"BudgetExceeded", "SemanticProviderError"}:
                    # Stop after any provider/quota failure; do not amplify quota/schema problems.
                    break
        args.output_results.parent.mkdir(parents=True, exist_ok=True)
        args.output_results.write_text(
            "".join(
                json.dumps(result_to_dict(item), sort_keys=True, ensure_ascii=False) + "\n"
                for item in outputs
            ),
            encoding="utf-8",
        )
        budget_summary = budget.summary()
        completed = len(outputs)
        error_count = sum(record.error is not None for record in outputs)
        full_success = completed == len(selected) and error_count == 0
        summary.update(
            status=(
                "LIVE_COMPLETED_WITH_EVIDENCE" if full_success
                else "LIVE_PARTIAL_OR_ERROR_NOT_ACCEPTED"
            ),
            credential_status="available_in_process_only",
            request_attempts_reserved=(
                budget_summary["job_requests_reserved"] - reserved_before
            ),
            total_job_requests_reserved=budget_summary["job_requests_reserved"],
            requested_case_count=len(selected),
            completed_case_count=completed,
            provider_error_count=error_count,
            stopped_early=completed < len(selected),
            results_path=str(args.output_results),
            budget=budget_summary,
            scores=score_semantic_results(cases, outputs),
        )
    _emit(summary, args.output_summary)
    return 0


def _emit(summary: dict, path: Path | None) -> None:
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    raise SystemExit(main())
