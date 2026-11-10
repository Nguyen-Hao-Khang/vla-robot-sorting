from __future__ import annotations

from pathlib import Path
from typing import Any

from vla_robot_sorting.evaluation.semantic_corpus import (
    SemanticCaseResult,
    load_corpus,
    score_semantic_results,
)

CORPUS = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "corpora"
    / "bilingual_commands.jsonl"
)


def _constraint(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    attrs = value.get("attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    return {
        "reference": value.get("reference"),
        "category": value.get("category"),
        "attributes": {
            "color": attrs.get("color"),
            "shape": attrs.get("shape"),
            "size": attrs.get("size"),
            "location": attrs.get("location"),
        },
        "relation": value.get("relation"),
        "relation_reference": value.get("relation_reference"),
    }


def _oracle(case: dict[str, Any]) -> SemanticCaseResult:
    return SemanticCaseResult(
        case_id=str(case["case_id"]),
        language=str(case["language"]),
        difficulty=str(case["difficulty"]),
        expected_behavior=str(case["expected_behavior"]),
        observed_behavior=str(case["expected_behavior"]),
        normalized_intent_id=(
            str(case["normalized_intent_id"])
            if case.get("normalized_intent_id") is not None
            else None
        ),
        target=_constraint(case.get("target")),
        destination=_constraint(case.get("destination")),
        latency_s=0.01,
        provider="oracle-test",
        model_id="oracle-test",
        fallback_used=False,
        prompt_sha256="a" * 64,
        schema_sha256="b" * 64,
    )


def test_oracle_scores_full_66_case_corpus_without_language_penalty() -> None:
    cases = load_corpus(CORPUS)
    results = [_oracle(case) for case in cases]

    score = score_semantic_results(cases, results)

    assert score["corpus_total"] == 66
    assert score["evaluated"] == 66
    assert score["missing"] == 0
    assert score["provider_errors"] == 0
    assert score["behavior_accuracy"] == 1.0
    assert score["intent_exact_match"] == 1.0
    assert score["constraint_field_micro"]["f1"] == 1.0
    assert score["bilingual_equivalence_consistency"] == 1.0
    assert score["by_language"]["en"]["accuracy"] == 1.0
    assert score["by_language"]["vi"]["accuracy"] == 1.0
    assert score["by_language"]["code_switch"]["accuracy"] == 1.0


def test_partial_smoke_uses_evaluated_denominator_not_full_corpus() -> None:
    cases = load_corpus(CORPUS)
    result = _oracle(cases[0])

    score = score_semantic_results(cases, [result])

    assert score["corpus_total"] == 66
    assert score["evaluated"] == 1
    assert score["missing"] == 65
    assert score["behavior_accuracy"] == 1.0
    assert score["by_language"]["en"]["evaluated"] == 1
    assert score["by_language"]["vi"]["evaluated"] == 0
    assert score["by_language"]["vi"]["accuracy"] is None


def test_provider_error_is_counted_as_attempted_failure() -> None:
    cases = load_corpus(CORPUS)
    case = cases[0]
    failed = SemanticCaseResult(
        case_id=str(case["case_id"]),
        language=str(case["language"]),
        difficulty=str(case["difficulty"]),
        expected_behavior=str(case["expected_behavior"]),
        observed_behavior=None,
        normalized_intent_id=None,
        target=None,
        destination=None,
        latency_s=1.0,
        provider=None,
        model_id=None,
        fallback_used=None,
        prompt_sha256=None,
        schema_sha256=None,
        error="SemanticProviderError: unavailable",
    )

    score = score_semantic_results(cases, [failed])

    assert score["evaluated"] == 1
    assert score["provider_errors"] == 1
    assert score["behavior_accuracy"] == 0.0


def test_wrong_normalized_intent_and_constraint_reduce_separate_metrics() -> None:
    cases = load_corpus(CORPUS)
    case = next(item for item in cases if item["case_id"] == "BC-001")
    result = _oracle(case)
    wrong = SemanticCaseResult(
        case_id=result.case_id,
        language=result.language,
        difficulty=result.difficulty,
        expected_behavior=result.expected_behavior,
        observed_behavior=result.observed_behavior,
        normalized_intent_id="sort:wrong:wrong",
        target={
            **(result.target or {}),
            "category": "cup",
        },
        destination=result.destination,
        latency_s=result.latency_s,
        provider=result.provider,
        model_id=result.model_id,
        fallback_used=result.fallback_used,
        prompt_sha256=result.prompt_sha256,
        schema_sha256=result.schema_sha256,
    )

    score = score_semantic_results(cases, [wrong])

    assert score["behavior_accuracy"] == 1.0
    assert score["intent_exact_match"] == 0.0
    assert score["constraint_field_micro"]["f1"] < 1.0
