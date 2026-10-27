"""Bilingual semantic corpus loading, result serialization, and scoring."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from vla_robot_sorting.contracts.semantic import (
    GroundingConstraint,
    SemanticDecision,
    SemanticResult,
)


@dataclass(frozen=True, slots=True)
class SemanticCaseResult:
    case_id: str
    language: str
    difficulty: str
    expected_behavior: str
    observed_behavior: str | None
    normalized_intent_id: str | None
    target: dict[str, Any] | None
    destination: dict[str, Any] | None
    latency_s: float | None
    provider: str | None
    model_id: str | None
    fallback_used: bool | None
    prompt_sha256: str | None
    schema_sha256: str | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if self.latency_s is not None and self.latency_s < 0:
            raise ValueError("latency_s must be non-negative")


def load_corpus(path: str | Path) -> tuple[dict[str, Any], ...]:
    corpus_path = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        corpus_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{corpus_path}:{line_number}: case must be an object")
        records.append(raw)
    return tuple(records)


def corpus_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _constraint_to_dict(value: GroundingConstraint | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "reference": value.reference,
        "category": value.category,
        "attributes": asdict(value.attributes),
        "relation": value.relation,
        "relation_reference": value.relation_reference,
    }


def result_from_semantic(
    case: dict[str, Any],
    result: SemanticResult,
    *,
    latency_s: float,
) -> SemanticCaseResult:
    return SemanticCaseResult(
        case_id=str(case["case_id"]),
        language=str(case["language"]),
        difficulty=str(case["difficulty"]),
        expected_behavior=str(case["expected_behavior"]),
        observed_behavior=result.decision.value,
        normalized_intent_id=result.normalized_intent_id,
        target=_constraint_to_dict(result.target),
        destination=_constraint_to_dict(result.destination),
        latency_s=latency_s,
        provider=result.provenance.provider,
        model_id=result.provenance.model_id,
        fallback_used=result.provenance.fallback_used,
        prompt_sha256=result.provenance.prompt_sha256,
        schema_sha256=result.provenance.schema_sha256,
    )


def error_result(
    case: dict[str, Any],
    *,
    latency_s: float,
    error: str,
) -> SemanticCaseResult:
    return SemanticCaseResult(
        case_id=str(case["case_id"]),
        language=str(case["language"]),
        difficulty=str(case["difficulty"]),
        expected_behavior=str(case["expected_behavior"]),
        observed_behavior=None,
        normalized_intent_id=None,
        target=None,
        destination=None,
        latency_s=latency_s,
        provider=None,
        model_id=None,
        fallback_used=None,
        prompt_sha256=None,
        schema_sha256=None,
        error=error,
    )


def _normalized_gold_constraint(value: Any) -> dict[str, Any] | None:
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


def _flatten_constraint(value: dict[str, Any] | None, prefix: str) -> dict[str, str]:
    if value is None:
        return {}
    attrs = value.get("attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    raw = {
        f"{prefix}.category": value.get("category"),
        f"{prefix}.color": attrs.get("color"),
        f"{prefix}.shape": attrs.get("shape"),
        f"{prefix}.size": attrs.get("size"),
        f"{prefix}.location": attrs.get("location"),
        f"{prefix}.relation": value.get("relation"),
        f"{prefix}.relation_reference": value.get("relation_reference"),
    }
    return {key: str(item) for key, item in raw.items() if item is not None}


def _semantic_constraint_signature(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    attrs = value.get("attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    return {
        "category": value.get("category"),
        "attributes": {
            "color": attrs.get("color"),
            "shape": attrs.get("shape"),
            "size": attrs.get("size"),
            "location": attrs.get("location"),
        },
        "relation": value.get("relation"),
    }


def _field_counts(
    gold: dict[str, Any] | None,
    predicted: dict[str, Any] | None,
    *,
    prefix: str,
) -> tuple[int, int, int]:
    gold_fields = _flatten_constraint(gold, prefix)
    predicted_fields = _flatten_constraint(predicted, prefix)
    true_positive = sum(
        predicted_fields.get(key) == value
        for key, value in gold_fields.items()
    )
    false_negative = len(gold_fields) - true_positive
    false_positive = sum(
        key not in gold_fields or gold_fields[key] != value
        for key, value in predicted_fields.items()
    )
    return true_positive, false_positive, false_negative


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_semantic_results(
    corpus: Iterable[dict[str, Any]],
    results: Iterable[SemanticCaseResult],
) -> dict[str, Any]:
    cases = tuple(corpus)
    outputs = tuple(results)
    by_case = {item.case_id: item for item in outputs}
    if len(by_case) != len(outputs):
        raise ValueError("duplicate case_id in semantic results")

    expected_ids = {str(case["case_id"]) for case in cases}
    unknown = set(by_case) - expected_ids
    if unknown:
        raise ValueError(f"semantic results contain unknown case IDs: {sorted(unknown)}")

    behavior_correct = 0
    evaluated = 0
    intent_correct = 0
    intent_total = 0
    tp = fp = fn = 0
    error_count = 0
    per_language: dict[str, Counter[str]] = defaultdict(Counter)
    per_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    per_behavior: dict[str, Counter[str]] = defaultdict(Counter)
    latencies: list[float] = []
    fallback_count = 0

    for case in cases:
        case_id = str(case["case_id"])
        output = by_case.get(case_id)
        language = str(case["language"])
        difficulty = str(case["difficulty"])
        behavior = str(case["expected_behavior"])
        per_language[language]["corpus_total"] += 1
        per_difficulty[difficulty]["corpus_total"] += 1
        per_behavior[behavior]["corpus_total"] += 1
        if output is None:
            continue
        evaluated += 1
        per_language[language]["evaluated"] += 1
        per_difficulty[difficulty]["evaluated"] += 1
        per_behavior[behavior]["evaluated"] += 1
        if output.error is not None:
            error_count += 1
            continue
        if output.latency_s is not None:
            latencies.append(output.latency_s)
        if output.fallback_used:
            fallback_count += 1

        correct_behavior = output.observed_behavior == behavior
        behavior_correct += int(correct_behavior)
        per_language[language]["correct"] += int(correct_behavior)
        per_difficulty[difficulty]["correct"] += int(correct_behavior)
        per_behavior[behavior]["correct"] += int(correct_behavior)

        if behavior != SemanticDecision.STRUCTURED_SUCCESS.value:
            continue
        intent_total += 1
        intent_correct += int(
            output.normalized_intent_id == case.get("normalized_intent_id")
        )
        gold_target = _normalized_gold_constraint(case.get("target"))
        gold_destination = _normalized_gold_constraint(case.get("destination"))
        for gold, predicted, prefix in (
            (gold_target, output.target, "target"),
            (gold_destination, output.destination, "destination"),
        ):
            c_tp, c_fp, c_fn = _field_counts(gold, predicted, prefix=prefix)
            tp += c_tp
            fp += c_fp
            fn += c_fn

    equivalence_groups: dict[str, list[SemanticCaseResult]] = defaultdict(list)
    for case in cases:
        eq_id = case.get("semantic_equivalence_id")
        output = by_case.get(str(case["case_id"]))
        if eq_id is not None and output is not None and output.error is None:
            equivalence_groups[str(eq_id)].append(output)

    eq_total = 0
    eq_consistent = 0
    for group in equivalence_groups.values():
        if len(group) < 2:
            continue
        eq_total += 1
        signatures = {
            (
                item.observed_behavior,
                item.normalized_intent_id,
                json.dumps(
                    _semantic_constraint_signature(item.target),
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                json.dumps(
                    _semantic_constraint_signature(item.destination),
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            )
            for item in group
        }
        eq_consistent += int(len(signatures) == 1)

    def _breakdown(source: dict[str, Counter[str]]) -> dict[str, Any]:
        return {
            key: {
                "corpus_total": counts["corpus_total"],
                "evaluated": counts["evaluated"],
                "correct": counts["correct"],
                "accuracy": _safe_ratio(counts["correct"], counts["evaluated"]),
            }
            for key, counts in sorted(source.items())
        }

    return {
        "corpus_total": len(cases),
        "evaluated": evaluated,
        "missing": len(cases) - evaluated,
        "provider_errors": error_count,
        "behavior_accuracy": _safe_ratio(behavior_correct, evaluated),
        "behavior_correct": behavior_correct,
        "intent_exact_match": _safe_ratio(intent_correct, intent_total),
        "intent_correct": intent_correct,
        "intent_total": intent_total,
        "constraint_field_counts": {"tp": tp, "fp": fp, "fn": fn},
        "constraint_field_micro": _prf(tp, fp, fn),
        "bilingual_equivalence_consistency": _safe_ratio(eq_consistent, eq_total),
        "bilingual_equivalence_consistent": eq_consistent,
        "bilingual_equivalence_total": eq_total,
        "fallback_count": fallback_count,
        "latency_measured_count": len(latencies),
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "by_language": _breakdown(per_language),
        "by_difficulty": _breakdown(per_difficulty),
        "by_expected_behavior": _breakdown(per_behavior),
    }


def result_to_dict(result: SemanticCaseResult) -> dict[str, Any]:
    return asdict(result)


def result_from_dict(data: dict[str, Any]) -> SemanticCaseResult:
    return SemanticCaseResult(**data)
