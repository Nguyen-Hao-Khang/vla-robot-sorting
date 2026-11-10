#!/usr/bin/env python3
"""Run or score the 66-case bilingual semantic corpus.

Live mode requires GEMINI_API_KEY and makes network/API calls. Contract and score
modes are offline and do not contact Gemini.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from vla_robot_sorting.contracts.semantic import SemanticRequest, SourceLanguage
from vla_robot_sorting.evaluation.semantic_corpus import (
    SemanticCaseResult,
    corpus_sha256,
    error_result,
    load_corpus,
    result_from_dict,
    result_from_semantic,
    result_to_dict,
    score_semantic_results,
)
from vla_robot_sorting.semantics.gemini import (
    GeminiSemanticReasoner,
    SemanticProviderError,
)

DEFAULT_CORPUS = Path("experiments/corpora/bilingual_commands.jsonl")


def _source_language(value: str) -> SourceLanguage:
    return SourceLanguage(value)


def _candidate_state(expectation: Any) -> dict[str, Any] | None:
    if expectation == "unique":
        return {"candidate_count": 1, "visibility": "visible"}
    if expectation == "multiple":
        return {"candidate_count": 2, "visibility": "visible"}
    if expectation == "not_visible":
        return {"candidate_count": 1, "visibility": "not_visible"}
    if expectation == "not_present":
        return {"candidate_count": 0, "visibility": "absent"}
    if expectation == "not_applicable":
        return None
    return {"candidate_count": None, "visibility": "unknown"}


def build_semantic_request(case: dict[str, Any]) -> SemanticRequest:
    grounding = case.get("grounding_expectation")
    grounding = grounding if isinstance(grounding, dict) else {}
    scene_summary = {
        "fixture": "bilingual-corpus-cardinality-v1",
        "target": _candidate_state(grounding.get("target")),
        "destination": _candidate_state(grounding.get("destination")),
        "note": (
            "Candidate counts/visibility are host-supplied scene evidence. "
            "Derive semantic constraints from the user command; do not invent motor data."
        ),
    }
    return SemanticRequest(
        command=str(case["command"]),
        source_language=_source_language(str(case["language"])),
        observation_revision=f"corpus-fixture-v1:{case['case_id']}",
        scene_summary=scene_summary,
        case_id=str(case["case_id"]),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, results: list[SemanticCaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result_to_dict(result),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )


def _read_results(path: Path) -> tuple[SemanticCaseResult, ...]:
    values: list[SemanticCaseResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number}: result must be an object")
        values.append(result_from_dict(raw))
    return tuple(values)


def _corpus_contract_summary(
    corpus_path: Path,
    cases: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "mode": "contract",
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus_sha256(corpus_path),
        "case_count": len(cases),
        "languages": dict(sorted(Counter(str(c["language"]) for c in cases).items())),
        "expected_behaviors": dict(
            sorted(Counter(str(c["expected_behavior"]) for c in cases).items())
        ),
        "difficulties": dict(
            sorted(Counter(str(c["difficulty"]) for c in cases).items())
        ),
        "network_calls": 0,
        "claim_boundary": (
            "Corpus/contract integrity only; no live-model or visual-grounding claim."
        ),
    }


def run_live(
    cases: tuple[dict[str, Any], ...],
    *,
    case_ids: set[str] | None = None,
    max_cases: int | None = None,
) -> list[SemanticCaseResult]:
    reasoner = GeminiSemanticReasoner.from_env()
    selected = [
        case
        for case in cases
        if case_ids is None or str(case["case_id"]) in case_ids
    ]
    if max_cases is not None:
        selected = selected[:max_cases]

    outputs: list[SemanticCaseResult] = []
    for case in selected:
        request = build_semantic_request(case)
        started = time.perf_counter()
        try:
            result = reasoner.reason(request)
        except (SemanticProviderError, ValueError, KeyError, TypeError) as exc:
            outputs.append(
                error_result(
                    case,
                    latency_s=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        outputs.append(
            result_from_semantic(
                case,
                result,
                latency_s=time.perf_counter() - started,
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("contract", "live", "score"),
        default="contract",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output-results", type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_corpus(args.corpus)

    if args.mode == "contract":
        summary = _corpus_contract_summary(args.corpus, cases)
    elif args.mode == "score":
        if args.results is None:
            raise SystemExit("--results is required in score mode")
        results = _read_results(args.results)
        summary = {
            "mode": "score",
            "corpus_sha256": corpus_sha256(args.corpus),
            "results_path": str(args.results),
            "scores": score_semantic_results(cases, results),
            "claim_boundary": (
                "Semantic structured-output scoring only; not visual instance grounding "
                "or manipulation success."
            ),
        }
    else:
        if args.output_results is None:
            raise SystemExit("--output-results is required in live mode")
        results = run_live(
            cases,
            case_ids=set(args.case_id) or None,
            max_cases=args.max_cases,
        )
        _write_jsonl(args.output_results, results)
        summary = {
            "mode": "live",
            "corpus_sha256": corpus_sha256(args.corpus),
            "results_path": str(args.output_results),
            "scores": score_semantic_results(cases, results),
            "claim_boundary": (
                "Live Gemini semantic/structured-output evaluation using a synthetic "
                "host scene-cardinality fixture; not visual instance grounding."
            ),
        }

    if args.output_summary is not None:
        _write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
