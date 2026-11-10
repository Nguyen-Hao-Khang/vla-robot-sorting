#!/usr/bin/env python3
"""Audit provisional Sprint-1 native closure from readiness + manipulation candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_native_closure(readiness_path: Path) -> dict[str, Any]:
    raw = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("readiness report must be an object")

    accepted = raw.get("accepted")
    closure = raw.get("native_closure_candidate")
    checks = {
        "schema": raw.get("schema") == "sprint1-readiness-v1",
        "accepted_all_true": (
            isinstance(accepted, dict)
            and len(accepted) in {11, 12}
            and all(value is True for value in accepted.values())
            and raw.get("accepted_count") == len(accepted)
            and raw.get("accepted_total") == len(accepted)
        ),
        "native_pending_cleared": (
            raw.get("native_pending_count") == 0
            and raw.get("native_pending") == {}
        ),
        "manipulation_candidate_ready": (
            isinstance(closure, dict)
            and closure.get("provided") is True
            and closure.get("ready") is True
            and (
                (
                    closure.get("source") == "provisional_candidate"
                    and closure.get("classification") == "PASS_AUDITED_CANDIDATE"
                    and closure.get("promotion_required_on_windows") is True
                )
                or (
                    closure.get("source") == "durable_manifest"
                    and closure.get("classification") in {"PASS", "PASS_AUDITED"}
                    and closure.get("promotion_required_on_windows") is False
                )
            )
            and isinstance(closure.get("linux_git_head"), str)
            and bool(closure["linux_git_head"].strip())
        ),
        "cloud_api_kept_separate": (
            isinstance(raw.get("cloud_api_pending"), dict)
            and "gemini_live_semantic_evaluation" in raw["cloud_api_pending"]
        ),
    }
    passed = all(checks.values())
    durable = (
        isinstance(closure, dict)
        and closure.get("source") == "durable_manifest"
        and closure.get("promotion_required_on_windows") is False
    )
    classification = (
        "PASS_DURABLE" if passed and durable else ("PASS_CANDIDATE" if passed else "FAIL")
    )
    return {
        "schema": "sprint1-native-closure-audit-v1",
        "classification": classification,
        "passed": passed,
        "checks": checks,
        "readiness_path": str(readiness_path.resolve()),
        "windows_promotion_required": passed and not durable,
        "claim_boundary": (
            (
                "PASS_DURABLE means the manipulation gate has been promoted to a durable "
                "accepted manifest after Windows review; native Sprint-1 closure is durable."
            )
            if durable
            else (
                "PASS_CANDIDATE means all existing Sprint-1 accepted gates remain valid and "
                "the project-owned manipulation native gate has three fixed-seed independent "
                "audited passes. Durable Sprint-1 acceptance still requires Windows review "
                "and promotion of the exported manipulation candidate manifest."
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("readiness", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_native_closure(args.readiness)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"NATIVE_CLOSURE_AUDIT=FAIL: {type(exc).__name__}: {exc}")
        return 2
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
