#!/usr/bin/env python3
"""Promote an exported manipulation closure candidate to a durable accepted manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from vla_robot_sorting.evidence import audit_project_manipulation_probe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(
    candidate_path: Path,
    evidence_dir: Path,
    *,
    source_bridge_relative: str,
    review: str,
    accepted_at: str,
) -> dict[str, Any]:
    candidate_path = candidate_path.resolve()
    evidence_dir = evidence_dir.resolve()
    raw = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("candidate must be an object")
    if raw.get("schema") != "native-manipulation-closure-candidate-v1":
        raise ValueError("unsupported manipulation closure candidate schema")
    if raw.get("gate_id") != "S1.6-MANIPULATION":
        raise ValueError("candidate gate mismatch")
    if raw.get("classification") != "PASS_AUDITED_CANDIDATE":
        raise ValueError("candidate classification is not pass-audited")
    if raw.get("promotion_required_on_windows") is not True:
        raise ValueError("candidate does not require Windows promotion")
    if raw.get("fixed_seeds") != [0, 1, 2]:
        raise ValueError("candidate fixed seeds must be 0,1,2")
    if raw.get("probe_pass_count") != 3 or raw.get("independent_audit_pass_count") != 3:
        raise ValueError("candidate pass counts are not 3/3")

    verified: list[dict[str, Any]] = []
    seed_rows = raw.get("seeds")
    if not isinstance(seed_rows, list) or len(seed_rows) != 3:
        raise ValueError("candidate seed evidence is incomplete")
    for row in seed_rows:
        if not isinstance(row, dict):
            raise ValueError("candidate seed row must be an object")
        seed = int(row.get("seed"))
        probe = evidence_dir / str(row.get("probe_relative"))
        audit_log = evidence_dir / str(row.get("audit_log_relative"))
        if not probe.is_file() or not audit_log.is_file():
            raise ValueError(f"seed {seed} evidence file missing")
        probe_sha = _sha256(probe)
        audit_sha = _sha256(audit_log)
        if probe_sha != row.get("probe_sha256"):
            raise ValueError(f"seed {seed} probe SHA256 mismatch")
        if audit_sha != row.get("audit_log_sha256"):
            raise ValueError(f"seed {seed} audit-log SHA256 mismatch")
        probe_raw = json.loads(probe.read_text(encoding="utf-8"))
        if not isinstance(probe_raw, dict) or probe_raw.get("passed") is not True:
            raise ValueError(f"seed {seed} probe no longer reports PASS")
        audit = audit_project_manipulation_probe(probe)
        if not audit.passed:
            names = [gate.name for gate in audit.failures]
            raise ValueError(f"seed {seed} independent re-audit failed: {names}")
        audit_text = audit_log.read_text(encoding="utf-8", errors="replace")
        if "AUDIT=PASS" not in audit_text:
            raise ValueError(f"seed {seed} audit log no longer reports AUDIT=PASS")
        recovered = row.get("recovered") is True
        candidate_id = row.get("candidate_id")
        if recovered and seed != 1:
            raise ValueError("only seed 1 may use deterministic recovery")
        if recovered and (
            not isinstance(candidate_id, str) or not candidate_id.strip()
        ):
            raise ValueError("recovered seed 1 is missing candidate_id")
        verified.append(
            {
                "seed": seed,
                "source_label": row.get("source_label"),
                "recovered": recovered,
                "candidate_id": candidate_id,
                "probe_relative": row["probe_relative"],
                "probe_sha256": probe_sha,
                "audit_log_relative": row["audit_log_relative"],
                "audit_log_sha256": audit_sha,
                "final_phase": probe_raw.get("final_phase"),
                "task_success_final": probe_raw.get("task_success_final"),
            }
        )

    if sorted(row["seed"] for row in verified) != [0, 1, 2]:
        raise ValueError("verified fixed seeds are not exactly 0,1,2")
    if not source_bridge_relative.strip():
        raise ValueError("source_bridge_relative must be non-empty")
    if not review.strip():
        raise ValueError("review must be non-empty")

    return {
        "schema": "accepted-native-gate-v1",
        "gate_id": "S1.6-MANIPULATION",
        "classification": "PASS_AUDITED",
        "accepted_at": accepted_at,
        "review": review,
        "source_bridge_relative": source_bridge_relative,
        "linux_git_head": raw.get("linux_git_head"),
        "candidate_sha256": _sha256(candidate_path),
        "fixed_seeds": [0, 1, 2],
        "probe_pass_count": 3,
        "independent_audit_pass_count": 3,
        "seeds": verified,
        "claim_boundary": (
            "Accepted only after Windows review re-verified every exported probe hash, "
            "audit-log hash, probe PASS state and independent manipulation audit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-bridge-relative", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--accepted-at", default=date.today().isoformat())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = promote(
            args.candidate,
            args.evidence_dir,
            source_bridge_relative=args.source_bridge_relative,
            review=args.review,
            accepted_at=args.accepted_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PROMOTION=FAIL: {type(exc).__name__}: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PROMOTION=PASS")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
