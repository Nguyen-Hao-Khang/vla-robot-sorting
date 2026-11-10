#!/usr/bin/env python3
"""Independent Windows/Linux preboot source lock, dry-run and evidence veto."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_robot_sorting.planning.manipulation_recovery import (
    RecoveryContractError,
    acceptance_readiness,
    audit_strict_campaign,
    create_source_lock,
    verify_source_lock,
    write_new_json,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--write-source-lock", type=Path)
    p.add_argument("--source-lock", type=Path)
    p.add_argument("--readiness-dir", type=Path)
    p.add_argument("--campaign-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        if args.write_source_lock:
            lock = create_source_lock(args.project_root)
            write_new_json(args.write_source_lock, lock)
            print(
                json.dumps(
                    {
                        "source_lock_written": str(args.write_source_lock),
                        "project_git_sha": lock["project_git_sha"],
                        "files": len(lock["files"]),
                    }
                )
            )
            return 0
        if not args.source_lock:
            p.error("--source-lock or --write-source-lock required")
        lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
        result = verify_source_lock(args.project_root, lock)
        if args.campaign_dir:
            audit = audit_strict_campaign(args.campaign_dir, lock["project_git_sha"])
            print(json.dumps(audit, indent=2))
            return 0 if audit["strict_native_campaign_audit_passed"] else 2
        if args.readiness_dir:
            readiness = acceptance_readiness(args.readiness_dir, lock["project_git_sha"])
            print(json.dumps(readiness, indent=2))
            return 0 if readiness["ready"] else 2
        if args.dry_run:
            result.update(
                mode="dry_run_no_native_calls",
                seeds=[0, 1, 2],
                stages=[
                    "pinned_dependencies",
                    "plan_only",
                    "physical_diagnostic_opt_in",
                    "strict_acceptance_requires_three_same_source_physical_audits",
                ],
                maximum_timeout_s=3600,
                automatic_retry=False,
                bundle_created=False,
                native_acceptance=False,
                controller_eligible=False,
            )
        print(json.dumps(result, indent=2))
        return 0
    except (RecoveryContractError, OSError, ValueError) as exc:
        print(
            json.dumps({"source_check": "REJECTED", "error": str(exc), "native_acceptance": False})
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
