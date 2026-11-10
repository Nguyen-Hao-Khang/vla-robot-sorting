#!/usr/bin/env python3
"""One-session bounded manipulation diagnostics and separately gated acceptance.

No bundle creation, perception, APIs, reboot, promotion or automatic retries.
Every failed process has immutable stdout, stderr, command and exit evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_robot_sorting.planning.manipulation_recovery import (
    UPSTREAM_SHA,
    RecoveryContractError,
    acceptance_readiness,
    audit_strict_campaign,
    audit_terminal_bounds,
    digest,
    plan_only_readiness,
    verify_source_lock,
    write_new_json,
)


def _stop_process(proc):
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=5)


def run_stage(label: str, command: list[str], output: Path, *, timeout_s: float) -> dict:
    if not re.fullmatch(r"[a-z0-9-]+", label):
        raise RecoveryContractError("unsafe stage label")
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 3600:
        raise RecoveryContractError("timeout must be finite and <=3600 seconds")
    targets = [output / f"{label}.{suffix}" for suffix in ("stdout.log", "stderr.log", "exit.json")]
    if any(p.exists() for p in targets):
        raise FileExistsError("refusing stage overwrite")
    start = time.monotonic()
    result = {
        "label": label,
        "command": command,
        "timeout_s": timeout_s,
        "exit_code": 127,
        "timed_out": False,
        "interrupted": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (
        targets[0].open("x", encoding="utf-8") as out,
        targets[1].open("x", encoding="utf-8") as err,
    ):
        try:
            proc = subprocess.Popen(
                command, stdout=out, stderr=err, start_new_session=(os.name == "posix")
            )
            try:
                result["exit_code"] = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                result.update(exit_code=124, timed_out=True)
                _stop_process(proc)
            except KeyboardInterrupt:
                result.update(exit_code=130, interrupted=True)
                _stop_process(proc)
        except OSError as exc:
            result["launch_error"] = repr(exc)
    result["elapsed_s"] = time.monotonic() - start
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_new_json(targets[2], result)
    return result


def known_diagnostic_failure(data: dict, seed: int) -> bool:
    """Only recognize recorded safe stopping points; never authorize a retry."""
    if not isinstance(data, dict) or data.get("seed") != seed:
        return False

    def contains_unknown(value):
        if isinstance(value, list):
            return any(contains_unknown(v) for v in value)
        if isinstance(value, dict):
            if value.get("diagnostic_error") or value.get("diagnostic_exception"):
                return True
            return any(contains_unknown(v) for v in value.values())
        return False

    if contains_unknown(data):
        return False
    stages = data.get("stages")
    if (
        seed in (1, 2)
        and stages == []
        and data.get("error") == "RuntimeError: B5 did not produce a safe selected grasp"
    ):
        b5 = data.get("B5")
        # Planning-only rejection is an observed, non-physical safe stop.
        # Record it and continue other seeds within this one boot, not a retry.
        return (
            isinstance(b5, dict)
            and b5.get("passed") is False
            and b5.get("selected") is None
            and b5.get("eligible_candidate_ids") == []
            and isinstance(b5.get("evaluations"), list)
            and len(b5["evaluations"]) > 0
            and data.get("task_success_final") is None
        )
    if not isinstance(stages, list) or not stages:
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            return False
        contacts = stage.get("contacts")
        policy = stage.get("contact_policy")
        if contacts:
            if not isinstance(contacts, dict) or contacts.get("robot_table_contact_frames") != 0:
                return False
            if stage.get("stage") in ("close_gripper", "release"):
                if (
                    stage.get("passed") is not True
                    or stage.get("terminal_joint_bounds", {}).get("within_bounds") is not True
                ):
                    return False
                continue
            if not isinstance(policy, dict):
                return False
            if (
                not policy.get("allow_robot_target_contact")
                and contacts.get("robot_target_contact_frames") != 0
            ):
                return False
    last = stages[-1]
    if seed == 1:
        return (
            last.get("stage") == "retreat"
            and last.get("plan_success") is False
            and last.get("execution_status") is None
            and data.get("task_success_after_release") is True
            and data.get("post_release_separation_diagnostic", {}).get("passed") is True
            and last.get("failure", {}).get("code") in ("optimization_failed", "ik_unreachable")
        )
    if seed == 2:
        # New strict terminal gate may stop before canonical place, improving evidence.
        return (
            last.get("terminal_joint_bounds", {}).get("within_bounds") is False
            and last.get("passed") is False
            and last.get("execution_status") == "completed"
            and last.get("runtime_contact_policy_pass") is True
        )
    return False


def manifest(output: Path) -> None:
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    if any(p.is_symlink() for p in files):
        raise RecoveryContractError("refusing symlink in evidence")
    data = "".join(f"{digest(p)}  ./{p.relative_to(output).as_posix()}\n" for p in files)
    with (output / "SHA256SUMS.txt").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--robotwin-root", type=Path, required=True)
    p.add_argument("--source-lock", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=("plan-only", "physical-diagnostic", "strict-acceptance"),
        default="plan-only",
    )
    p.add_argument("--readiness-dir", type=Path)
    p.add_argument("--plan-readiness-dir", type=Path)
    p.add_argument("--accept-after-diagnostics", action="store_true")
    p.add_argument("--additional-recovery-diagnostics", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    def stop_on_term(signum, frame):
        raise KeyboardInterrupt("session termination requested")

    signal.signal(signal.SIGTERM, stop_on_term)
    if args.additional_recovery_diagnostics:
        from vla_robot_sorting.planning.recovery_query_diagnostics import validate_diagnostic_mode

        validate_diagnostic_mode(args.mode, args.accept_after_diagnostics)
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if root == output or root in output.parents:
        raise RecoveryContractError("evidence must be outside project checkout")
    if args.accept_after_diagnostics and args.mode != "physical-diagnostic":
        raise RecoveryContractError("accept-after-diagnostics requires physical-diagnostic mode")
    if args.mode == "strict-acceptance" and args.readiness_dir is None:
        raise RecoveryContractError("same-source physical readiness directory required")
    if args.dry_run:
        lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
        verify_source_lock(root, lock)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "seeds": [0, 1, 2],
                    "timeouts_s": {"dependencies": 120, "plan-only": 1800, "physical": 3600},
                    "project_git_sha": lock["project_git_sha"],
                    "no_native_calls": True,
                    "native_acceptance": False,
                    "output_created": False,
                    "automatic_retry": False,
                }
            )
        )
        return 0
    output.mkdir(parents=True, exist_ok=False)
    outcome = {
        "schema": "s1m.final-session.v1",
        "mode": args.mode,
        "project_git_sha": None,
        "stages": [],
        "strict_native_campaign_accepted": False,
        "controller_eligible": False,
        "automatic_retry": False,
        "status": "BLOCKED",
    }
    code = 2
    os.chdir(root)
    os.environ["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root)
    try:
        lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
        verify_source_lock(root, lock)
        outcome["project_git_sha"] = lock["project_git_sha"]
        write_new_json(output / "source-lock.json", lock)
        if sys.platform != "linux":
            raise RecoveryContractError("native session requires Linux; use --dry-run on Windows")
        for name, content in {
            "git-head.txt": lock["project_git_sha"],
            "robotwin-upstream-pin.txt": UPSTREAM_SHA,
            "classification.txt": "PHYSICAL_DIAGNOSTIC_NOT_CAMPAIGN_ACCEPTANCE"
            if args.mode == "physical-diagnostic"
            else "PLAN_ONLY_DIAGNOSTIC_NOT_ACCEPTED"
            if args.mode == "plan-only"
            else "STRICT_CAMPAIGN_REQUIRES_INDEPENDENT_AUDIT",
        }.items():
            with (output / name).open("x", encoding="utf-8", newline="\n") as f:
                f.write(content + "\n")
        dependencies = run_stage(
            "dependencies",
            [
                sys.executable,
                str(root / "scripts/check_native_manipulation_dependencies.py"),
                "--robotwin-root",
                str(args.robotwin_root),
            ],
            output,
            timeout_s=120,
        )
        outcome["stages"].append(dependencies)
        if dependencies["exit_code"] != 0:
            raise RecoveryContractError("dependency/source checkpoint failed; no native task")
        run_stage("gpu-status", ["nvidia-smi"], output, timeout_s=30)
        if args.mode == "plan-only":
            for seed in (0, 1, 2):
                result = run_stage(
                    f"plan-seed-{seed}",
                    [
                        sys.executable,
                        "-u",
                        str(root / "scripts/linux/probe_panthera_plan_only_recovery.py"),
                        "--project-root",
                        str(root),
                        "--robotwin-root",
                        str(args.robotwin_root),
                        "--source-lock",
                        str(args.source_lock.resolve()),
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(output / f"plan-seed-{seed}"),
                    ],
                    output,
                    timeout_s=1800,
                )
                outcome["stages"].append(result)
                if result["exit_code"] != 0:
                    raise RecoveryContractError(
                        f"plan seed {seed} collector failed; preserve and stop"
                    )
            outcome["status"] = "PLAN_ONLY_COLLECTION_COMPLETE_NOT_ACCEPTED"
            code = 0
        else:
            if args.mode == "physical-diagnostic":
                if args.plan_readiness_dir is None:
                    raise RecoveryContractError(
                        "same-source plan-only API checkpoint required before physical diagnostic"
                    )
                plan_ready = plan_only_readiness(args.plan_readiness_dir, lock["project_git_sha"])
                write_new_json(output / "plan-readiness-checkpoint.json", plan_ready)
                if not plan_ready["ready"]:
                    raise RecoveryContractError(
                        "plan-only API/trace checkpoint incomplete; no physical motion"
                    )
            if args.mode == "strict-acceptance":
                ready = acceptance_readiness(args.readiness_dir, lock["project_git_sha"])
                write_new_json(output / "readiness-checkpoint.json", ready)
                if not ready["ready"]:
                    raise RecoveryContractError(
                        "strict acceptance remains blocked by physical evidence"
                    )
            for seed in (0, 1, 2):
                command = [
                    sys.executable,
                    "-u",
                    str(root / "scripts/linux/probe_panthera_project_manipulation.py"),
                    "--project-root",
                    str(root),
                    "--robotwin-root",
                    str(args.robotwin_root),
                    "--output",
                    str(output / f"manipulation-seed-{seed}.json"),
                    "--seed",
                    str(seed),
                    "--plan-diagnostic-dir",
                    str(output / f"plan-traces-seed-{seed}"),
                ]
                if args.additional_recovery_diagnostics:
                    command += [
                        "--additional-diagnostic-dir",
                        str(output / f"recovery-queries-seed-{seed}"),
                    ]
                if seed == 1:
                    command += [
                        "--b5-evaluate-all-height-groups",
                        "--preferred-candidate-id",
                        "pillbottle:cyl:side:0:s-1:up75",
                    ]
                if args.mode == "physical-diagnostic":
                    command += ["--diagnose-native-ik-blockers"]
                result = run_stage(f"manipulation-seed-{seed}", command, output, timeout_s=3600)
                outcome["stages"].append(result)
                if result["timed_out"] or result["interrupted"]:
                    raise RecoveryContractError("native timeout; stop without retry")
                path = output / f"manipulation-seed-{seed}.json"
                audit = run_stage(
                    f"audit-seed-{seed}",
                    [
                        sys.executable,
                        str(root / "scripts/audit_project_manipulation.py"),
                        str(path),
                    ],
                    output,
                    timeout_s=120,
                )
                outcome["stages"].append(audit)
                data = json.loads(path.read_text(encoding="utf-8"))
                bound = audit_terminal_bounds(data)
                write_new_json(output / f"terminal-audit-seed-{seed}.json", bound)
                success = result["exit_code"] == 0 and audit["exit_code"] == 0 and bound["passed"]
                if not success:
                    if (
                        args.mode == "physical-diagnostic"
                        and result["exit_code"] == 2
                        and known_diagnostic_failure(data, seed)
                    ):
                        outcome.setdefault("known_blockers", []).append(seed)
                    else:
                        raise RecoveryContractError(
                            f"seed {seed} failed/unknown; stop without retry"
                        )
            outcome["status"] = (
                "PHYSICAL_DIAGNOSTICS_EXPORTED_NOT_ACCEPTED"
                if args.mode == "physical-diagnostic"
                else "STRICT_RESULTS_REQUIRE_INDEPENDENT_CAMPAIGN_AUDIT"
            )
            code = 0
    except Exception as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write_new_json(output / "session-result.json", outcome)
        manifest(output)
    if code == 0 and args.mode == "strict-acceptance":
        final_audit = audit_strict_campaign(output, lock["project_git_sha"])
        final_audit["evidence_inventory_sha256"] = digest(output / "SHA256SUMS.txt")
        audit_path = output.with_name(output.name + "-campaign-audit.json")
        write_new_json(audit_path, final_audit)
        with audit_path.with_suffix(".json.sha256").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(f"{digest(audit_path)}  {audit_path.name}\n")
        print(json.dumps(final_audit, indent=2))
        return 0 if final_audit["strict_native_campaign_audit_passed"] else 2
    # The optional second checkpoint can open within this SAME Ubuntu session,
    # only if all three physical diagnostics independently passed on this source.
    if code == 0 and args.accept_after_diagnostics:
        ready = acceptance_readiness(output, lock["project_git_sha"])
        print(json.dumps(ready, indent=2))
        if not ready["ready"]:
            print("STRICT ACCEPTANCE BLOCKED; preserve evidence, no automatic rerun.")
            return 2
        return subprocess.call(
            [
                sys.executable,
                __file__,
                "--project-root",
                str(root),
                "--robotwin-root",
                str(args.robotwin_root),
                "--source-lock",
                str(args.source_lock.resolve()),
                "--output-dir",
                str(output.with_name(output.name + "-strict")),
                "--mode",
                "strict-acceptance",
                "--readiness-dir",
                str(output),
            ]
        )
    print(json.dumps(outcome, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
