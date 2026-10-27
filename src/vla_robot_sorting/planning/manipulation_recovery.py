"""Exact bound diagnostics and source-locked native recovery checkpoints.

Integrity checks do not prove physical safety. No plan-only record authorizes motion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from vla_robot_sorting.contracts import JointLimits, JointState
from vla_robot_sorting.planning.approach_safety_audit import verify_checksum_inventory

UPSTREAM_SHA = "2aee31ad75cb7b0cd779540441c6025853a5b60d"
CUROBO_SHA = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
TASK_CONFIG = "move_pillbottle_pad_panthera_single"


class RecoveryContractError(ValueError):
    """Required source or safety evidence is missing or inconsistent."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new_json(path: str | Path, record: Any) -> None:
    """Serialize first so malformed/nonfinite input leaves no partial artifact."""
    data = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)


def joint_bound_report(state: JointState, limits: JointLimits) -> dict[str, Any]:
    """Strict declared bounds. Never clip, round, widen or correct measured state."""
    if state.joint_names != limits.joint_names:
        raise RecoveryContractError("joint order does not match limits")
    from vla_robot_sorting.planning.native_plan_diagnostics import finite_vector

    finite_vector(list(state.positions), len(limits.joint_names))
    rows = []
    for name, q, lower, upper in zip(
        state.joint_names,
        state.positions,
        limits.lower_position_rad,
        limits.upper_position_rad,
        strict=True,
    ):
        margin = min(q - lower, upper - q)
        rows.append(
            {
                "joint": name,
                "measured_rad": q,
                "lower_rad": lower,
                "upper_rad": upper,
                "margin_rad": margin,
                "excess_rad": max(0.0, -margin),
            }
        )
    return {
        "within_bounds": all(row["margin_rad"] >= 0.0 for row in rows),
        "minimum_margin_rad": min(row["margin_rad"] for row in rows),
        "joints": rows,
        "state_modified": False,
    }


def _git(root: Path, *args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
            .decode("utf-8")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryContractError("Git source check failed") from exc


def _relative(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise RecoveryContractError("unsafe manifest path")
    p = PurePosixPath(name)
    if p.is_absolute() or p.as_posix() != name or any(x in (".", "..", "") for x in p.parts):
        raise RecoveryContractError("unsafe manifest path")
    return name


def _require_clean_source_checkout(root: Path) -> None:
    """Reject all dirty checkouts except byte-equivalent LF/CRLF text conversions.

    Git status can mark an LF-indexed text file modified on Linux even when
    the working bytes differ only by CRLF. This exception is checked against
    the committed HEAD blob, not against an untrusted source-lock manifest.
    """
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "-z"],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryContractError("Git source check failed") from exc
    if not raw:
        return
    modes = _tracked_modes(root)
    for row in raw.decode("utf-8").split("\0"):
        if not row:
            continue
        # Only an unstaged edit to a tracked LF-indexed file can be EOL-only.
        # Reject untracked, staged, deleted, renamed and unmerged entries.
        if not row.startswith(" M "):
            raise RecoveryContractError("source checkout is not clean")
        name = _relative(row[3:])
        path = root / name
        if (
            modes.get(name) != "lf_text"
            or path.is_symlink()
            or not path.is_file()
            or root not in path.resolve().parents
        ):
            raise RecoveryContractError("source checkout is not clean")
        try:
            head_blob = subprocess.check_output(
                ["git", "-C", str(root), "show", f"HEAD:{name}"],
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RecoveryContractError("Git source check failed") from exc
        if path.read_bytes().replace(b"\r\n", b"\n") != head_blob:
            raise RecoveryContractError("source checkout is not clean")
        # Matching content does not authorize a working-tree mode change.
        if _git(root, "diff", "--summary", "HEAD", "--", name):
            raise RecoveryContractError("source checkout is not clean")


def create_source_lock(root: str | Path) -> dict[str, Any]:
    """Lock ALL tracked files of a clean committed checkout; no bundle creation."""
    root = Path(root).resolve()
    _require_clean_source_checkout(root)
    head = _git(root, "rev-parse", "HEAD")
    files = {}
    modes = _tracked_modes(root)
    for name in modes:
        _relative(name)
        p = (root / name).resolve()
        if root not in p.parents or not p.is_file() or (root / name).is_symlink():
            raise RecoveryContractError("source manifest contains missing/unsafe file")
        files[name] = _source_digest(p, modes[name])
    if not files:
        raise RecoveryContractError("empty source checkout")
    return {
        "schema": "s1m.final-source-lock.v1",
        "project_git_sha": head,
        "robotwin_git_sha": UPSTREAM_SHA,
        "curobo_git_sha": CUROBO_SHA,
        "task_config": TASK_CONFIG,
        "files": files,
        "eol_modes": modes,
    }


def verify_source_lock(root: str | Path, lock: Any) -> dict[str, Any]:
    root = Path(root).resolve()
    if not isinstance(lock, dict) or lock.get("schema") != "s1m.final-source-lock.v1":
        raise RecoveryContractError("invalid source lock schema")
    if lock.get("robotwin_git_sha") != UPSTREAM_SHA or lock.get("curobo_git_sha") != CUROBO_SHA:
        raise RecoveryContractError("native source pin mismatch")
    if lock.get("task_config") != TASK_CONFIG:
        raise RecoveryContractError("canonical task mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(lock.get("project_git_sha", ""))):
        raise RecoveryContractError("invalid project SHA")
    if _git(root, "rev-parse", "HEAD") != lock["project_git_sha"]:
        raise RecoveryContractError("project HEAD mismatch")
    _require_clean_source_checkout(root)
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise RecoveryContractError("missing source inventory")
    modes = _tracked_modes(root)
    if lock.get("eol_modes") != modes:
        raise RecoveryContractError("source text/binary modes mismatch")
    if set(modes) != set(files):
        raise RecoveryContractError("source inventory is incomplete or has extra paths")
    for name, expected in files.items():
        _relative(name)
        p = (root / name).resolve()
        if root not in p.parents or not p.is_file() or (root / name).is_symlink():
            raise RecoveryContractError("unsafe or missing source path")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RecoveryContractError("invalid source digest")
        if _source_digest(p, modes[name]) != expected:
            raise RecoveryContractError(f"source checksum mismatch: {name}")
    return {"verified_file_count": len(files), "project_git_sha": lock["project_git_sha"]}


def _three_seed_readiness(
    evidence_dir: str | Path, source_sha: str, classification: str
) -> dict[str, Any]:
    """Require three independently audited physical diagnostics on this exact source.

    Plan-only/FK/FCL checks, historical seed0 repeats and RGB-D never count.
    This is a veto gate, not authentication of arbitrary self-declared native JSON.
    """
    from vla_robot_sorting.evidence import audit_project_manipulation_probe
    from vla_robot_sorting.planning.native_plan_diagnostics import audit_native_plan_trace

    root = Path(evidence_dir).resolve()
    accepted = []
    errors = []
    try:
        hashes = verify_checksum_inventory(root)
        if (root / "git-head.txt").read_text(encoding="utf-8").strip() != source_sha:
            raise RecoveryContractError("diagnostic source SHA mismatch")
        if (root / "robotwin-upstream-pin.txt").read_text(encoding="utf-8").strip() != UPSTREAM_SHA:
            raise RecoveryContractError("diagnostic RoboTwin SHA mismatch")
        if (root / "classification.txt").read_text(encoding="utf-8").strip() != classification:
            raise RecoveryContractError("required physical evidence classification mismatch")
        for required in ("git-head.txt", "robotwin-upstream-pin.txt", "classification.txt"):
            if required not in hashes:
                raise RecoveryContractError("metadata not bound to checksum inventory")
        for seed in (0, 1, 2):
            name = f"manipulation-seed-{seed}.json"
            if name not in hashes:
                errors.append(f"seed{seed}: missing checksum-bound physical record")
                continue
            data = json.loads((root / name).read_text(encoding="utf-8"))
            if (
                not isinstance(data, dict)
                or data.get("seed") != seed
                or type(data.get("seed")) is not int
            ):
                errors.append(f"seed{seed}: wrong seed record")
                continue
            if (
                data.get("task_config") != TASK_CONFIG
                or data.get("upstream_manipulation_macro_used") is not False
            ):
                errors.append(f"seed{seed}: canonical task/source mismatch")
                continue
            result = audit_project_manipulation_probe(root / name)
            if not result.passed:
                errors.append(f"seed{seed}: independent manipulation audit failed")
                continue
            if not audit_terminal_bounds(data)["passed"]:
                errors.append(f"seed{seed}: independent strict terminal bounds failed")
                continue
            if not audit_runtime_checkpoints(data)["passed"]:
                errors.append(f"seed{seed}: runtime measured checkpoint incomplete")
                continue
            trace_names = sorted(
                n
                for n in hashes
                if n.startswith(f"plan-traces-seed-{seed}/") and n.endswith(".json")
            )
            if not trace_names or any(
                not audit_native_plan_trace(json.loads((root / n).read_text(encoding="utf-8")))[
                    "passed"
                ]
                for n in trace_names
            ):
                errors.append(f"seed{seed}: missing/invalid native plan traces")
                continue
            # New boundary evidence is mandatory; historical records do not contain it.
            motion = [s for s in data.get("stages", []) if s.get("plan_success") is True]
            if not motion or any(
                s.get("terminal_joint_bounds", {}).get("within_bounds") is not True for s in motion
            ):
                errors.append(f"seed{seed}: missing strict terminal-bound gate")
                continue
            accepted.append(seed)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "ready": accepted == [0, 1, 2] and not errors,
        "accepted_seed_count": len(accepted),
        "accepted_seeds": accepted,
        "errors": errors,
        "controller_eligible": False,
        "claim": "prerequisite veto only; strict campaign and audits still required",
    }


def acceptance_readiness(evidence_dir: str | Path, source_sha: str) -> dict[str, Any]:
    return _three_seed_readiness(
        evidence_dir, source_sha, "PHYSICAL_DIAGNOSTIC_NOT_CAMPAIGN_ACCEPTANCE"
    )


def audit_strict_campaign(evidence_dir: str | Path, source_sha: str) -> dict[str, Any]:
    result = _three_seed_readiness(
        evidence_dir, source_sha, "STRICT_CAMPAIGN_REQUIRES_INDEPENDENT_AUDIT"
    )
    result["strict_native_campaign_audit_passed"] = result.pop("ready")
    result["claim"] = (
        "three source-bound native seed records audited; no promotion/controller authorization"
    )
    return result


# Extracted from the pinned URDF LFS object; never planner tolerances.
CANONICAL_URDF_SHA256 = "1a3ca789a350aa5084c58f1c7ff985bb46a08064656178dab29c05b0288fd4f9"
CANONICAL_LOWER_RAD = (-2.4, 0.0, 0.0, -1.6, -1.7, -2.5)
CANONICAL_UPPER_RAD = (2.4, 3.2, 4.0, 1.6, 1.7, 2.5)


def audit_terminal_bounds(data: Any) -> dict[str, Any]:
    """Recompute every executed motion's qpos against the pinned URDF bounds."""
    from vla_robot_sorting.planning.native_plan_diagnostics import finite_vector

    checked = 0
    errors = []

    def walk(value: Any) -> None:
        nonlocal checked
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if "stage" in value and value.get("execution_status") == "completed":
                checked += 1
                try:
                    qpos = finite_vector(value.get("qpos_after_rad"), 6)
                    if any(
                        q < lo or q > hi
                        for q, lo, hi in zip(
                            qpos, CANONICAL_LOWER_RAD, CANONICAL_UPPER_RAD, strict=True
                        )
                    ):
                        errors.append("measured terminal qpos outside pinned URDF bounds")
                    if value.get("terminal_joint_bounds", {}).get("within_bounds") is not True:
                        errors.append("missing terminal bounds record")
                except (ValueError, TypeError, AttributeError) as exc:
                    errors.append(str(exc))
            for child in value.values():
                walk(child)

    walk(data)
    return {
        "passed": checked > 0 and not errors,
        "motion_records_checked": checked,
        "errors": errors,
        "urdf_sha256": CANONICAL_URDF_SHA256,
        "controller_eligible": False,
    }


def _tracked_modes(root: Path) -> dict[str, str]:
    # -z keeps Unicode, spaces and quoting independent of core.quotepath.
    result = {}
    for row in _git(root, "ls-files", "--eol", "-z").split("\0"):
        if not row:
            continue
        header, name = row.split("\t", 1)
        _relative(name)
        result[name] = "lf_text" if header.split()[0] == "i/lf" else "raw"
    return result


def _source_digest(path: Path, mode: str) -> str:
    data = path.read_bytes()
    if mode == "lf_text":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def audit_runtime_checkpoints(data: Any) -> dict[str, Any]:
    """Require measured per-step stops in all executed nested motion records."""
    checked, errors = 0, []

    def walk(value: Any) -> None:
        nonlocal checked
        if isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, dict):
            if "stage" in value and value.get("execution_status") == "completed":
                checked += 1
                checkpoint = value.get("runtime_safety_checkpoint")
                contacts = value.get("contacts")
                if not isinstance(checkpoint, dict) or not isinstance(contacts, dict):
                    errors.append("missing runtime safety checkpoint")
                elif (
                    type(checkpoint.get("joint_checks")) is not int
                    or checkpoint["joint_checks"] <= 0
                    or checkpoint["joint_checks"] != contacts.get("physics_steps")
                    or checkpoint.get("first_failure") is not None
                    or not isinstance(checkpoint.get("minimum_margin_rad"), (int, float))
                    or isinstance(checkpoint["minimum_margin_rad"], bool)
                    or not math.isfinite(checkpoint["minimum_margin_rad"])
                    or checkpoint["minimum_margin_rad"] < 0
                ):
                    errors.append("incomplete/failed measured runtime checkpoint")
            for child in value.values():
                walk(child)

    walk(data)
    return {
        "passed": checked > 0 and not errors,
        "motion_records_checked": checked,
        "errors": errors,
        "controller_eligible": False,
    }


def plan_only_readiness(evidence_dir: str | Path, source_sha: str) -> dict[str, Any]:
    """API/collection prerequisite only; never a physical safety certification."""
    from vla_robot_sorting.planning.native_plan_diagnostics import audit_native_plan_trace

    root = Path(evidence_dir).resolve()
    checked, errors = [], []
    try:
        hashes = verify_checksum_inventory(root)
        required = {
            "git-head.txt": source_sha,
            "robotwin-upstream-pin.txt": UPSTREAM_SHA,
            "classification.txt": "PLAN_ONLY_DIAGNOSTIC_NOT_ACCEPTED",
        }
        for name, value in required.items():
            if name not in hashes or (root / name).read_text(encoding="utf-8").strip() != value:
                raise RecoveryContractError("plan-only source/classification checkpoint mismatch")
        for seed in (0, 1, 2):
            name = f"plan-seed-{seed}/diagnostic.json"
            if name not in hashes:
                raise RecoveryContractError("missing checksum-bound plan-only diagnostic")
            record = json.loads((root / name).read_text(encoding="utf-8"))
            if (
                record.get("schema") != "s1m.initial-plan-diagnostic.v1"
                or type(record.get("seed")) is not int
                or record["seed"] != seed
                or record.get("project_git_sha") != source_sha
                or record.get("robotwin_git_sha") != UPSTREAM_SHA
                or record.get("task_config") != TASK_CONFIG
                or record.get("diagnostic_complete") is not True
                or record.get("executed") is not False
                or record.get("native_acceptance") is not False
            ):
                raise RecoveryContractError("incomplete/wrong-source initial diagnostic")
            names = [
                n
                for n in hashes
                if n.startswith(f"plan-seed-{seed}/candidate-") and n.endswith(".json")
            ]
            if not names or any(
                not audit_native_plan_trace(json.loads((root / n).read_text(encoding="utf-8")))[
                    "passed"
                ]
                for n in names
            ):
                raise RecoveryContractError("missing/invalid native initial plan traces")
            checked.append(seed)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "ready": checked == [0, 1, 2] and not errors,
        "checked_seeds": checked,
        "errors": errors,
        "controller_eligible": False,
        "claim": "native API/trace prerequisite only; physical motion gates remain mandatory",
    }
