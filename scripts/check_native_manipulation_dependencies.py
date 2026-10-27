#!/usr/bin/env python3
"""Pinned runtime/source preflight; read-only, no task or planner initialization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_robot_sorting.planning.manipulation_recovery import (
    CANONICAL_URDF_SHA256,
    CUROBO_SHA,
    UPSTREAM_SHA,
    RecoveryContractError,
    _git,
    digest,
)
from vla_robot_sorting.planning.native_plan_diagnostics import MPLIB_PLANNER_SHA256


def inspect_git_checkout(root: Path, expected_sha: str) -> dict:
    """Separate wrong revision, tracked source edits and allowlisted runtime residue.

    This is read-only: it never resets, deletes, stashes or hides tracked edits.
    """
    root = root.resolve()
    observed = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tracked_dirty = _git(root, "diff", "--name-only", "HEAD", "--")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    files = [item for item in untracked.splitlines() if item]

    def benign_runtime(name: str) -> bool:
        normalized = name.replace("\\", "/")
        parts = normalized.split("/")
        return (
            ("__pycache__" in parts and normalized.endswith(".pyc"))
            or (
                ".pytest_cache" in parts
                and normalized.endswith(
                    (".gitignore", ".txt", ".json", ".nodeids", ".lastfailed", ".stepwise")
                )
            )
            or (".runtime" in parts and normalized.endswith((".txt", ".log", ".json")))
            or (
                len(parts) > 1
                and parts[0] in ("logs", "output", "outputs")
                and normalized.endswith((".log", ".txt", ".png", ".jpg", ".jpeg", ".csv"))
            )
        )

    unknown = [name for name in files if not benign_runtime(name)]
    return {
        "root": str(root),
        "expected_head": expected_sha,
        "observed_head": observed,
        "head_matches": observed == expected_sha,
        "tracked_modified": tracked_dirty.splitlines() if tracked_dirty else [],
        "untracked_allowlisted": [x for x in files if x not in unknown],
        "untracked_unrecognized": unknown,
        "raw_git_status": status.splitlines() if status else [],
        "source_accepted": observed == expected_sha and not tracked_dirty and not unknown,
    }


def verify_upstream_sources(root: Path, *, require_clean: bool = True) -> dict:
    """Validate pinned RoboTwin blobs/LFS, never bypass tracked source mismatch.

    The diagnostic-only mode may inspect critical bytes even when unrelated
    untracked items remain. Native execution always requires require_clean=True.
    """
    state = inspect_git_checkout(root, UPSTREAM_SHA)
    if not state["head_matches"]:
        raise RecoveryContractError(
            f"RoboTwin wrong revision: expected={UPSTREAM_SHA} observed={state['observed_head']}"
        )
    if require_clean and state["tracked_modified"]:
        raise RecoveryContractError(
            "RoboTwin tracked changes detected: " + repr(state["tracked_modified"][:20])
        )
    if require_clean and state["untracked_unrecognized"]:
        raise RecoveryContractError(
            "RoboTwin unrecognized untracked files: " + repr(state["untracked_unrecognized"][:20])
        )
    prefix = _git(root, "rev-parse", "--show-prefix")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    names = [
        "envs/_base_task.py",
        "envs/robot/robot.py",
        "envs/robot/planner.py",
        "envs/move_pillbottle_pad.py",
        "script/collect_data.py",
        "task_config/move_pillbottle_pad_panthera_single.yml",
        "assets/embodiments/panthera-6dof/panthera_6dof.urdf",
        "assets/embodiments/panthera-6dof/curobo.yml",
        "assets/embodiments/panthera-6dof/collision_panthera.yml",
    ]
    sources = {}
    for name in names:
        blob = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{prefix}{name}"])
        pointer = re.fullmatch(
            rb"version https://git-lfs.github.com/spec/v1\r?\n"
            rb"oid sha256:([0-9a-f]{64})\r?\nsize ([0-9]+)\r?\n?",
            blob,
        )
        p = root / name
        if pointer:
            expected = pointer.group(1).decode()
            if (
                not p.is_file()
                or p.stat().st_size != int(pointer.group(2))
                or digest(p) != expected
            ):
                raise RecoveryContractError(f"missing/incorrect LFS payload: {name}")
        else:
            expected = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
            if (
                not p.is_file()
                or hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected
            ):
                raise RecoveryContractError(f"upstream file differs from Git pin: {name}")
        sources[name] = expected
    if sources[names[6]] != CANONICAL_URDF_SHA256:
        raise RecoveryContractError("canonical URDF lock mismatch")
    return sources


def diagnose_dependencies(robotwin_root: Path) -> int:
    """Read-only comprehensive preflight report; do not stop at the first blocker."""
    report: dict = {
        "schema": "damh.native-dependency-diagnostic.v1",
        "claim": "READ_ONLY_DEPENDENCY_DIAGNOSTIC_NOT_TASK_ACCEPTANCE",
        "python_version": list(sys.version_info[:3]),
        "platform": sys.platform,
        "checks": {},
        "native_acceptance": False,
        "controller_eligible": False,
    }
    checks = report["checks"]

    def attempt(label: str, fn):
        try:
            value = fn()
            checks[label] = {"passed": True, "value": value}
        except Exception as exc:
            checks[label] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    attempt(
        "pinned_python",
        lambda: (
            sys.version.split()[0]
            if sys.platform == "linux" and sys.version_info[:2] == (3, 10)
            else (_ for _ in ()).throw(RecoveryContractError("must use native Linux Python 3.10"))
        ),
    )
    attempt("upstream_git_status", lambda: inspect_git_checkout(robotwin_root, UPSTREAM_SHA))
    # Source status includes candidate violations. Report them as gate failures,
    # rather than hiding them behind a true 'git command succeeded' condition.
    if checks["upstream_git_status"]["passed"]:
        state = checks["upstream_git_status"]["value"]
        checks["upstream_git_status"]["passed"] = bool(state["source_accepted"])
    attempt("upstream_source_blobs_and_lfs", lambda: verify_upstream_sources(robotwin_root))
    attempt(
        "critical_source_digest_diagnostic_only",
        lambda: verify_upstream_sources(robotwin_root, require_clean=False),
    )
    expected = {"mplib": "0.2.1", "sapien": "3.0.0b1", "toppra": "0.6.9"}

    def package_versions():
        found = {}
        for package in expected:
            try:
                found[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                found[package] = "MISSING"
        if found != expected:
            raise RecoveryContractError(f"expected={expected} observed={found}")
        return found

    attempt("pinned_packages", package_versions)

    def mplib_source():
        import mplib.planner

        planner = Path(mplib.planner.__file__).resolve()
        actual = digest(planner)
        if actual != MPLIB_PLANNER_SHA256:
            raise RecoveryContractError(
                f"MPlib planner SHA mismatch: expected={MPLIB_PLANNER_SHA256} actual={actual}"
            )
        return {"path": str(planner), "sha256": actual}

    attempt("mplib_source_sha", mplib_source)

    def curobo_source():
        import curobo

        package = Path(curobo.__file__).resolve()
        source = inspect_git_checkout(package.parent, CUROBO_SHA)
        if not source["source_accepted"]:
            raise RecoveryContractError(f"CuRobo revision/source status: {source}")
        return source

    attempt("curobo_source_git", curobo_source)

    def cuda_status():
        import torch

        if not torch.cuda.is_available():
            raise RecoveryContractError(f"CUDA unavailable: torch={torch.__version__}")
        return {
            "torch_version": str(torch.__version__),
            "cuda_device_name": torch.cuda.get_device_name(0),
        }

    attempt("cuda_runtime", cuda_status)
    report["all_preflight_gates_passed"] = all(v["passed"] for v in checks.values())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_preflight_gates_passed"] else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robotwin-root", type=Path, required=True)
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="collect independent provenance and package diagnostics, never run a native task",
    )
    a = p.parse_args()
    if a.diagnose:
        return diagnose_dependencies(a.robotwin_root)
    try:
        if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
            raise RecoveryContractError(
                "requires native Linux Python 3.10; Windows is offline only"
            )
        versions = {n: importlib.metadata.version(n) for n in ("mplib", "sapien", "toppra")}
        if versions != {"mplib": "0.2.1", "sapien": "3.0.0b1", "toppra": "0.6.9"}:
            raise RecoveryContractError(f"dependency version mismatch: {versions}")
        sources = verify_upstream_sources(a.robotwin_root.resolve())
        import curobo
        import mplib.planner
        import torch

        if digest(Path(mplib.planner.__file__)) != MPLIB_PLANNER_SHA256:
            raise RecoveryContractError("MPlib Python API implementation mismatch")
        curobo_state = inspect_git_checkout(Path(curobo.__file__).parent, CUROBO_SHA)
        if not curobo_state["source_accepted"]:
            raise RecoveryContractError("CuRobo source pin/content mismatch: " + repr(curobo_state))
        if not torch.cuda.is_available():
            raise RecoveryContractError("CUDA unavailable")
        print(
            json.dumps(
                {
                    "dependencies_verified": True,
                    "versions": versions,
                    "source_files": sources,
                    "curobo_git_sha": CUROBO_SHA,
                    "cuda_device": torch.cuda.get_device_name(0),
                    "native_acceptance": False,
                }
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "dependencies_verified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "native_acceptance": False,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
