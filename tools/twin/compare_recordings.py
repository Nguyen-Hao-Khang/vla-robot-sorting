#!/usr/bin/env python3
"""Check that two recordings made with the same seed agree (plan item 1.3).

Episodes are paired by index. For each pair the script compares the task definition stored in
meta.json (command, references, targets, destination, scene, spawn poses, success) exactly,
and the joint-state trajectories within a tolerance, because GPU motion planning may differ
in the last digits between runs.

  python tools/twin/compare_recordings.py data/run_a data/run_b --tol 1e-3

Exit code 0 when every episode matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK_FIELDS = (
    "task",
    "language",
    "command_kind",
    "object_reference",
    "container_reference",
    "targets",
    "object",
    "destination",
    "scene",
    "spawns",
    "success",
)


def episodes(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("episode_*") if (p / "meta.json").is_file())


def compare_episode(a: Path, b: Path, tol: float) -> dict:
    meta_a = json.loads((a / "meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((b / "meta.json").read_text(encoding="utf-8"))
    fields = [f for f in TASK_FIELDS if meta_a.get(f) != meta_b.get(f)]
    states_a, states_b = np.load(a / "states.npy"), np.load(b / "states.npy")
    result = {
        "episode": a.name,
        "task_fields_differ": fields,
        "frames": [len(states_a), len(states_b)],
    }
    if states_a.shape == states_b.shape and states_a.size:
        result["max_state_diff"] = float(np.max(np.abs(states_a - states_b)))
    else:
        result["max_state_diff"] = None
    result["task_identical"] = not fields
    result["trajectory_close"] = (
        result["max_state_diff"] is not None and result["max_state_diff"] <= tol
    )
    return result


def compare(root_a: Path, root_b: Path, tol: float) -> tuple[list[dict], list[str]]:
    eps_a, eps_b = episodes(root_a), episodes(root_b)
    problems = []
    if len(eps_a) != len(eps_b):
        problems.append(f"episode count differs: {len(eps_a)} vs {len(eps_b)}")
    results = [compare_episode(a, b, tol) for a, b in zip(eps_a, eps_b, strict=False)]
    return results, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--tol", type=float, default=1e-3, help="max joint difference (rad)")
    parser.add_argument("--json", type=Path, default=None, help="write per-episode results")
    args = parser.parse_args(argv)
    results, problems = compare(args.run_a, args.run_b, args.tol)
    task_ok = sum(r["task_identical"] for r in results)
    traj_ok = sum(r["trajectory_close"] for r in results)
    print(f"episodes compared: {len(results)}")
    print(f"identical task definitions: {task_ok}/{len(results)}")
    print(f"trajectories within {args.tol} rad: {traj_ok}/{len(results)}")
    for r in results:
        if not (r["task_identical"] and r["trajectory_close"]):
            print(
                f"  {r['episode']}: fields {r['task_fields_differ']}, frames {r['frames']}, "
                f"max diff {r['max_state_diff']}"
            )
    for p in problems:
        print(f"  {p}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    ok = not problems and task_ok == traj_ok == len(results) and results
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
