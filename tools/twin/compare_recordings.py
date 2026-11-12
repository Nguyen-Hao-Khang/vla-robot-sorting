#!/usr/bin/env python3
"""Check that two recordings made with the same seed agree (plan item 1.3).

Episodes are paired by ``episode_key`` (layout and command index, written by twin_record.py)
or, for older recordings, by position. A failed episode ends its layout and is not saved, so
a key present in only one run is reported, not paired.

For each pair the task definition stored in meta.json (command, references, targets,
destination, scene, spawn poses, success) must be identical. The joint trajectories are
compared after resampling both to the same normalised time; the script reports the largest
joint difference and the difference in length, because GPU motion planning and contact
physics are not bit-exact between runs.

  python tools/twin/compare_recordings.py data/run_a data/run_b

Exit code 0 when every common episode has an identical task definition and at least 90% of
the episodes of each run are paired; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import statistics
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
MIN_PAIRED = 0.9


def episodes(root: Path) -> dict[str, Path]:
    folders = sorted(p for p in root.glob("episode_*") if (p / "meta.json").is_file())
    out = {}
    for i, folder in enumerate(folders):
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        out[str(meta.get("episode_key", f"#{i}"))] = folder
    return out


def resample(states: np.ndarray, length: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(states))
    dst = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(dst, src, states[:, j]) for j in range(states.shape[1])], axis=1)


def compare_episode(key: str, a: Path, b: Path) -> dict:
    meta_a = json.loads((a / "meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((b / "meta.json").read_text(encoding="utf-8"))
    fields = [f for f in TASK_FIELDS if meta_a.get(f) != meta_b.get(f)]
    states_a, states_b = np.load(a / "states.npy"), np.load(b / "states.npy")
    diff = None
    if states_a.ndim == states_b.ndim == 2 and len(states_a) > 1 and len(states_b) > 1:
        n = max(len(states_a), len(states_b))
        diff = float(np.max(np.abs(resample(states_a, n) - resample(states_b, n))))
    return {
        "episode_key": key,
        "task_fields_differ": fields,
        "task_identical": not fields,
        "frames": [len(states_a), len(states_b)],
        "max_state_diff": diff,
    }


def compare(root_a: Path, root_b: Path) -> tuple[list[dict], dict]:
    eps_a, eps_b = episodes(root_a), episodes(root_b)
    common = [k for k in eps_a if k in eps_b]
    results = [compare_episode(k, eps_a[k], eps_b[k]) for k in common]
    extra = {
        "only_in_a": [k for k in eps_a if k not in eps_b],
        "only_in_b": [k for k in eps_b if k not in eps_a],
        "count_a": len(eps_a),
        "count_b": len(eps_b),
    }
    return results, extra


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="write per-episode results")
    args = parser.parse_args(argv)
    results, extra = compare(args.run_a, args.run_b)
    n = len(results)
    identical = sum(r["task_identical"] for r in results)
    diffs = [r["max_state_diff"] for r in results if r["max_state_diff"] is not None]
    lengths = [abs(r["frames"][0] - r["frames"][1]) for r in results]
    print(f"episodes: {extra['count_a']} vs {extra['count_b']}, paired: {n}")
    print(f"identical task definitions: {identical}/{n}")
    if diffs:
        print(
            f"max joint difference (rad, time-normalised): median {statistics.median(diffs):.4f}, "
            f"max {max(diffs):.4f}"
        )
        print(
            f"length difference (frames): median {statistics.median(lengths)}, max {max(lengths)}"
        )
    for r in results:
        if not r["task_identical"]:
            print(f"  {r['episode_key']}: fields {r['task_fields_differ']}")
    for side in ("only_in_a", "only_in_b"):
        if extra[side]:
            print(f"  {side}: {', '.join(extra[side])}")
    if args.json:
        args.json.write_text(json.dumps({"episodes": results, **extra}, indent=2) + "\n", "utf-8")
    paired_ok = n >= MIN_PAIRED * max(extra["count_a"], extra["count_b"], 1)
    return 0 if n and identical == n and paired_ok else 1


if __name__ == "__main__":
    sys.exit(main())
