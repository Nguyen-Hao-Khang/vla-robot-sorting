#!/usr/bin/env python3
"""Statistics and a dataset card (README.md) for a raw recording (plan item 2.1).

Reads only meta.json files written by tools/twin/twin_record.py. The card lists how the
episodes split by language, command form, criterion, object and container counts, the
destination container attributes, the episode lengths and the commands that occur most
often, so the dataset can be checked against docs/task-spec.md before training.

  python tools/data/dataset_card.py data/twin_raw --name panthera_twin_v1 \\
      --out data/twin_raw/README.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load_metas(raw: Path) -> list[dict]:
    metas = []
    for path in sorted(raw.glob("episode_*/meta.json")):
        metas.append(json.loads(path.read_text(encoding="utf-8")))
    return metas


def criterion(meta: dict) -> str:
    ref = meta.get("object_reference") or {}
    keys = list(ref) if isinstance(ref, dict) else [k for k, _v in ref]
    return "+".join(keys) or "unknown"


def destination_attributes(meta: dict) -> dict:
    for c in meta.get("scene", {}).get("containers", []):
        name = f"{c['position']} {c['colour']} {c['depth']} {c['outline']} container"
        if name == meta.get("destination"):
            return c
    return {}


def statistics_of(metas: list[dict]) -> dict:
    ok = [m for m in metas if m.get("success")]
    frames = [m.get("frames", 0) for m in ok]
    fps = ok[0].get("fps", 15) if ok else 15
    dests = [destination_attributes(m) for m in ok]
    return {
        "episodes": len(ok),
        "failed_episodes_kept": len(metas) - len(ok),
        "frames": sum(frames),
        "hours": round(sum(frames) / fps / 3600, 2),
        "frames_per_episode": {
            "min": min(frames) if frames else 0,
            "median": statistics.median(frames) if frames else 0,
            "max": max(frames) if frames else 0,
        },
        "language": dict(Counter(m.get("language", "?") for m in ok)),
        "command_kind": dict(Counter(m.get("command_kind", "?") for m in ok)),
        "object_reference": dict(Counter(criterion(m) for m in ok).most_common()),
        "container_reference": dict(
            Counter((m.get("container_reference") or ["?"])[0] for m in ok).most_common()
        ),
        "objects_in_scene": dict(
            sorted(Counter(len(m.get("scene", {}).get("objects", [])) for m in ok).items())
        ),
        "containers_in_scene": dict(
            sorted(Counter(len(m.get("scene", {}).get("containers", [])) for m in ok).items())
        ),
        "destination_outline": dict(Counter(d.get("outline", "?") for d in dests)),
        "destination_depth": dict(Counter(d.get("depth", "?") for d in dests)),
        "picked_shape": dict(Counter(m.get("object", "? ? ?").split(" ", 2)[-1] for m in ok)),
        "distinct_commands": len({m.get("task") for m in ok}),
        "seeds": sorted({m.get("seed") for m in ok if m.get("seed") is not None}),
    }


def _table(counts: dict, head: str) -> str:
    total = sum(counts.values()) or 1
    lines = [f"| {head} | Episodes | Share |", "|---|---|---|"]
    lines += [f"| {k} | {v} | {v / total:.0%} |" for k, v in counts.items()]
    return "\n".join(lines)


def card(name: str, stats: dict, source_commit: str) -> str:
    f = stats["frames_per_episode"]
    parts = [
        f"# {name}",
        "",
        "Simulated pick-and-place demonstrations of the Panthera-HT arm in RoboTwin for "
        "language-conditioned sorting (task specification v1.2). A scripted expert performs each "
        "command; only successful episodes are kept. Cameras: wrist and workspace, 320x240. "
        "State and action: six joints and the gripper; the action is the next joint-position "
        "target. Rate: 15 Hz.",
        "",
        f"- Episodes: {stats['episodes']} ({stats['frames']} frames, {stats['hours']} h)",
        f"- Frames per episode: min {f['min']}, median {f['median']}, max {f['max']}",
        f"- Distinct commands: {stats['distinct_commands']}",
        f"- Recording seeds: {', '.join(map(str, stats['seeds'])) or 'n/a'}",
        f"- Source commit: {source_commit}",
        "",
        "## Commands",
        "",
        _table(stats["language"], "Language"),
        "",
        _table(stats["command_kind"], "Command form"),
        "",
        _table(stats["object_reference"], "Object reference"),
        "",
        _table(stats["container_reference"], "Container reference"),
        "",
        "## Scenes",
        "",
        _table(stats["objects_in_scene"], "Objects in scene"),
        "",
        _table(stats["containers_in_scene"], "Containers in scene"),
        "",
        _table(stats["destination_outline"], "Destination outline"),
        "",
        _table(stats["destination_depth"], "Destination depth"),
        "",
        _table(stats["picked_shape"], "Picked shape"),
        "",
        "## Intended use",
        "",
        "Fine-tuning and evaluating small vision-language-action policies in simulation. "
        "The held-out test commands and the evaluation scenes (seed 9) are not part of this "
        "dataset.",
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("--name", default="panthera_twin_v1")
    parser.add_argument(
        "--out", type=Path, default=None, help="README.md to write (default: print)"
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    metas = load_metas(args.raw)
    if not metas:
        parser.error(f"no episodes in {args.raw}")
    stats = statistics_of(metas)
    commit_file = args.raw / "SOURCE_COMMIT"
    commit = commit_file.read_text().strip()[:12] if commit_file.is_file() else "unknown"
    text = card(args.name, stats, commit)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.json:
        args.json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
