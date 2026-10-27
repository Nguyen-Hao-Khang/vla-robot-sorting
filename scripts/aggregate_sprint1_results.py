#!/usr/bin/env python3
"""Aggregate Sprint 1 JSONL scenario records into a reproducible JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_robot_sorting.evaluation.aggregate import load_jsonl, summarize, summary_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.records)
    summary = summary_to_dict(summarize(records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
