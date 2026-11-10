#!/usr/bin/env python3
"""Create a JSON resource summary from native telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_robot_sorting.evaluation.resource_usage import summarize_resource_usage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--time-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_resource_usage(args.gpu_csv, args.time_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
