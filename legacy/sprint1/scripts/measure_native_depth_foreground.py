#!/usr/bin/env python3
# ruff: noqa: E501  # Scoped imported native evidence source; no semantic changes.
"""Generate source-linked read-only native foreground proposal JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_robot_sorting.perception.native_depth_foreground_proposal import (
    SurfaceEvidenceError,
    evaluate_native_depth_foreground,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_native_depth_foreground(args.native_dir)
    except (SurfaceEvidenceError, OSError, ValueError, KeyError) as exc:
        print(f'S1I_DEPTH_FOREGROUND_PROPOSALS_FAIL: {type(exc).__name__}: {exc}')
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print('S1I_DEPTH_FOREGROUND_PROPOSALS_PASS_NON_CONTROL')
    print(f'output={args.output}')
    for cam in report['cameras']:
        print(f"CAMERA={cam['camera_id']} pixels={cam['foreground_pixel_count']} bbox={cam['bbox_xyxy_px']} visible_height_q05q95_m={cam['visible_foreground_vertical_q05_q95_span_m']:.6f}")
    print("source_verified=True instance_identity=False scene_snapshot=False controller_eligible=False")
    print(f"report_sha256={report['report_sha256']}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
