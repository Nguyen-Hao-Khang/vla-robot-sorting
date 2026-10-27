"""Resource-telemetry parsing shared by native evaluation harnesses."""

from __future__ import annotations

import csv
import re
from pathlib import Path


def _number(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"No numeric value in {value!r}")
    return float(match.group(0))


def summarize_resource_usage(gpu_csv: Path, time_file: Path) -> dict[str, object]:
    rows: list[list[str]] = []
    with gpu_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 6:
                rows.append(row)
    if not rows:
        raise ValueError("GPU telemetry has no samples")

    time_text = time_file.read_text(encoding="utf-8", errors="replace")
    rss_match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        time_text,
    )
    elapsed_wall_clock = None
    exit_status = None
    for line in time_text.splitlines():
        if "Elapsed (wall clock) time" in line:
            elapsed_wall_clock = line.rsplit(": ", 1)[-1].strip()
        elif "Exit status:" in line:
            exit_status = int(line.rsplit(":", 1)[-1].strip())

    return {
        "gpu_samples": len(rows),
        "max_gpu_utilization_percent": max(_number(row[2]) for row in rows),
        "max_gpu_memory_utilization_percent": max(_number(row[3]) for row in rows),
        "peak_vram_used_mib": max(_number(row[4]) for row in rows),
        "vram_total_mib": max(_number(row[5]) for row in rows),
        "max_host_rss_kib": int(rss_match.group(1)) if rss_match else None,
        "elapsed_wall_clock": elapsed_wall_clock,
        "command_exit_status": exit_status,
    }
