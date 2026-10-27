"""Pure summaries for repeated native backend probes."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from statistics import median


def latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {"count": 0, "min_s": None, "median_s": None, "p95_s": None, "max_s": None}
    p95_index = min(len(clean) - 1, max(0, round(0.95 * (len(clean) - 1))))
    return {
        "count": len(clean),
        "min_s": clean[0],
        "median_s": median(clean),
        "p95_s": clean[p95_index],
        "max_s": clean[-1],
    }


def memory_summary(samples_mib: Sequence[float], *, warmup_ignore: int = 3) -> dict[str, object]:
    samples = [float(value) for value in samples_mib]
    if not samples:
        return {
            "count": 0,
            "first_mib": None,
            "last_mib": None,
            "min_mib": None,
            "max_mib": None,
            "span_mib": None,
            "tail_span_mib": None,
            "tail_net_growth_mib": None,
            "monotonic_growth_after_warmup": False,
        }
    tail = samples[min(warmup_ignore, len(samples) - 1) :]
    monotonic = (
        len(tail) > 1
        and tail[-1] > tail[0]
        and all(right >= left for left, right in pairwise(tail))
    )
    return {
        "count": len(samples),
        "first_mib": samples[0],
        "last_mib": samples[-1],
        "min_mib": min(samples),
        "max_mib": max(samples),
        "span_mib": max(samples) - min(samples),
        "tail_span_mib": max(tail) - min(tail),
        "tail_net_growth_mib": tail[-1] - tail[0],
        "monotonic_growth_after_warmup": monotonic,
    }
