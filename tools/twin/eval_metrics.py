"""Episode judging and metrics for simulated evaluation (docs/evaluation-protocol.md).

Pure functions without simulator dependencies, so they run and are tested anywhere:
  * ``judge_episode`` applies the success criteria of docs/task-spec.md section 6 to object
    locations before and after an episode;
  * ``failure_phase`` maps an executor step name to a protocol phase;
  * ``wilson`` gives the Wilson 95% interval;
  * ``summarise`` groups JSONL episode records by conditions and reports every metric.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence

Z95 = 1.959963984540054
PHASES = ("approach", "grasp", "lift", "transport", "place")
RATE_METRICS = ("success", "target_selected", "grasped")
# Executor step names (twin_console) -> protocol phase. Checked in order, first match wins.
PHASE_KEYWORDS = (
    ("approach", "approach"),
    ("pre-grasp", "approach"),
    ("grasp", "grasp"),
    ("lift", "lift"),
    ("transit", "transport"),
    ("turn", "transport"),
    ("over", "transport"),
    ("descend", "place"),
    ("release", "place"),
    ("place", "place"),
    ("retreat", "place"),
)


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when n is 0."""
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside 0..{n}")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def failure_phase(step_name: str | None) -> str | None:
    """Protocol phase for an executor step name, or None if it is not recognised."""
    if not step_name:
        return None
    name = step_name.lower()
    for keyword, phase in PHASE_KEYWORDS:
        if keyword in name:
            return phase
    return None


def policy_failure_phase(lifted: Sequence[str], allowed: Iterable[str], placed: bool) -> str:
    """Phase where a learned policy stopped, inferred from what it lifted.

    Nothing lifted: grasp (approach and grasp cannot be told apart without the gripper state).
    A wrong object lifted: grasp. An allowed object lifted but the episode failed: place when
    an allowed object ended in a container (``placed``), otherwise transport.
    """
    allowed = set(allowed)
    if not lifted or lifted[0] not in allowed:
        return "grasp"
    return "place" if placed else "transport"


def judge_episode(
    before: dict[str, str],
    after: dict[str, str],
    allowed: Iterable[str],
    destination: str,
    lifted: Sequence[str],
    safety_stop: bool,
) -> dict:
    """Success criteria of docs/task-spec.md section 6.

    ``before`` and ``after`` map each object to its location ("table", "off table" or a
    container name). ``lifted`` lists objects in the order they first left the table.
    """
    allowed = set(allowed)
    placed = sorted(
        o for o in allowed if after.get(o) == destination and before.get(o) != destination
    )
    containers = {loc for loc in list(before.values()) + list(after.values())} - {
        "table",
        "off table",
    }
    intruders = sorted(
        o
        for o, loc in after.items()
        if o not in placed and loc in containers and before.get(o) != loc
    )
    success = bool(placed) and not intruders and not safety_stop
    return {
        "success": success,
        "placed": placed,
        "intruders": intruders,
        "target_selected": bool(lifted) and lifted[0] in allowed,
        "grasped": bool(lifted),
        "safety_stop": safety_stop,
    }


def _key(record: dict, by: Sequence[str]) -> tuple:
    return tuple(str(record.get(field, "")) for field in by)


def summarise(records: Iterable[dict], by: Sequence[str] = ()) -> list[dict]:
    """One row per condition: n, rates with Wilson intervals, time, interventions, phases."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[_key(record, by)].append(record)
    rows = []
    for key in sorted(groups):
        episodes = groups[key]
        n = len(episodes)
        row: dict = dict(zip(by, key, strict=True))
        row["n"] = n
        for metric in RATE_METRICS:
            k = sum(bool(e.get(metric)) for e in episodes)
            lo, hi = wilson(k, n)
            row[metric] = {"k": k, "rate": k / n, "ci95": [round(lo, 4), round(hi, 4)]}
        times = [e["episode_time_s"] for e in episodes if e.get("episode_time_s") is not None]
        row["episode_time_s"] = {
            "median": round(statistics.median(times), 2) if times else None,
            "n": len(times),
        }
        row["safety_interventions_mean"] = round(
            sum(e.get("safety_interventions", 0) for e in episodes) / n, 3
        )
        phases = [e.get("failure_phase") or "unknown" for e in episodes if not e.get("success")]
        row["failure_phases"] = {p: phases.count(p) for p in (*PHASES, "unknown") if p in phases}
        rows.append(row)
    return rows


def format_table(rows: list[dict], by: Sequence[str]) -> str:
    """Markdown table of the summary rows."""
    head = [*by, "n", "success (95% CI)", "target selected", "grasped", "median time (s)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def cell(metric: dict, n: int) -> str:
        lo, hi = metric["ci95"]
        return f"{metric['k']}/{n} = {metric['rate']:.2f} [{lo:.2f}, {hi:.2f}]"

    for row in rows:
        median = row["episode_time_s"]["median"]
        values = [
            *(str(row[f]) for f in by),
            str(row["n"]),
            cell(row["success"], row["n"]),
            cell(row["target_selected"], row["n"]),
            cell(row["grasped"], row["n"]),
            "n/a" if median is None else f"{median:.1f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
