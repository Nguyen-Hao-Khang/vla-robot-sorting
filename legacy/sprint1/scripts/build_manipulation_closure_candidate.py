#!/usr/bin/env python3
"""Build a provisional Sprint-1 manipulation closure candidate from native evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vla_robot_sorting.evidence import audit_project_manipulation_probe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _resolved_evidence_file(root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = (root / relative).resolve()
    if path.parent != root:
        raise ValueError(f"{label} must resolve directly inside the result directory")
    return path


def _seed_sources(
    root: Path,
    seed_resolution: Path | None,
) -> dict[int, dict[str, Any]]:
    if seed_resolution is None:
        return {
            seed: {
                "seed": seed,
                "source_label": f"manipulation-seed-{seed}",
                "probe_relative": f"manipulation-seed-{seed}.json",
                "audit_log_relative": f"manipulation-seed-{seed}-audit.log",
                "recovered": False,
                "candidate_id": None,
            }
            for seed in (0, 1, 2)
        }

    resolution_path = seed_resolution.resolve()
    raw = json.loads(resolution_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("seed resolution must be an object")
    if raw.get("schema") != "manipulation-seed-resolution-v1":
        raise ValueError("unsupported manipulation seed resolution schema")
    rows = raw.get("seeds")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("manipulation seed resolution must contain three seed rows")

    resolved: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manipulation seed resolution row must be an object")
        seed = int(row.get("seed"))
        if seed not in (0, 1, 2) or seed in resolved:
            raise ValueError("manipulation seed resolution seeds must be unique 0,1,2")
        probe_relative = row.get("probe_relative")
        audit_relative = row.get("audit_log_relative")
        _resolved_evidence_file(root, probe_relative, label=f"seed {seed} probe")
        _resolved_evidence_file(root, audit_relative, label=f"seed {seed} audit log")
        recovered = row.get("recovered") is True
        candidate_id = row.get("candidate_id")
        if recovered and seed not in (1, 2):
            raise ValueError("only seeds 1 and 2 may use deterministic recovery")
        if recovered and (
            not isinstance(candidate_id, str) or not candidate_id.strip()
        ):
            raise ValueError(
                f"recovered seed {seed} must declare a candidate_id"
            )
        if not recovered and candidate_id is not None:
            raise ValueError("non-recovered seed rows must not declare candidate_id")
        resolved[seed] = {
            "seed": seed,
            "source_label": str(row.get("source_label") or f"seed-{seed}"),
            "probe_relative": probe_relative,
            "audit_log_relative": audit_relative,
            "recovered": recovered,
            "candidate_id": candidate_id,
        }
    if sorted(resolved) != [0, 1, 2]:
        raise ValueError("manipulation seed resolution must cover seeds 0,1,2")
    return resolved

def build_candidate(
    result_dir: Path,
    *,
    linux_git_head: str,
    seed_resolution: Path | None = None,
) -> dict[str, Any]:
    root = result_dir.resolve()
    sources = _seed_sources(root, seed_resolution)
    seeds: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        source = sources[seed]
        probe = _resolved_evidence_file(
            root,
            source["probe_relative"],
            label=f"seed {seed} probe",
        )
        audit_log = _resolved_evidence_file(
            root,
            source["audit_log_relative"],
            label=f"seed {seed} audit log",
        )
        if not probe.is_file():
            raise ValueError(f"missing manipulation probe for seed {seed}: {probe}")
        raw = json.loads(probe.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("passed") is not True:
            raise ValueError(f"seed {seed} probe is not PASS")
        if source["recovered"]:
            selected = ((raw.get("B5") or {}).get("selected") or {}).get(
                "candidate_id"
            )
            if selected != source["candidate_id"]:
                raise ValueError(
                    f"seed {seed} recovery candidate mismatch: "
                    f"{selected!r} != {source['candidate_id']!r}"
                )
            if seed == 2:
                # Seed2 recovery is permitted only for a provably different
                # B5-ranked eligible grasp, never a relabeled original.
                original = _resolved_evidence_file(
                    root, "manipulation-seed-2.json", label="seed 2 original"
                )
                if not original.is_file():
                    raise ValueError("seed 2 recovery requires original evidence")
                original_raw = json.loads(original.read_text(encoding="utf-8"))
                original_id = (
                    ((original_raw.get("B5") or {}).get("selected") or {})
                    .get("candidate_id")
                )
                ranking = (raw.get("B5") or {}).get("ranking") or []
                if (
                    not isinstance(selected, str)
                    or not selected
                    or selected == original_id
                    or not isinstance(ranking, list)
                    or not any(
                        isinstance(item, dict)
                        and item.get("candidate_id") == selected
                        and item.get("eligible") is True
                        for item in ranking
                    )
                ):
                    raise ValueError("seed 2 recovery B5 provenance is invalid")
        audit = audit_project_manipulation_probe(probe)
        if not audit.passed:
            names = [gate.name for gate in audit.failures]
            raise ValueError(f"seed {seed} independent audit failed: {names}")
        if not audit_log.is_file():
            raise ValueError(f"missing independent audit log for seed {seed}")
        audit_text = audit_log.read_text(encoding="utf-8", errors="replace")
        if "AUDIT=PASS" not in audit_text:
            raise ValueError(f"seed {seed} audit log does not report AUDIT=PASS")
        seeds.append(
            {
                "seed": seed,
                "source_label": source["source_label"],
                "recovered": source["recovered"],
                "candidate_id": source["candidate_id"],
                "probe_relative": probe.name,
                "probe_sha256": _sha256(probe),
                "audit_log_relative": audit_log.name,
                "audit_log_sha256": _sha256(audit_log),
                "probe_passed": True,
                "independent_audit_passed": True,
                "final_phase": raw.get("final_phase"),
                "task_success_final": raw.get("task_success_final"),
            }
        )

    return {
        "schema": "native-manipulation-closure-candidate-v1",
        "gate_id": "S1.6-MANIPULATION",
        "classification": "PASS_AUDITED_CANDIDATE",
        "linux_git_head": linux_git_head,
        "result_dir": root.name,
        "fixed_seeds": [0, 1, 2],
        "probe_pass_count": 3,
        "independent_audit_pass_count": 3,
        "seeds": seeds,
        "promotion_required_on_windows": True,
        "claim_boundary": (
            "This candidate proves the native manipulation gate from three fixed-seed "
            "project-owned probes and independent audits. It is not a durable accepted "
            "manifest until the exported evidence is reviewed and promoted on Windows."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--linux-git-head", required=True)
    parser.add_argument("--seed-resolution", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        candidate = build_candidate(
            args.result_dir,
            linux_git_head=args.linux_git_head.strip(),
            seed_resolution=args.seed_resolution,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"CLOSURE_CANDIDATE=FAIL: {type(exc).__name__}: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("CLOSURE_CANDIDATE=PASS")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
