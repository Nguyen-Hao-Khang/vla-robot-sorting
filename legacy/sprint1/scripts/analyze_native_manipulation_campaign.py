"""Read-only bounded native manipulation campaign diagnostics.

This script never grants native acceptance or changes the campaign's seed resolution.
It reports exactly what immutable stage evidence says, marking missing records as unknown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SEED_LABELS = (
    "manipulation-seed-0",
    "manipulation-seed-1",
    "manipulation-seed-2",
    "manipulation-seed-1-expanded",
    "manipulation-seed-1-recovery-1",
    "manipulation-seed-1-recovery-2",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _stage(probe: dict[str, Any], name: str) -> dict[str, Any] | None:
    stages = probe.get("stages")
    if not isinstance(stages, list):
        return None
    return next(
        (
            stage
            for stage in stages
            if isinstance(stage, dict) and stage.get("stage") == name
        ),
        None,
    )


def _dictionary(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _failure(stage: dict[str, Any] | None) -> dict[str, Any]:
    failure = _dictionary(_dictionary(stage).get("failure"))
    details = _dictionary(failure.get("details"))
    return {
        "code": failure.get("code"),
        "native_status": details.get("native_status"),
        "native_valid_query": _dictionary(details.get("native_diagnostics")).get(
            "valid_query"
        ),
    }


def _contacts(stage: dict[str, Any] | None) -> dict[str, Any]:
    contacts = _dictionary(_dictionary(stage).get("contacts"))
    return {
        key: contacts.get(key)
        for key in (
            "physics_steps",
            "robot_target_contact_frames",
            "robot_target_finger_contact_frames",
            "robot_target_other_link_contact_frames",
            "first_robot_target_step",
            "last_robot_target_step",
            "target_pad_contact_frames",
            "target_table_contact_frames",
            "robot_table_contact_frames",
        )
    }


def _place(stage: dict[str, Any] | None) -> dict[str, Any]:
    place = _dictionary(stage)
    recovery = _dictionary(place.get("placement_recovery"))
    vertical = _dictionary(recovery.get("vertical_reachability_diagnostic"))
    staged = _dictionary(recovery.get("staged_descent"))
    yaw = recovery.get("attempts")
    heights = vertical.get("attempts")
    return {
        "passed": place.get("passed"),
        "failure": _failure(stage),
        "yaw": [
            {
                "offset_rad": row.get("yaw_offset_rad"),
                "plan_success": row.get("plan_success"),
                "native_status": row.get("native_status"),
                "executed": row.get("execution_status") is not None,
            }
            for row in yaw
            if isinstance(row, dict)
        ]
        if isinstance(yaw, list)
        else None,
        "staged_descent": {
            "strategy": staged.get("strategy"),
            "extra_height_m": staged.get("extra_height_m"),
            "elevated_passed": _dictionary(staged.get("elevated_motion")).get("passed"),
            "canonical_attempted": staged.get("canonical_attempted"),
            "canonical_motion_passed": staged.get("canonical_motion_passed"),
        }
        if staged
        else None,
        "height_diagnostic": {
            "planning_only": vertical.get("planning_only"),
            "executed": vertical.get("executed"),
            "acceptance_override": vertical.get("acceptance_override"),
            "attempts": [
                {
                    "extra_height_m": row.get("extra_height_m"),
                    "plan_success": row.get("plan_success"),
                    "native_status": row.get("native_status"),
                    "desired_gripper_xyz_m": row.get("desired_gripper_xyz_m"),
                }
                for row in heights
                if isinstance(row, dict)
            ]
            if isinstance(heights, list)
            else None,
        }
        if vertical
        else None,
    }


def _seed_summary(label: str, probe: dict[str, Any] | None) -> dict[str, Any]:
    if probe is None:
        return {"source_label": label, "present": False, "passed": None}
    stages = probe.get("stages")
    stages = stages if isinstance(stages, list) else []
    retreat = _stage(probe, "retreat")
    return {
        "source_label": label,
        "present": True,
        "passed": probe.get("passed"),
        "error": probe.get("error"),
        "selected_grasp": _dictionary(_dictionary(probe.get("B5")).get("selected")).get(
            "candidate_id"
        ),
        "stages": [
            {"name": row.get("stage"), "passed": row.get("passed")}
            for row in stages
            if isinstance(row, dict)
        ],
        "task_success_after_release": probe.get("task_success_after_release"),
        "task_success_final": probe.get("task_success_final"),
        "place": _place(_stage(probe, "place_above_support")),
        "separation": {
            "direction_strategy": _dictionary(
                _dictionary(
                    _dictionary(retreat).get("post_release_separation")
                ).get("separation_geometry")
            ).get("strategy"),
            "passed": _dictionary(
                _dictionary(retreat).get("post_release_separation")
            ).get("passed"),
            "finger_contacts_after": _dictionary(
                _dictionary(retreat).get("post_release_separation")
            ).get("finger_contacts_after"),
        },
        "retreat": {
            "direction_strategy": _dictionary(
                _dictionary(retreat).get("retreat_geometry")
            ).get("strategy"),
            "passed": _dictionary(retreat).get("passed"),
            "contact_policy": _dictionary(retreat).get("contact_policy"),
            "contacts": _contacts(retreat),
        },
    }


def analyze(result_dir: Path) -> dict[str, Any]:
    campaign_path = result_dir / "CAMPAIGN_RESULT.txt"
    campaign: dict[str, str] = {}
    if campaign_path.is_file():
        for line in campaign_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                campaign[key] = value
    followup = _load_json(result_dir / "FOLLOWUP_DECISION.json")
    return {
        "schema": "native-manipulation-diagnostic-digest-v1",
        "source_dir": result_dir.name,
        "campaign_result_present": campaign_path.is_file(),
        "campaign_reported": {
            "manip_pass_count": campaign.get("MANIP_PASS_COUNT"),
            "manip_audit_pass_count": campaign.get("MANIP_AUDIT_PASS_COUNT"),
            "accepted": campaign.get("CAMPAIGN_ACCEPTED"),
            "first_blocker": campaign.get("FIRST_BLOCKER"),
            "seed1_effective_label": campaign.get("SEED1_EFFECTIVE_LABEL"),
        },
        "claim_boundary": (
            "Diagnostic-only: no native PASS or acceptance inferred from this digest. "
            "Use immutable SHA256 evidence, audited seeds, resolution and Windows "
            "promotion for any closure claim."
        ),
        "followup_primary_decision": (
            followup.get("primary_decision") if followup is not None else None
        ),
        "seeds": {
            label: _seed_summary(label, _load_json(result_dir / f"{label}.json"))
            for label in SEED_LABELS
        },
    }


def as_text(result: dict[str, Any]) -> str:
    reported = result["campaign_reported"]
    lines = [
        "NATIVE MANIPULATION DIAGNOSTIC DIGEST (not acceptance)",
        f"Source: {result['source_dir']}",
        f"Campaign result available: {result['campaign_result_present']}",
        (
            "Campaign reported: pass="
            f"{reported['manip_pass_count']}/3 "
            f"audit={reported['manip_audit_pass_count']}/3 "
            f"accepted={reported['accepted']}"
        ),
        f"Primary followup: {result['followup_primary_decision']}",
        "",
    ]
    for label, seed in result["seeds"].items():
        lines.append(
            f"{label}: present={seed['present']} passed={seed['passed']}"
        )
        if not seed["present"]:
            continue
        lines.append(
            f"  release oracle={seed['task_success_after_release']} "
            f"retreat={seed['retreat']['passed']} "
            f"robot-target contacts={seed['retreat']['contacts']['robot_target_contact_frames']}"
        )
        contacts = seed["retreat"]["contacts"]
        lines.append(
            f"  retreat direction={seed['retreat']['direction_strategy']}"
        )
        lines.append(
            f"  contact link attribution: finger={contacts['robot_target_finger_contact_frames']}"
            f" other={contacts['robot_target_other_link_contact_frames']}"
            f" first={contacts['first_robot_target_step']}"
            f" last={contacts['last_robot_target_step']}"
        )
        place = seed["place"]
        lines.append(
            f"  placement={place['passed']} native={place['failure']['native_status']}"
        )
        lines.append(
            f"  separation direction={seed['separation']['direction_strategy']} "
            f"passed={seed['separation']['passed']}"
        )
        if place["staged_descent"]:
            staged = place["staged_descent"]
            lines.append(
                f"  staged descent={staged['strategy']} "
                f"waypoint_pass={staged['elevated_passed']} "
                f"canonical_pass={staged['canonical_motion_passed']}"
            )
        diagnostic = place["height_diagnostic"]
        if diagnostic:
            lines.append(
                f"  height diagnostic: planning_only={diagnostic['planning_only']} "
                f"executed={diagnostic['executed']}"
            )
            for attempt in diagnostic["attempts"] or []:
                lines.append(
                    f"    Z+{attempt['extra_height_m']}: "
                    f"plan={attempt['plan_success']} native={attempt['native_status']}"
                )
    lines.append("")
    lines.append(result["claim_boundary"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args()
    if not args.result_dir.is_dir():
        parser.error("result_dir must be an existing campaign directory")
    result = analyze(args.result_dir)
    rendered = as_text(result)
    print(rendered, end="")
    if args.json_output:
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.text_output:
        args.text_output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
