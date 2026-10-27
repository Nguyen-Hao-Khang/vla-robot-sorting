"""Classify the next engineering actions from a native manipulation campaign.

This tool is intentionally post-processing only. It never changes acceptance
manifests and never executes robot motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _campaign_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _stage(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in payload.get("stages") or []:
        if isinstance(row, dict) and row.get("stage") == name:
            return row
    return None


def _native_status(stage: dict[str, Any] | None) -> str | None:
    if not isinstance(stage, dict):
        return None
    failure = stage.get("failure")
    if not isinstance(failure, dict):
        return None
    details = failure.get("details")
    if not isinstance(details, dict):
        return None
    value = details.get("native_status")
    return str(value) if value is not None else None


def _start_diag(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("carried_start_state_diagnostic")
    return value if isinstance(value, dict) else {}


def _table_diag(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("lift_table_disabled_diagnostic")
    return value if isinstance(value, dict) else {}


def _support_geom(payload: dict[str, Any]) -> dict[str, Any]:
    attachment = payload.get("carried_object_attachment")
    if not isinstance(attachment, dict):
        return {}
    value = attachment.get("support_contact_geometry")
    return value if isinstance(value, dict) else {}


def _status_has(status: str | None, *terms: str) -> bool:
    if status is None:
        return False
    lowered = status.lower()
    return any(term.lower() in lowered for term in terms)


def classify_campaign(result_dir: Path) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    campaign = _campaign_values(result_dir / "CAMPAIGN_RESULT.txt")
    payloads = {
        seed: _load_json(result_dir / f"manipulation-seed-{seed}.json")
        for seed in (0, 1, 2)
    }

    accepted = campaign.get("CAMPAIGN_ACCEPTED") == "1"
    closure_ready = (
        accepted
        and campaign.get("CLOSURE_CANDIDATE_RC") == "0"
        and campaign.get("NATIVE_CLOSURE_RC") == "0"
    )
    decisions: list[dict[str, Any]] = []
    seed_summaries: dict[str, Any] = {}

    if closure_ready:
        decisions.append(
            {
                "code": "READY_FOR_SPRINT1_CLOSURE",
                "seeds": [0, 1, 2],
                "reason": (
                    "manipulation acceptance, provisional closure candidate, readiness "
                    "and native closure audit all passed"
                ),
                "next_action": (
                    "review exported evidence on Windows and promote the closure candidate "
                    "to a durable accepted manifest"
                ),
            }
        )
    elif accepted:
        decisions.append(
            {
                "code": "CLOSURE_PIPELINE_ERROR",
                "seeds": [0, 1, 2],
                "reason": (
                    "manipulation campaign passed but closure candidate/readiness audit "
                    "did not complete successfully"
                ),
                "next_action": (
                    "preserve the manipulation PASS and repair only the closure bookkeeping "
                    "pipeline; do not rerun native manipulation unless evidence is incomplete"
                ),
            }
        )

    for seed, payload in payloads.items():
        if (
            isinstance(payload, dict)
            and payload.get("passed") is False
            and "pinned upstream task source hash mismatch" in str(payload.get("error"))
            and not payload.get("stages")
        ):
            decisions.append(
                {
                    "code": "UPSTREAM_TASK_SOURCE_ATTESTATION_MISMATCH",
                    "seeds": [seed],
                    "reason": (
                        "native task setup stopped at strict source SHA256 attestation, "
                        "before any manipulation planning or execution"
                    ),
                    "next_action": (
                        "validate expected hash against the pinned upstream Git blob; "
                        "retain fail-closed attestation and rerun native only after correction"
                    ),
                }
            )

    for seed in (0, 2):
        payload = payloads.get(seed)
        if not isinstance(payload, dict):
            seed_summaries[str(seed)] = {"present": False}
            continue
        lift = _stage(payload, "lift")
        start = _start_diag(payload)
        table = _table_diag(payload)
        support = _support_geom(payload)
        normal_status = _native_status(lift)
        table_status = table.get("native_status")
        table_start = table.get("start_state")
        table_start = table_start if isinstance(table_start, dict) else {}

        seed_summaries[str(seed)] = {
            "present": True,
            "passed": payload.get("passed"),
            "lift_plan_success": lift.get("plan_success") if lift else None,
            "normal_native_status": normal_status,
            "start_state": start,
            "table_disabled": table,
            "support_contact_geometry": support,
        }

        world_collision = (
            start.get("valid") is False
            and _status_has(str(start.get("status")), "colliding with world")
        )
        table_recovers = (
            table.get("obstacle_disabled") is True
            and table.get("obstacle_reenabled") is True
            and table.get("executed") is False
            and table_start.get("valid") is True
            and table.get("plan_success") is True
        )
        support_escape = (
            lift.get("support_escape")
            if isinstance(lift, dict) and isinstance(lift.get("support_escape"), dict)
            else {}
        )
        support_recovered = (
            isinstance(lift, dict)
            and lift.get("passed") is True
            and support_escape.get("passed") is True
        )
        retreat = _stage(payload, "retreat")
        release = _stage(payload, "release")
        retreat_contacts = retreat.get("contacts") if retreat else None
        if (
            support_recovered
            and isinstance(release, dict)
            and release.get("passed") is True
            and payload.get("task_success_after_release") is True
            and isinstance(retreat, dict)
            and retreat.get("passed") is False
            and isinstance(retreat_contacts, dict)
            and int(retreat_contacts.get("robot_target_contact_frames", 0)) > 0
        ):
            decisions.append(
                {
                    "code": "POST_RELEASE_CONTACT_SEPARATION",
                    "seeds": [seed],
                    "reason": "release passed but retreat retained forbidden robot-target contact",
                    "next_action": (
                        "validate bounded upward separation, pad support, finger clearance "
                        "and remaining contact-free retreat with independent native audit"
                    ),
                }
            )
            continue

        place = _stage(payload, "place_above_support")
        if (
            support_recovered
            and isinstance(place, dict)
            and place.get("passed") is False
            and place.get("plan_success") is False
            and _native_status(place) == "IK Fail"
        ):
            decisions.append(
                {
                    "code": "PLACEMENT_YAW_IK_RECOVERY",
                    "seeds": [seed],
                    "reason": "carried lift passed but place-above-support ended in native IK Fail",
                    "next_action": (
                        "test bounded yaw-only placement at identical target XYZ; "
                        "retry exclusively after exact native IK Fail"
                    ),
                }
            )
            continue

        if world_collision and table_recovers and not support_recovered:
            decisions.append(
                {
                    "code": "SUPPORT_CONTACT_START_COLLISION",
                    "seeds": [seed],
                    "reason": (
                        "normal attached start state is world-colliding while table-only "
                        "planning diagnostic makes the same start valid and lift-plan feasible"
                    ),
                    "signed_gap_m": support.get("signed_gap_m"),
                    "next_action": (
                        "design a bounded support-contact escape policy with independent "
                        "robot/table and carried-object/table safety validation"
                    ),
                }
            )
            continue

        if start.get("valid") is True and _status_has(
            normal_status,
            "ik",
            "trajopt",
            "optimization",
        ):
            decisions.append(
                {
                    "code": "IK_OR_TRAJOPT",
                    "seeds": [seed],
                    "reason": (
                        "attached start state is valid but native carried planner still "
                        f"fails with status={normal_status!r}"
                    ),
                    "next_action": (
                        "audit carried goal transform, attached sphere pose and payload "
                        "collision geometry without weakening thresholds"
                    ),
                }
            )
            continue

        same_failure = (
            bool(table)
            and table.get("plan_success") is False
            and normal_status is not None
            and str(table_status) == str(normal_status)
        )
        if same_failure:
            decisions.append(
                {
                    "code": "TABLE_NOT_PRIMARY",
                    "seeds": [seed],
                    "reason": (
                        "table-only disable does not change the carried lift failure status"
                    ),
                    "next_action": (
                        "focus on self-collision, IK, attachment pose and collision-sphere "
                        "placement rather than support contact"
                    ),
                }
            )
            continue

        if lift is not None and lift.get("plan_success") is False:
            decisions.append(
                {
                    "code": "CARRIED_LIFT_UNRESOLVED",
                    "seeds": [seed],
                    "reason": "carried lift planning remains failed but evidence is not decisive",
                    "next_action": "inspect native MotionGen diagnostics and attachment geometry",
                }
            )

    seed1 = payloads.get(1)
    if isinstance(seed1, dict):
        approach = _stage(seed1, "approach")
        collision = approach.get("collision_validation") if approach else None
        collisions = (
            collision.get("collisions")
            if isinstance(collision, dict)
            else None
        )
        contacts = approach.get("contacts") if approach else None
        contact_policy = approach.get("contact_policy") if approach else None
        seed_summaries["1"] = {
            "present": True,
            "passed": seed1.get("passed"),
            "approach_planner": approach.get("planner_id") if approach else None,
            "approach_plan_success": approach.get("plan_success") if approach else None,
            "approach_collision_safe": approach.get("collision_safe") if approach else None,
            "approach_execution_status": (
                approach.get("execution_status") if approach else None
            ),
            "robot_target_contact_frames": (
                contacts.get("robot_target_contact_frames")
                if isinstance(contacts, dict)
                else None
            ),
            "collision_count": len(collisions) if isinstance(collisions, list) else 0,
        }
        execution_contact_rejected = (
            isinstance(approach, dict)
            and approach.get("passed") is False
            and approach.get("planner_id") == "robotwin-mplib-rrt"
            and approach.get("plan_success") is True
            and approach.get("dynamic_safe") is True
            and approach.get("collision_safe") is True
            and approach.get("execution_status") == "completed"
            and isinstance(contacts, dict)
            and int(contacts.get("robot_target_contact_frames", 0)) > 0
            and isinstance(contact_policy, dict)
            and contact_policy.get("allow_robot_target_contact") is False
        )
        target_collision_rejected = (
            isinstance(approach, dict)
            and approach.get("passed") is False
            and approach.get("planner_id") == "robotwin-mplib-rrt"
            and approach.get("plan_success") is True
            and approach.get("collision_safe") is False
            and isinstance(collisions, list)
            and any(
                isinstance(row, dict)
                and "target" in (row.get("matched_environment_roles") or [])
                for row in collisions
            )
        )
        if target_collision_rejected:
            decisions.append(
                {
                    "code": "SEED1_ACQUISITION_TARGET_COLLISION",
                    "seeds": [1],
                    "reason": (
                        "target-aware RRT path was rejected by independent "
                        "target collision checks"
                    ),
                    "next_action": (
                        "launch deterministic ranked fresh-process acquisition recovery; "
                        "never accept the colliding original approach"
                    ),
                }
            )
        if execution_contact_rejected:
            decisions.append(
                {
                    "code": "SEED1_EXECUTION_CONTACT_REJECTED",
                    "seeds": [1],
                    "reason": (
                        "target-aware RRT passed planning/collision checks but runtime "
                        "execution produced forbidden robot-target contact"
                    ),
                    "robot_target_contact_frames": int(
                        contacts.get("robot_target_contact_frames", 0)
                    ),
                    "next_action": (
                        "run bounded alternate-candidate recovery in fresh processes; "
                        "do not relax the approach contact policy"
                    ),
                }
            )
    else:
        seed_summaries["1"] = {"present": False}

    recovery_count = int(campaign.get("RECOVERY_DIAG_COUNT", "0") or 0)
    recovery_pass = int(campaign.get("RECOVERY_PASS_COUNT", "0") or 0)
    if recovery_pass > 0:
        decisions.append(
            {
                "code": "SEED1_RECOVERY_CANDIDATE",
                "seeds": [1],
                "reason": (
                    f"{recovery_pass}/{recovery_count} diagnostic alternate "
                    "candidates passed"
                ),
                "next_action": (
                    "audit the successful candidate provenance and design a formal recovery "
                    "selection policy before rerunning fixed-seed acceptance"
                ),
            }
        )
    elif recovery_count > 0:
        decisions.append(
            {
                "code": "SEED1_RECOVERY_FAILED",
                "seeds": [1],
                "reason": f"0/{recovery_count} alternate candidate diagnostics passed",
                "next_action": (
                    "inspect alternate-candidate failure modes; do not brute-force more "
                    "candidates without a geometric or planner-policy rationale"
                ),
            }
        )

    if not decisions:
        decisions.append(
            {
                "code": "INSUFFICIENT_EVIDENCE",
                "seeds": [],
                "reason": "campaign evidence does not match a known follow-up branch",
                "next_action": "audit raw per-seed JSON and campaign logs",
            }
        )

    merged: list[dict[str, Any]] = []
    for decision in decisions:
        decision_body = {
            key: value for key, value in decision.items() if key != "seeds"
        }
        match = next(
            (
                row
                for row in merged
                if {
                    key: value for key, value in row.items() if key != "seeds"
                }
                == decision_body
            ),
            None,
        )
        if match is None:
            merged.append(dict(decision))
            continue
        match["seeds"] = sorted(
            set(match.get("seeds") or []) | set(decision.get("seeds") or [])
        )

    priority = {
        "READY_FOR_SPRINT1_CLOSURE": 0,
        "CLOSURE_PIPELINE_ERROR": 5,
        "UPSTREAM_TASK_SOURCE_ATTESTATION_MISMATCH": 8,
        "SUPPORT_CONTACT_START_COLLISION": 10,
        "POST_RELEASE_CONTACT_SEPARATION": 15,
        "IK_OR_TRAJOPT": 20,
        "PLACEMENT_YAW_IK_RECOVERY": 25,
        "TABLE_NOT_PRIMARY": 30,
        "CARRIED_LIFT_UNRESOLVED": 40,
        "SEED1_RECOVERY_CANDIDATE": 42,
        "SEED1_EXECUTION_CONTACT_REJECTED": 45,
        "SEED1_ACQUISITION_TARGET_COLLISION": 46,
        "SEED1_RECOVERY_FAILED": 60,
        "INSUFFICIENT_EVIDENCE": 99,
    }
    merged.sort(key=lambda row: priority.get(str(row.get("code")), 90))

    return {
        "schema": "manipulation-followup-classifier-v1",
        "result_dir": str(result_dir),
        "campaign": campaign,
        "campaign_accepted": accepted,
        "native_closure_ready": closure_ready,
        "primary_decision": merged[0]["code"],
        "decisions": merged,
        "seed_summaries": seed_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = classify_campaign(args.result_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"primary_decision={result['primary_decision']}")
        for row in result["decisions"]:
            print(
                f"- {row['code']}: {row['reason']}\n"
                f"  next: {row['next_action']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
