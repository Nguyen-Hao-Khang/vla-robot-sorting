#!/usr/bin/env python3
"""Run deterministic Windows-safe Sprint 1 scenarios and emit JSONL evidence.

These records are explicitly tagged offline_fake and must never be reported as
native RoboTwin manipulation success.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from vla_robot_sorting.contracts.evaluation import (
    EvaluationRuntime,
    ScenarioOutcome,
    ScenarioRecord,
)
from vla_robot_sorting.contracts.grasping import GraspRequest, ParallelJawGripper
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.motion import (
    JointLimits,
    JointState,
    MotionGoal,
    PlanningContext,
)
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    DestinationRegion,
    GeometrySource,
    ObjectVisibility,
    SceneObject,
    SceneSnapshot,
)
from vla_robot_sorting.contracts.recovery import (
    RecoveryAction,
    RecoveryEvent,
    RecoveryState,
    RecoveryTrigger,
)
from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator
from vla_robot_sorting.orchestration import BoundedMotionController
from vla_robot_sorting.orchestration.recovery import ClosedLoopRecoveryPolicy
from vla_robot_sorting.perception.change_detection import SceneChangeDetector
from vla_robot_sorting.safety import JointLimitSafetyValidator
from vla_robot_sorting.semantics.grounding import CatalogGrounder, GroundingRequest
from vla_robot_sorting.testing import (
    RecordingTrajectoryExecutor,
    StraightLineMotionPlanner,
    UniformTrajectoryParameterizer,
)


def _scene(
    revision: str,
    *,
    target_x: float = 0.0,
    destination_x: float = 0.3,
    visibility: ObjectVisibility = ObjectVisibility.VISIBLE,
    include_target: bool = True,
) -> SceneSnapshot:
    objects: list[SceneObject] = []
    if include_target:
        objects.append(
            SceneObject(
                object_id="red-block",
                pose=Pose6D(
                    (target_x, 0.0, 0.06),
                    (0.0, 0.0, 0.0, 1.0),
                    "world",
                ),
                geometry=BoxGeometry((0.04, 0.04, 0.08)),
                geometry_source=GeometrySource.ESTIMATED_PROXY,
                semantic_labels=("red block", "khối đỏ", "block"),
                identity_confidence=0.95,
                visibility=visibility,
            )
        )
    objects.append(
        SceneObject(
            object_id="blue-block",
            pose=Pose6D(
                (0.12, 0.0, 0.06),
                (0.0, 0.0, 0.0, 1.0),
                "world",
            ),
            geometry=BoxGeometry((0.04, 0.04, 0.08)),
            geometry_source=GeometrySource.ESTIMATED_PROXY,
            semantic_labels=("blue block", "khối xanh", "block"),
            identity_confidence=0.9,
        )
    )

    return SceneSnapshot(
        scene_revision=revision,
        transform_revision="tf-1",
        frame_id="world",
        objects=tuple(objects),
        destinations=(
            DestinationRegion(
                destination_id="bin",
                pose=Pose6D(
                    (destination_x, 0.0, 0.06),
                    (0.0, 0.0, 0.0, 1.0),
                    "world",
                ),
                tolerance_xyz_m=(0.03, 0.03, 0.03),
                semantic_labels=("bin", "thùng"),
            ),
        ),
        table_height_m=0.0,
    )


def _record(
    scenario_id: str,
    suite: str,
    expected: str,
    observed: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_id=scenario_id,
        suite=suite,
        runtime=EvaluationRuntime.OFFLINE_FAKE,
        outcome=ScenarioOutcome.PASS,
        expected_behavior=expected,
        observed_behavior=observed,
        metadata=metadata or {},
    )


def run() -> tuple[ScenarioRecord, ...]:
    records: list[ScenarioRecord] = []
    scene = _scene("scene-1")

    grounded = CatalogGrounder().ground(
        GroundingRequest("khối đỏ", "scene-1", language="vi"),
        scene,
    )
    if not grounded.succeeded or grounded.object_id != "red-block":
        raise RuntimeError("offline grounding regression failed")
    records.append(
        _record(
            "S1-R-001",
            "regression",
            "structured_success",
            "structured_success",
            metadata={"grounded_object_id": grounded.object_id, "language": "vi"},
        )
    )

    gripper = ParallelJawGripper(
        min_width_m=0.01,
        max_width_m=0.09,
        finger_thickness_m=0.01,
        finger_height_m=0.02,
        pregrasp_standoff_m=0.08,
    )
    grasp_result = GeometryFirstGraspGenerator().generate(
        GraspRequest("red-block", "scene-1", "tf-1", gripper),
        scene,
    )
    if not grasp_result.succeeded:
        raise RuntimeError("offline grasp generation regression failed")
    records.append(
        _record(
            "S1-R-002",
            "regression",
            "grasp_candidates_generated",
            "grasp_candidates_generated",
            metadata={"candidate_count": len(grasp_result.candidates)},
        )
    )

    joint_names = ("j1", "j2")
    limits = JointLimits(
        joint_names=joint_names,
        lower_position_rad=(-2.0, -2.0),
        upper_position_rad=(2.0, 2.0),
        max_velocity_rad_s=(2.0, 2.0),
        max_acceleration_rad_s2=(5.0, 5.0),
        max_jerk_rad_s3=(20.0, 20.0),
    )
    controller = BoundedMotionController(
        planner=StraightLineMotionPlanner(
            joint_names=joint_names,
            target_positions=(0.4, -0.2),
        ),
        parameterizer=UniformTrajectoryParameterizer(segment_duration_s=1.0),
        safety=JointLimitSafetyValidator(limits),
        executor=RecordingTrajectoryExecutor(),
        limits=limits,
        controller_period_s=1.0,
        horizon_s=1.0,
    )
    motion = controller.run(
        JointState(joint_names=joint_names, positions=(0.0, 0.0)),
        MotionGoal(
            Pose6D((0.2, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), "world")
        ),
        PlanningContext("scene-1", "fake-transform"),
        scene={},
        segment_id="offline-demo",
    )
    if not motion.succeeded:
        raise RuntimeError("offline bounded motion regression failed")
    records.append(
        _record(
            "S1-R-003",
            "regression",
            "bounded_motion_cycle",
            "bounded_motion_cycle",
            metadata={"executed_until_s": motion.execution.executed_until_s},
        )
    )

    ambiguous = CatalogGrounder().ground(
        GroundingRequest("block", "scene-1", language="en"),
        scene,
    )
    if ambiguous.succeeded or ambiguous.failure is None:
        raise RuntimeError("offline ambiguity regression failed")
    records.append(
        _record(
            "S1-A-001",
            "adversarial",
            "clarify",
            "clarify",
            metadata={"candidate_ids": list(ambiguous.failure.candidate_ids)},
        )
    )


    policy = ClosedLoopRecoveryPolicy()
    moved = SceneChangeDetector().detect(
        scene,
        _scene("scene-2", target_x=0.04),
        target_object_id="red-block",
        destination_id="bin",
    )
    moved_decision = policy.decide(
        RecoveryEvent(
            RecoveryTrigger.SCENE_CHANGE,
            observed_at_s=time.monotonic(),
            scene_change=moved,
        ),
        RecoveryState(),
    )
    if moved_decision.action is not RecoveryAction.REPLAN:
        raise RuntimeError("offline moved-target recovery regression failed")
    records.append(
        _record(
            "S1-A-002",
            "adversarial",
            "replan",
            moved_decision.action.value,
            metadata={"scene_change": moved.kind.value},
        )
    )

    destination_moved = SceneChangeDetector().detect(
        scene,
        _scene("scene-2-destination", destination_x=0.35),
        target_object_id="red-block",
        destination_id="bin",
    )
    destination_decision = policy.decide(
        RecoveryEvent(
            RecoveryTrigger.SCENE_CHANGE,
            observed_at_s=time.monotonic(),
            scene_change=destination_moved,
        ),
        RecoveryState(),
    )
    if destination_decision.action is not RecoveryAction.REPLAN:
        raise RuntimeError("offline moved-destination recovery regression failed")
    records.append(
        _record(
            "S1-A-003",
            "adversarial",
            "replan",
            destination_decision.action.value,
            metadata={"scene_change": destination_moved.kind.value},
        )
    )

    occluded = SceneChangeDetector().detect(
        scene,
        _scene("scene-3", visibility=ObjectVisibility.OCCLUDED),
        target_object_id="red-block",
        destination_id="bin",
    )
    occluded_decision = policy.decide(
        RecoveryEvent(
            RecoveryTrigger.SCENE_CHANGE,
            observed_at_s=time.monotonic(),
            scene_change=occluded,
        ),
        RecoveryState(),
    )
    if occluded_decision.action is not RecoveryAction.REOBSERVE:
        raise RuntimeError("offline occlusion recovery regression failed")
    records.append(
        _record(
            "S1-A-004",
            "adversarial",
            "reobserve",
            occluded_decision.action.value,
            metadata={"scene_change": occluded.kind.value},
        )
    )

    missing = SceneChangeDetector().detect(
        scene,
        _scene("scene-4", include_target=False),
        target_object_id="red-block",
        destination_id="bin",
    )
    missing_decision = policy.decide(
        RecoveryEvent(
            RecoveryTrigger.SCENE_CHANGE,
            observed_at_s=time.monotonic(),
            scene_change=missing,
        ),
        RecoveryState(),
    )
    if missing_decision.action not in {RecoveryAction.REASON, RecoveryAction.ABORT}:
        raise RuntimeError("offline removed-target recovery regression failed")
    records.append(
        _record(
            "S1-A-005",
            "adversarial",
            "reason_or_abort",
            missing_decision.action.value,
            metadata={"scene_change": missing.kind.value},
        )
    )

    grasp_failure = policy.decide(
        RecoveryEvent(
            RecoveryTrigger.GRASP_FAILURE,
            observed_at_s=time.monotonic(),
            reason="observable grasp-retention failure",
        ),
        RecoveryState(),
    )
    if grasp_failure.action is not RecoveryAction.REGRASP:
        raise RuntimeError("offline grasp-failure recovery regression failed")
    records.append(
        _record(
            "S1-A-006",
            "adversarial",
            "regrasp",
            grasp_failure.action.value,
            metadata={"trigger": RecoveryTrigger.GRASP_FAILURE.value},
        )
    )
    return tuple(records)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(asdict(record)), sort_keys=True) + "\n")
    print(f"offline_scenarios_passed={len(records)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
