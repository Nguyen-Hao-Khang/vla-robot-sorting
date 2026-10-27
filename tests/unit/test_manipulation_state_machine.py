from __future__ import annotations

import pytest

from vla_robot_sorting.contracts import (
    GripperCommand,
    GripperExecutionResult,
    ManipulationPhase,
    ManipulationSignal,
    Pose6D,
)
from vla_robot_sorting.orchestration import (
    ManipulationStateMachine,
    gripper_pose_for_desired_object_pose,
)


def test_gripper_command_is_revision_locked_and_bounded() -> None:
    command = GripperCommand(
        normalized_position=0.0,
        compliance_profile="fragile-soft",
        force_scale=0.45,
        scene_revision="scene-1",
        transform_revision="tf-1",
    )

    assert command.normalized_position == 0.0
    assert command.force_scale == pytest.approx(0.45)

    with pytest.raises(ValueError):
        GripperCommand(
            normalized_position=-0.1,
            compliance_profile="default",
            force_scale=1.0,
            scene_revision="scene-1",
            transform_revision="tf-1",
        )


def test_failed_gripper_result_requires_reason() -> None:
    with pytest.raises(ValueError):
        GripperExecutionResult(
            completed=False,
            actual_normalized_position=0.5,
            contact_observed=False,
            holding_object=False,
        )


def test_pick_place_state_machine_requires_explicit_grasp_confirmation() -> None:
    machine = ManipulationStateMachine("scene-1", "tf-1")

    assert (
        machine.apply(ManipulationSignal.MOTION_COMPLETED).next_phase is ManipulationPhase.APPROACH
    )
    assert machine.apply(ManipulationSignal.MOTION_COMPLETED).next_phase is ManipulationPhase.CLOSE
    assert (
        machine.apply(ManipulationSignal.GRIPPER_CLOSED).next_phase
        is ManipulationPhase.VERIFY_GRASP
    )

    illegal = machine.apply(ManipulationSignal.MOTION_COMPLETED)
    assert not illegal.accepted
    assert machine.phase is ManipulationPhase.VERIFY_GRASP

    assert machine.apply(ManipulationSignal.GRASP_CONFIRMED).next_phase is ManipulationPhase.LIFT
    assert (
        machine.apply(ManipulationSignal.MOTION_COMPLETED).next_phase is ManipulationPhase.TRANSPORT
    )
    assert machine.apply(ManipulationSignal.MOTION_COMPLETED).next_phase is ManipulationPhase.PLACE
    assert (
        machine.apply(ManipulationSignal.MOTION_COMPLETED).next_phase is ManipulationPhase.RELEASE
    )
    assert machine.apply(ManipulationSignal.GRIPPER_OPENED).next_phase is ManipulationPhase.RETREAT
    assert (
        machine.apply(ManipulationSignal.MOTION_COMPLETED).next_phase is ManipulationPhase.COMPLETE
    )


def test_scene_stale_fails_closed_into_recovery_required() -> None:
    machine = ManipulationStateMachine("scene-old", "tf-1")
    machine.apply(ManipulationSignal.MOTION_COMPLETED)

    transition = machine.apply(ManipulationSignal.SCENE_STALE)

    assert transition.accepted
    assert transition.next_phase is ManipulationPhase.RECOVERY_REQUIRED
    assert machine.phase is ManipulationPhase.RECOVERY_REQUIRED
    rejected = machine.apply(ManipulationSignal.MOTION_COMPLETED)
    assert not rejected.accepted


def test_grasp_failure_never_advances_to_lift() -> None:
    machine = ManipulationStateMachine("scene-1", "tf-1")
    machine.apply(ManipulationSignal.MOTION_COMPLETED)
    machine.apply(ManipulationSignal.MOTION_COMPLETED)
    machine.apply(ManipulationSignal.GRIPPER_CLOSED)

    transition = machine.apply(ManipulationSignal.GRASP_FAILED)

    assert transition.next_phase is ManipulationPhase.RECOVERY_REQUIRED
    assert machine.phase is not ManipulationPhase.LIFT


def test_placement_pose_preserves_object_to_gripper_transform() -> None:
    current_object = Pose6D(
        (0.1, 0.2, 0.8),
        (0.0, 0.0, 0.0, 1.0),
        "world",
    )
    current_gripper = Pose6D(
        (0.16, 0.2, 0.8),
        (0.0, 0.0, 0.0, 1.0),
        "world",
    )
    desired_object = Pose6D(
        (-0.2, 0.05, 0.741),
        (0.0, 0.0, 0.0, 1.0),
        "world",
    )

    desired_gripper = gripper_pose_for_desired_object_pose(
        current_object,
        current_gripper,
        desired_object,
    )

    assert desired_gripper.position_xyz == pytest.approx((-0.14, 0.05, 0.741))
    assert desired_gripper.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_placement_pose_rotates_rigid_grasp_offset_with_object() -> None:
    current_object = Pose6D(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        "world",
    )
    current_gripper = Pose6D(
        (0.1, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        "world",
    )
    desired_object = Pose6D(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 2**-0.5, 2**-0.5),
        "world",
    )

    desired_gripper = gripper_pose_for_desired_object_pose(
        current_object,
        current_gripper,
        desired_object,
    )

    assert desired_gripper.position_xyz == pytest.approx((1.0, 2.1, 3.0))
