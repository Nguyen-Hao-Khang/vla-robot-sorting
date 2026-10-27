import math

import pytest

from vla_robot_sorting.contracts import GraspTarget, Pose6D, SemanticSkill, SkillCommand


def test_pose6d_accepts_normalized_quaternion() -> None:
    pose = Pose6D(
        position_xyz=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    assert pose.frame_id == "world"


def test_pose6d_rejects_non_normalized_quaternion() -> None:
    with pytest.raises(ValueError, match="normalized"):
        Pose6D(
            position_xyz=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 2.0),
            frame_id="world",
        )


def test_pose6d_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError, match="finite"):
        Pose6D(
            position_xyz=(math.nan, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        )


def test_pick_requires_target_object() -> None:
    with pytest.raises(ValueError, match="target_object"):
        SkillCommand(skill=SemanticSkill.PICK)


def test_place_requires_destination() -> None:
    with pytest.raises(ValueError, match="destination"):
        SkillCommand(skill=SemanticSkill.PLACE, target_object="red_cube")


def test_grasp_target_validates_confidence() -> None:
    pose = Pose6D(
        position_xyz=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    with pytest.raises(ValueError, match="confidence"):
        GraspTarget(
            object_id="object-1",
            pose=pose,
            approach_vector=(0.0, 0.0, -1.0),
            gripper_width_m=0.04,
            confidence=1.1,
        )


def test_valid_grasp_target_is_immutable_contract() -> None:
    pose = Pose6D(
        position_xyz=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    target = GraspTarget(
        object_id="object-1",
        pose=pose,
        approach_vector=(0.0, 0.0, -1.0),
        gripper_width_m=0.04,
        confidence=0.95,
        compliance_profile="fragile",
    )
    assert target.pose.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert target.compliance_profile == "fragile"


def test_pose6d_requires_exact_dimensions() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        Pose6D(
            position_xyz=(0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        )
    with pytest.raises(ValueError, match="exactly 4"):
        Pose6D(
            position_xyz=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 1.0),
            frame_id="world",
        )
