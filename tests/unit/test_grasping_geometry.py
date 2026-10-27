from __future__ import annotations

from math import sqrt

import pytest

from vla_robot_sorting.contracts.grasping import (
    GraspFailureCode,
    GraspRequest,
    ParallelJawGripper,
)
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    CylinderGeometry,
    GeometrySource,
    SceneObject,
    SceneSnapshot,
)
from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator


def gripper() -> ParallelJawGripper:
    return ParallelJawGripper(
        min_width_m=0.01,
        max_width_m=0.09,
        finger_thickness_m=0.01,
        finger_height_m=0.02,
        pregrasp_standoff_m=0.08,
        jaw_clearance_m=0.004,
    )


def scene_with(obj: SceneObject) -> SceneSnapshot:
    return SceneSnapshot(
        scene_revision="scene-1",
        transform_revision="tf-1",
        frame_id="world",
        objects=(obj,),
        table_height_m=0.0,
    )


def request(object_id: str) -> GraspRequest:
    return GraspRequest(
        object_id=object_id,
        scene_revision="scene-1",
        transform_revision="tf-1",
        gripper=gripper(),
    )


def box_object(*, fragile: bool = False, orientation=(0.0, 0.0, 0.0, 1.0)) -> SceneObject:
    return SceneObject(
        object_id="box-1",
        pose=Pose6D((0.2, 0.0, 0.08), orientation, "world"),
        geometry=BoxGeometry((0.04, 0.06, 0.10)),
        geometry_source=GeometrySource.SIM_GROUND_TRUTH,
        semantic_labels=("blue box", "box"),
        identity_confidence=0.95,
        fragile=fragile,
    )


def test_box_generator_emits_full_se3_and_rejects_too_wide_axis() -> None:
    scene = scene_with(box_object())

    result = GeometryFirstGraspGenerator().generate(request("box-1"), scene)

    assert result.succeeded
    assert result.failure is None
    assert len(result.candidates) >= 4
    assert all(candidate.grasp_pose.frame_id == "world" for candidate in result.candidates)
    assert all(candidate.jaw_width_m <= gripper().max_width_m for candidate in result.candidates)
    assert all(
        abs(sum(v * v for v in candidate.grasp_pose.orientation_xyzw) - 1.0) < 1e-6
        for candidate in result.candidates
    )
    assert all(candidate.metadata["privileged_geometry"] is True for candidate in result.candidates)


def test_rotated_box_preserves_object_specific_orientation() -> None:
    half = sqrt(0.5)
    scene = scene_with(box_object(orientation=(0.0, 0.0, half, half)))

    result = GeometryFirstGraspGenerator().generate(request("box-1"), scene)

    assert result.succeeded
    assert any(abs(candidate.closing_direction[0]) > 0.9 for candidate in result.candidates)
    assert any(abs(candidate.closing_direction[1]) > 0.9 for candidate in result.candidates)


def test_fragile_object_sets_compliance_proxy() -> None:
    scene = scene_with(box_object(fragile=True))

    result = GeometryFirstGraspGenerator().generate(request("box-1"), scene)

    assert result.succeeded
    assert {candidate.compliance_profile for candidate in result.candidates} == {"fragile-soft"}
    assert all(
        candidate.metadata["recommended_force_scale"] == pytest.approx(0.45)
        for candidate in result.candidates
    )


def test_cylinder_generator_emits_side_and_top_candidates() -> None:
    obj = SceneObject(
        object_id="bottle",
        pose=Pose6D((0.1, 0.1, 0.10), (0.0, 0.0, 0.0, 1.0), "world"),
        geometry=CylinderGeometry(radius_m=0.025, height_m=0.12),
        geometry_source=GeometrySource.ESTIMATED_PROXY,
        semantic_labels=("pill bottle", "bottle"),
        identity_confidence=0.8,
    )

    result = GeometryFirstGraspGenerator(cylinder_radial_samples=8).generate(
        request("bottle"),
        scene_with(obj),
    )

    assert result.succeeded
    modes = {candidate.metadata["mode"] for candidate in result.candidates}
    assert modes == {"side", "top"}
    assert all(candidate.jaw_width_m == pytest.approx(0.054) for candidate in result.candidates)


def test_stale_scene_is_separate_generation_failure() -> None:
    scene = scene_with(box_object())
    stale_request = GraspRequest(
        object_id="box-1",
        scene_revision="old-scene",
        transform_revision="tf-1",
        gripper=gripper(),
    )

    result = GeometryFirstGraspGenerator().generate(stale_request, scene)

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is GraspFailureCode.SCENE_STALE


def test_missing_target_is_not_reported_as_grasp_geometry_failure() -> None:
    scene = scene_with(box_object())

    result = GeometryFirstGraspGenerator().generate(request("missing"), scene)

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is GraspFailureCode.TARGET_NOT_FOUND


def test_box_generator_emits_elevated_roll_variants_for_side_grasps() -> None:
    obj = SceneObject(
        object_id="tall-box",
        pose=Pose6D((0.2, 0.0, 0.08), (0.0, 0.0, 0.0, 1.0), "world"),
        geometry=BoxGeometry((0.04, 0.06, 0.10)),
        geometry_source=GeometrySource.SIM_GROUND_TRUTH,
        semantic_labels=("tall box",),
        identity_confidence=1.0,
    )
    req = GraspRequest(
        object_id="tall-box",
        scene_revision="scene-1",
        transform_revision="tf-1",
        gripper=gripper(),
    )

    result = GeometryFirstGraspGenerator(
        box_side_grasp_height_fractions=(0.0, 0.75),
        box_parallel_jaw_roll_variants=True,
    ).generate(req, scene_with(obj))

    assert result.succeeded
    elevated = [
        candidate
        for candidate in result.candidates
        if candidate.metadata.get("grasp_height_fraction") == pytest.approx(0.75)
    ]
    assert elevated
    assert all(candidate.metadata["up_axis_local_index"] == 2 for candidate in elevated)
    assert all(
        candidate.grasp_pose.position_xyz[2] == pytest.approx(0.11)
        for candidate in elevated
    )
    assert any(":roll180" in candidate.candidate_id for candidate in elevated)

    baseline = next(
        candidate
        for candidate in result.candidates
        if candidate.candidate_id == "tall-box:box:c0:a1:s-1"
    )
    rolled = next(
        candidate
        for candidate in result.candidates
        if candidate.candidate_id == "tall-box:box:c0:a1:s-1:roll180"
    )
    closing_dot = sum(
        left * right
        for left, right in zip(
            baseline.closing_direction,
            rolled.closing_direction,
            strict=True,
        )
    )
    assert closing_dot == pytest.approx(-1.0)


def test_gripper_contact_offset_moves_planner_frame_not_contact_geometry() -> None:
    obj = box_object()
    offset_gripper = ParallelJawGripper(
        min_width_m=0.01,
        max_width_m=0.09,
        finger_thickness_m=0.01,
        finger_height_m=0.02,
        pregrasp_standoff_m=0.08,
        jaw_clearance_m=0.004,
        contact_center_offset_m=0.10,
    )
    req = GraspRequest(
        object_id="box-1",
        scene_revision="scene-1",
        transform_revision="tf-1",
        gripper=offset_gripper,
    )

    result = GeometryFirstGraspGenerator().generate(req, scene_with(obj))

    candidate = next(
        item
        for item in result.candidates
        if item.candidate_id == "box-1:box:c0:a1:s-1"
    )
    assert candidate.approach_direction == pytest.approx((0.0, -1.0, 0.0))
    assert candidate.metadata["contact_center_position_xyz_m"] == pytest.approx(
        obj.pose.position_xyz
    )
    assert candidate.grasp_pose.position_xyz == pytest.approx((0.2, 0.10, 0.08))
    assert candidate.pregrasp_pose.position_xyz == pytest.approx((0.2, 0.18, 0.08))
    assert candidate.metadata["gripper_frame_contact_offset_m"] == pytest.approx(0.10)
    assert candidate.metadata["table_clearance_m"] == pytest.approx(0.07)


def test_cylinder_axis_local_supports_radial_side_grasps_and_elevation() -> None:
    half = sqrt(0.5)
    obj = SceneObject(
        object_id="upright-y-cylinder",
        pose=Pose6D((0.0, 0.0, 0.20), (half, 0.0, 0.0, half), "world"),
        geometry=CylinderGeometry(
            radius_m=0.035,
            height_m=0.16,
            axis_local=(0.0, 1.0, 0.0),
        ),
        geometry_source=GeometrySource.SIM_GROUND_TRUTH,
        semantic_labels=("bottle",),
        identity_confidence=1.0,
    )
    req = GraspRequest(
        object_id=obj.object_id,
        scene_revision="scene-1",
        transform_revision="tf-1",
        gripper=gripper(),
    )

    result = GeometryFirstGraspGenerator(
        cylinder_radial_samples=8,
        cylinder_side_grasp_height_fractions=(0.0, 0.75),
    ).generate(req, scene_with(obj))

    assert result.succeeded
    side = [
        candidate
        for candidate in result.candidates
        if candidate.metadata.get("mode") == "side"
    ]
    assert len(side) == 32
    assert any(
        candidate.metadata["sample_index"] == 1
        and candidate.metadata["grasp_height_fraction"] == pytest.approx(0.75)
        for candidate in side
    )
    elevated = [
        candidate
        for candidate in side
        if candidate.metadata["grasp_height_fraction"] == pytest.approx(0.75)
    ]
    expected_offset = 0.75 * (0.5 * 0.16 - 0.5 * gripper().finger_height_m)
    assert all(
        candidate.metadata["contact_center_position_xyz_m"][2]
        == pytest.approx(0.20 + expected_offset)
        for candidate in elevated
    )
