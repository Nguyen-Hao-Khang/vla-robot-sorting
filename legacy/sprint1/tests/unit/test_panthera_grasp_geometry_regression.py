from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from vla_robot_sorting.contracts.grasping import GraspRequest, ParallelJawGripper
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import GeometrySource, SceneObject, SceneSnapshot
from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/linux/probe_panthera_grasp_recovery.py"

# Preserved native cross-backend evidence from
# linux-planner-crosscheck-fixed-20261006T083405Z.
VENDOR_GRASP_POSITION = (
    -0.12821649713611305,
    -0.011511797399243448,
    0.8289397123884011,
)
VENDOR_GRASP_WXYZ = (
    0.9211215217338933,
    -0.026079527071710015,
    0.07117667651679982,
    0.3818231019547873,
)
VENDOR_PREGRASP_POSITION = (
    -0.18407969706149654,
    -0.06748767302222822,
    0.8410229339098652,
)


def _probe_module():
    spec = importlib.util.spec_from_file_location(
        "probe_panthera_grasp_geometry_regression",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quat_angle_rad(
    left_xyzw: tuple[float, float, float, float],
    right_xyzw: tuple[float, float, float, float],
) -> float:
    dot = abs(sum(a * b for a, b in zip(left_xyzw, right_xyzw, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def test_seed0_collision_proxy_contains_vendor_compatible_non_top_down_candidate() -> None:
    module = _probe_module()
    geometry, provenance = module.collision_bounds_geometry_proxy(
        (0.074118492603302, 0.15479133799672126, 0.0732840633392334),
        mesh_max_radial_distance_m=0.039,
        mesh_long_axis_local_index=1,
    )
    assert provenance["primitive"] == "cylinder"

    target = SceneObject(
        object_id="pillbottle",
        pose=Pose6D(
            (
                -0.058386655905964584,
                0.058461746900672185,
                0.817362591965982,
            ),
            (
                0.499254047870636,
                0.5000146627426147,
                0.4999855160713196,
                0.5007448196411133,
            ),
            "world",
        ),
        geometry=geometry,
        geometry_source=GeometrySource.SIM_GROUND_TRUTH,
        semantic_labels=("pill bottle", "pillbottle"),
        identity_confidence=1.0,
    )
    scene = SceneSnapshot(
        scene_revision="b57-seed-0-initial",
        transform_revision="robotwin-static-v1",
        frame_id="world",
        objects=(target,),
        table_height_m=0.7400000095367432,
    )
    gripper = ParallelJawGripper(
        min_width_m=0.0,
        max_width_m=0.08,
        finger_thickness_m=0.036,
        finger_height_m=0.036,
        pregrasp_standoff_m=0.08,
        jaw_clearance_m=0.0,
        contact_center_offset_m=0.10,
    )

    result = GeometryFirstGraspGenerator(
        cylinder_radial_samples=8,
        cylinder_side_grasp_height_fractions=(0.0, 0.25, 0.75),
    ).generate(
        GraspRequest(
            object_id="pillbottle",
            scene_revision=scene.scene_revision,
            transform_revision=scene.transform_revision,
            gripper=gripper,
        ),
        scene,
    )

    candidate = next(
        item
        for item in result.candidates
        if item.candidate_id == "pillbottle:cyl:side:1:s-1:up25"
    )
    vendor_xyzw = (
        VENDOR_GRASP_WXYZ[1],
        VENDOR_GRASP_WXYZ[2],
        VENDOR_GRASP_WXYZ[3],
        VENDOR_GRASP_WXYZ[0],
    )
    position_error = math.sqrt(
        sum(
            (actual - expected) ** 2
            for actual, expected in zip(
                candidate.grasp_pose.position_xyz,
                VENDOR_GRASP_POSITION,
                strict=True,
            )
        )
    )
    orientation_error = _quat_angle_rad(
        candidate.grasp_pose.orientation_xyzw,
        vendor_xyzw,
    )
    pregrasp_position_error = math.sqrt(
        sum(
            (actual - expected) ** 2
            for actual, expected in zip(
                candidate.pregrasp_pose.position_xyz,
                VENDOR_PREGRASP_POSITION,
                strict=True,
            )
        )
    )

    assert abs(candidate.approach_direction[2]) <= 0.5
    assert candidate.jaw_width_m <= gripper.max_width_m
    assert candidate.metadata["table_clearance_m"] > 0.07
    assert candidate.metadata["gripper_frame_contact_offset_m"] == 0.10
    assert position_error < 0.005
    assert pregrasp_position_error < 0.012
    assert math.dist(
        candidate.pregrasp_pose.position_xyz,
        candidate.grasp_pose.position_xyz,
    ) == pytest.approx(0.08)
    assert math.degrees(orientation_error) < 10.0
