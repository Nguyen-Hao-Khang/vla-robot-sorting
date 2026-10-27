from __future__ import annotations

from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    DestinationRegion,
    GeometrySource,
    ObjectVisibility,
    SceneObject,
    SceneSnapshot,
)
from vla_robot_sorting.contracts.recovery import SceneChangeKind
from vla_robot_sorting.perception.change_detection import SceneChangeDetector


def make_scene(
    *,
    scene_revision: str,
    target_x: float = 0.0,
    destination_x: float = 0.3,
    visibility: ObjectVisibility = ObjectVisibility.VISIBLE,
    transform_revision: str = "tf-1",
) -> SceneSnapshot:
    target = SceneObject(
        object_id="target",
        pose=Pose6D((target_x, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0), "world"),
        geometry=BoxGeometry((0.04, 0.04, 0.08)),
        geometry_source=GeometrySource.ESTIMATED_PROXY,
        semantic_labels=("target",),
        identity_confidence=0.9,
        visibility=visibility,
    )
    destination = DestinationRegion(
        destination_id="bin",
        pose=Pose6D((destination_x, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0), "world"),
        tolerance_xyz_m=(0.03, 0.03, 0.03),
        semantic_labels=("bin",),
    )
    return SceneSnapshot(
        scene_revision=scene_revision,
        transform_revision=transform_revision,
        frame_id="world",
        objects=(target,),
        destinations=(destination,),
    )


def test_target_motion_crossing_threshold_requires_replan() -> None:
    previous = make_scene(scene_revision="s1")
    current = make_scene(scene_revision="s2", target_x=0.04)

    change = SceneChangeDetector().detect(
        previous,
        current,
        target_object_id="target",
        destination_id="bin",
    )

    assert change.kind is SceneChangeKind.TARGET_MOVED
    assert change.position_delta_m == 0.04


def test_occlusion_has_priority_over_pose_delta() -> None:
    previous = make_scene(scene_revision="s1")
    current = make_scene(
        scene_revision="s2",
        target_x=0.2,
        visibility=ObjectVisibility.OCCLUDED,
    )

    change = SceneChangeDetector().detect(
        previous,
        current,
        target_object_id="target",
        destination_id="bin",
    )

    assert change.kind is SceneChangeKind.TARGET_OCCLUDED


def test_destination_motion_is_detected() -> None:
    previous = make_scene(scene_revision="s1")
    current = make_scene(scene_revision="s2", destination_x=0.36)

    change = SceneChangeDetector().detect(
        previous,
        current,
        target_object_id="target",
        destination_id="bin",
    )

    assert change.kind is SceneChangeKind.DESTINATION_MOVED
    assert change.position_delta_m == 0.06


def test_transform_revision_change_has_priority() -> None:
    previous = make_scene(scene_revision="s1", transform_revision="tf-1")
    current = make_scene(scene_revision="s2", transform_revision="tf-2")

    change = SceneChangeDetector().detect(
        previous,
        current,
        target_object_id="target",
        destination_id="bin",
    )

    assert change.kind is SceneChangeKind.TRANSFORM_CHANGED


def test_small_scene_change_continues() -> None:
    previous = make_scene(scene_revision="s1")
    current = make_scene(scene_revision="s2", target_x=0.005, destination_x=0.302)

    change = SceneChangeDetector().detect(
        previous,
        current,
        target_object_id="target",
        destination_id="bin",
    )

    assert change.kind is SceneChangeKind.NONE
    assert change.position_delta_m == 0.0
