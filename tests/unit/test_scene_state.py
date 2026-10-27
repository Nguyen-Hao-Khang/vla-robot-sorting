import pytest

from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    DestinationRegion,
    GeometrySource,
    SceneObject,
    SceneSnapshot,
)
from vla_robot_sorting.perception.scene_state import SceneStateStore, StaleSceneUpdateError


def snapshot() -> SceneSnapshot:
    return SceneSnapshot(
        scene_revision="scene-1",
        transform_revision="tf-1",
        frame_id="world",
        objects=(
            SceneObject(
                object_id="obj",
                pose=Pose6D((0.0, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0), "world"),
                geometry=BoxGeometry((0.04, 0.04, 0.08)),
                geometry_source=GeometrySource.ESTIMATED_PROXY,
                semantic_labels=("object",),
                identity_confidence=0.9,
            ),
        ),
        destinations=(
            DestinationRegion(
                destination_id="bin",
                pose=Pose6D((0.2, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0), "world"),
                tolerance_xyz_m=(0.03, 0.03, 0.03),
                semantic_labels=("bin",),
            ),
        ),
    )


def test_object_pose_update_advances_scene_revision() -> None:
    store = SceneStateStore(snapshot())
    new_pose = Pose6D((0.1, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0), "world")

    result = store.update_object_pose(
        "obj",
        new_pose,
        expected_scene_revision="scene-1",
        new_scene_revision="scene-2",
    )

    assert result.scene_revision == "scene-2"
    assert result.object("obj").pose == new_pose


def test_stale_update_is_rejected() -> None:
    store = SceneStateStore(snapshot())

    with pytest.raises(StaleSceneUpdateError):
        store.update_object_pose(
            "obj",
            Pose6D((0.1, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0), "world"),
            expected_scene_revision="old",
            new_scene_revision="scene-2",
        )


def test_destination_pose_update_is_versioned() -> None:
    store = SceneStateStore(snapshot())
    pose = Pose6D((0.25, 0.05, 0.05), (0.0, 0.0, 0.0, 1.0), "world")

    result = store.update_destination_pose(
        "bin",
        pose,
        expected_scene_revision="scene-1",
        new_scene_revision="scene-2",
    )

    assert result.destination("bin").pose == pose
    assert result.scene_revision == "scene-2"
