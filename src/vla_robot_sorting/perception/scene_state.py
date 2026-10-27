"""Versioned scene-state updates used by perception and closed-loop replanning."""

from __future__ import annotations

from dataclasses import replace

from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    DestinationRegion,
    SceneObject,
    SceneSnapshot,
)


class StaleSceneUpdateError(RuntimeError):
    pass


class SceneStateStore:
    """Small in-memory versioned scene store; no simulator dependency."""

    def __init__(self, initial: SceneSnapshot) -> None:
        self._snapshot = initial

    @property
    def snapshot(self) -> SceneSnapshot:
        return self._snapshot

    def _check_revision(self, expected_scene_revision: str) -> None:
        if expected_scene_revision != self._snapshot.scene_revision:
            raise StaleSceneUpdateError(
                f"expected {expected_scene_revision!r}, "
                f"current is {self._snapshot.scene_revision!r}"
            )

    def update_object_pose(
        self,
        object_id: str,
        pose: Pose6D,
        *,
        expected_scene_revision: str,
        new_scene_revision: str,
    ) -> SceneSnapshot:
        self._check_revision(expected_scene_revision)
        if not new_scene_revision.strip() or new_scene_revision == expected_scene_revision:
            raise ValueError("new_scene_revision must be non-empty and different")
        if pose.frame_id != self._snapshot.frame_id:
            raise ValueError("updated object pose frame does not match scene frame")

        updated: list[SceneObject] = []
        found = False
        for item in self._snapshot.objects:
            if item.object_id == object_id:
                updated.append(replace(item, pose=pose))
                found = True
            else:
                updated.append(item)
        if not found:
            raise KeyError(object_id)

        self._snapshot = replace(
            self._snapshot,
            scene_revision=new_scene_revision,
            objects=tuple(updated),
        )
        return self._snapshot

    def update_destination_pose(
        self,
        destination_id: str,
        pose: Pose6D,
        *,
        expected_scene_revision: str,
        new_scene_revision: str,
    ) -> SceneSnapshot:
        self._check_revision(expected_scene_revision)
        if not new_scene_revision.strip() or new_scene_revision == expected_scene_revision:
            raise ValueError("new_scene_revision must be non-empty and different")
        if pose.frame_id != self._snapshot.frame_id:
            raise ValueError("updated destination pose frame does not match scene frame")

        updated: list[DestinationRegion] = []
        found = False
        for item in self._snapshot.destinations:
            if item.destination_id == destination_id:
                updated.append(replace(item, pose=pose))
                found = True
            else:
                updated.append(item)
        if not found:
            raise KeyError(destination_id)

        self._snapshot = replace(
            self._snapshot,
            scene_revision=new_scene_revision,
            destinations=tuple(updated),
        )
        return self._snapshot
