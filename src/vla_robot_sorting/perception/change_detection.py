"""Deterministic scene-change detection for S1.5 closed-loop recovery."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, sqrt

from vla_robot_sorting.contracts.perception import ObjectVisibility, SceneSnapshot
from vla_robot_sorting.contracts.recovery import SceneChange, SceneChangeKind


@dataclass(frozen=True, slots=True)
class SceneChangeThresholds:
    position_m: float = 0.015
    orientation_rad: float = 0.12

    def __post_init__(self) -> None:
        if self.position_m < 0 or self.orientation_rad < 0:
            raise ValueError("scene-change thresholds must be non-negative")


def _position_delta(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _orientation_delta(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * acos(dot)


class SceneChangeDetector:
    def __init__(self, thresholds: SceneChangeThresholds | None = None) -> None:
        self.thresholds = thresholds or SceneChangeThresholds()

    def detect(
        self,
        previous: SceneSnapshot,
        current: SceneSnapshot,
        *,
        target_object_id: str,
        destination_id: str,
    ) -> SceneChange:
        if previous.transform_revision != current.transform_revision:
            return SceneChange(
                SceneChangeKind.TRANSFORM_CHANGED,
                previous.scene_revision,
                current.scene_revision,
                target_object_id,
                destination_id,
            )

        try:
            current_target = current.object(target_object_id)
        except KeyError:
            return SceneChange(
                SceneChangeKind.TARGET_MISSING,
                previous.scene_revision,
                current.scene_revision,
                target_object_id,
                destination_id,
            )

        if current_target.visibility is ObjectVisibility.NOT_PRESENT:
            return SceneChange(
                SceneChangeKind.TARGET_MISSING,
                previous.scene_revision,
                current.scene_revision,
                target_object_id,
                destination_id,
            )
        if current_target.visibility is ObjectVisibility.OCCLUDED:
            return SceneChange(
                SceneChangeKind.TARGET_OCCLUDED,
                previous.scene_revision,
                current.scene_revision,
                target_object_id,
                destination_id,
            )

        try:
            previous_target = previous.object(target_object_id)
        except KeyError:
            previous_target = current_target

        pos_delta = _position_delta(
            previous_target.pose.position_xyz,
            current_target.pose.position_xyz,
        )
        ori_delta = _orientation_delta(
            previous_target.pose.orientation_xyzw,
            current_target.pose.orientation_xyzw,
        )
        if (
            pos_delta > self.thresholds.position_m
            or ori_delta > self.thresholds.orientation_rad
        ):
            return SceneChange(
                SceneChangeKind.TARGET_MOVED,
                previous.scene_revision,
                current.scene_revision,
                target_object_id,
                destination_id,
                pos_delta,
                ori_delta,
            )


        try:
            previous_destination = previous.destination(destination_id)
            current_destination = current.destination(destination_id)
        except KeyError:
            return SceneChange(
                SceneChangeKind.DESTINATION_MISSING,
                previous.scene_revision,
                current.scene_revision,
                target_object_id,
                destination_id,
            )

        destination_pos_delta = _position_delta(
            previous_destination.pose.position_xyz,
            current_destination.pose.position_xyz,
        )
        destination_ori_delta = _orientation_delta(
            previous_destination.pose.orientation_xyzw,
            current_destination.pose.orientation_xyzw,
        )
        if (
            destination_pos_delta > self.thresholds.position_m
            or destination_ori_delta > self.thresholds.orientation_rad
        ):
            return SceneChange(
                SceneChangeKind.DESTINATION_MOVED,
                previous.scene_revision,
                current.scene_revision,
                target_object_id,
                destination_id,
                destination_pos_delta,
                destination_ori_delta,
            )

        return SceneChange(
            SceneChangeKind.NONE,
            previous.scene_revision,
            current.scene_revision,
            target_object_id,
            destination_id,
            0.0,
            0.0,
        )
