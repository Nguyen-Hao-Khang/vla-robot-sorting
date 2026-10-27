"""Compatibility timing adapter for RoboTwin/CuRobo interpolated plans.

This preserves the vendor's source-audited 250 Hz interpolation for bring-up.
It is not the final TOPPRA decision and remains replaceable behind the project
TrajectoryParameterizer contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

from vla_robot_sorting.contracts.motion import (
    JointLimits,
    JointPath,
    Trajectory,
    TrajectoryWaypoint,
)


class RoboTwinVendorTrajectoryParameterizer:
    """Convert a source-audited RoboTwin interpolated path into a Trajectory."""

    def parameterize(self, path: JointPath, limits: JointLimits) -> Trajectory:
        if path.joint_names != limits.joint_names:
            raise ValueError("path/limits joint order mismatch")

        dt = self._sample_period(path.metadata)
        velocities = self._velocities(path.metadata, len(path.positions), len(path.joint_names))


        waypoints = tuple(
            TrajectoryWaypoint(
                time_from_start_s=index * dt,
                positions=position,
                velocities=velocity,
            )
            for index, (position, velocity) in enumerate(
                zip(path.positions, velocities, strict=True)
            )
        )
        return Trajectory(
            joint_names=path.joint_names,
            waypoints=waypoints,
            source="robotwin-vendor-interpolated",
            limits_used=limits,
            interpolation="robotwin-curobo-interpolated-250hz",
            scene_revision=_optional_text(path.metadata.get("scene_revision")),
            transform_revision=_optional_text(path.metadata.get("transform_revision")),
            metadata={
                "geometric_path_source": path.source,
                "timing_source": "RoboTwin CuroboPlanner MotionGen interpolation",
                "controller_period_s": dt,
                "compatibility_bridge": True,
            },
        )


    @staticmethod
    def _sample_period(metadata: dict[str, Any]) -> float:
        value = metadata.get("vendor_interpolation_dt_s")
        try:
            dt = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("path is missing vendor_interpolation_dt_s") from exc
        if not isfinite(dt) or dt <= 0:
            raise ValueError("vendor_interpolation_dt_s must be finite and positive")
        return dt

    @staticmethod
    def _velocities(
        metadata: dict[str, Any],
        expected_rows: int,
        dof: int,
    ) -> tuple[tuple[float, ...], ...]:
        raw = metadata.get("vendor_velocity_samples_rad_s")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("path is missing aligned vendor velocity samples")
        try:
            rows = tuple(tuple(float(value) for value in row) for row in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("vendor velocity samples must be numeric") from exc
        if len(rows) != expected_rows or any(len(row) != dof for row in rows):
            raise ValueError("vendor velocity samples do not align with path positions")
        if any(not isfinite(value) for row in rows for value in row):
            raise ValueError("vendor velocity samples contain non-finite values")
        return rows


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None
