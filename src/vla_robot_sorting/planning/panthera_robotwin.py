"""Project-owned Panthera/RoboTwin motion-planner adapter.

The vendor planner consumes world-frame target poses in [x, y, z, qw, qx, qy, qz]
order. Project Pose6D uses xyzw, so the conversion is explicit here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite, sqrt
from time import perf_counter
from typing import Any

from vla_robot_sorting.contracts.motion import (
    JointPath,
    JointState,
    MotionFailure,
    MotionFailureCode,
    MotionGoal,
    PlannerResult,
    PlanningContext,
)
from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES

ROBOTWIN_WORLD_FRAME = "world"


def pose6d_to_robotwin_target(goal: MotionGoal) -> tuple[float, ...]:
    """Convert project xyzw quaternion convention to RoboTwin wxyz."""

    pose = goal.pose
    if pose.frame_id != ROBOTWIN_WORLD_FRAME:
        raise ValueError(
            f"Panthera/RoboTwin planner requires frame_id={ROBOTWIN_WORLD_FRAME!r}, "
            f"got {pose.frame_id!r}"
        )
    x, y, z = pose.position_xyz
    qx, qy, qz, qw = pose.orientation_xyzw
    return (x, y, z, qw, qx, qy, qz)


def _numeric_rows(values: Any, *, label: str) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(float(item) for item in row) for row in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a 2D numeric sequence") from exc
    if not rows:
        raise ValueError(f"{label} must not be empty")
    for row in rows:
        if len(row) != len(PANTHERA_JOINT_NAMES):
            raise ValueError(f"{label} rows must contain six Panthera joints")
        if not all(isfinite(item) for item in row):
            raise ValueError(f"{label} contains non-finite values")
    return rows


def _measured_arm_state(robot: Any) -> tuple[float, ...]:
    raw = robot.get_left_arm_real_jointState()
    if not isinstance(raw, Sequence) or len(raw) != 7:
        raise ValueError("RoboTwin real joint state must contain six joints plus gripper")
    values = tuple(float(item) for item in raw[:6])
    if not all(isfinite(item) for item in values):
        raise ValueError("RoboTwin real joint state contains non-finite values")
    return values


def _normalized_wxyz(values: Any, *, label: str) -> tuple[float, float, float, float]:
    try:
        q = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain four numeric values") from exc
    if len(q) != 4 or not all(isfinite(item) for item in q):
        raise ValueError(f"{label} must contain four finite values")
    norm = sqrt(sum(item * item for item in q))
    if norm <= 0.0:
        raise ValueError(f"{label} quaternion norm must be positive")
    return tuple(item / norm for item in q)


def _wxyz_conjugate(q: tuple[float, float, float, float]) -> tuple[float, ...]:
    return (q[0], -q[1], -q[2], -q[3])


def _wxyz_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotation_matrix_wxyz(
    q: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = q
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _transpose_rotate(
    rotation: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(rotation[row][column] * vector[row] for row in range(3))
        for column in range(3)
    )


def curobo_table_world_from_scene(
    robot: Any,
    scene: Any,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Build an exact tabletop cuboid in the Panthera base frame.

    RoboTwin's pinned CuRobo wrapper hard-codes the table from the robot's world-y
    coordinate and therefore places the table on the wrong side of a +90 degree
    yawed Panthera base. Native independent collision evidence exposed the mismatch
    as camera_link <-> table contact on an otherwise CuRobo-successful path.
    """

    metadata = getattr(scene, "metadata", None)
    if metadata is None and isinstance(scene, Mapping):
        metadata = scene.get("metadata")
    if not isinstance(metadata, Mapping):
        return None

    table = metadata.get("curobo_table_collision")
    if table is None:
        return None
    if not isinstance(table, Mapping):
        raise ValueError("curobo_table_collision metadata must be a mapping")
    if table.get("frame_id") != ROBOTWIN_WORLD_FRAME:
        raise ValueError("curobo_table_collision must use world frame")

    try:
        center_world = tuple(float(value) for value in table["center_xyz_m"])
        dims = tuple(float(value) for value in table["dims_xyz_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid CuRobo tabletop center/dimensions metadata") from exc
    if (
        len(center_world) != 3
        or len(dims) != 3
        or not all(isfinite(value) for value in center_world)
        or not all(isfinite(value) and value > 0.0 for value in dims)
    ):
        raise ValueError("CuRobo tabletop center/dimensions must be finite and positive")

    table_q = _normalized_wxyz(
        table.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0)),
        label="table orientation",
    )
    origin = getattr(robot, "left_entity_origion_pose", None)
    if origin is None:
        raise ValueError("Panthera robot origin pose is unavailable for CuRobo world sync")
    try:
        base_world = tuple(float(value) for value in origin.p)
        base_q = _normalized_wxyz(origin.q, label="Panthera base orientation")
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("invalid Panthera robot origin pose") from exc
    if len(base_world) != 3 or not all(isfinite(value) for value in base_world):
        raise ValueError("Panthera base position must contain three finite values")

    relative_world = tuple(
        center_world[index] - base_world[index] for index in range(3)
    )
    base_rotation = _rotation_matrix_wxyz(base_q)
    center_base = _transpose_rotate(base_rotation, relative_world)
    table_base_q = _normalized_wxyz(
        _wxyz_multiply(_wxyz_conjugate(base_q), table_q),
        label="table base orientation",
    )
    world_dict = {
        "cuboid": {
            "table": {
                "dims": list(dims),
                "pose": [*center_base, *table_base_q],
            }
        }
    }
    provenance = {
        "source": str(table.get("source", "scene metadata")),
        "table_world_center_xyz_m": center_world,
        "table_world_orientation_wxyz": table_q,
        "table_dims_xyz_m": dims,
        "robot_base_world_xyz_m": base_world,
        "robot_base_world_orientation_wxyz": base_q,
        "table_base_pose_wxyz": (*center_base, *table_base_q),
    }
    return world_dict, provenance


def _sync_curobo_world(robot: Any, scene: Any) -> dict[str, Any] | None:
    prepared = curobo_table_world_from_scene(robot, scene)
    if prepared is None:
        return None
    world_dict, provenance = prepared

    planner = getattr(robot, "left_planner", None)
    if planner is None:
        raise RuntimeError("RoboTwin CuRobo planner object is unavailable")
    signature = repr(world_dict)
    if getattr(planner, "_vla_table_world_signature", None) == signature:
        return {
            **provenance,
            "synchronized": True,
            "cache_hit": True,
            "updated_motion_generators": (),
        }

    from curobo.geom.types import WorldConfig

    world = WorldConfig.from_dict(world_dict)
    updated: list[str] = []
    for name in ("motion_gen", "motion_gen_batch"):
        motion_gen = getattr(planner, name, None)
        if motion_gen is None:
            continue
        update_world = getattr(motion_gen, "update_world", None)
        if not callable(update_world):
            continue
        update_world(world)
        updated.append(name)
    if not updated:
        raise RuntimeError("RoboTwin CuRobo MotionGen update_world API is unavailable")
    planner._vla_table_world_signature = signature
    return {
        **provenance,
        "synchronized": True,
        "cache_hit": False,
        "updated_motion_generators": tuple(updated),
    }


class PantheraRoboTwinMotionPlanner:
    """Translate live single-arm RoboTwin planning into project PlannerResult."""

    def __init__(
        self,
        robot: Any,
        *,
        planner_id: str = "robotwin-curobo",
        start_state_tolerance_rad: float = 0.02,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not planner_id.strip():
            raise ValueError("planner_id must be non-empty")
        if start_state_tolerance_rad < 0 or not isfinite(start_state_tolerance_rad):
            raise ValueError("start_state_tolerance_rad must be finite and non-negative")
        self._robot = robot
        self._planner_id = planner_id
        self._start_state_tolerance_rad = start_state_tolerance_rad
        self._clock = clock


    def plan(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: Any,
    ) -> PlannerResult:
        started = self._clock()

        if start_state.joint_names != PANTHERA_JOINT_NAMES:
            return self._failure(
                started,
                MotionFailureCode.INVALID_GOAL,
                "start_state joint order does not match Panthera joint1..joint6",
                recoverable=False,
            )

        try:
            target = pose6d_to_robotwin_target(goal)
            measured = _measured_arm_state(self._robot)
        except ValueError as exc:
            return self._failure(
                started,
                MotionFailureCode.INVALID_GOAL,
                str(exc),
                recoverable=False,
            )

        max_start_error = max(
            abs(expected - actual)
            for expected, actual in zip(start_state.positions, measured, strict=True)
        )
        if max_start_error > self._start_state_tolerance_rad:
            return self._failure(
                started,
                MotionFailureCode.EXECUTION_STALE,
                "planner start_state is stale relative to simulator articulation qpos",
                recoverable=True,
                details={"max_start_error_rad": max_start_error},
            )

        try:
            world_sync = _sync_curobo_world(self._robot, scene)
        except Exception as exc:  # native CuRobo boundary
            return self._failure(
                started,
                MotionFailureCode.BACKEND_UNAVAILABLE,
                f"CuRobo world synchronization failed: {type(exc).__name__}: {exc}",
                recoverable=True,
                details={"scene_revision": context.scene_revision},
            )

        search_policy = None
        try:
            if getattr(self._robot, "_vla_native_curobo_search", False):
                from vla_robot_sorting.planning.native_curobo_interior import (
                    scoped_native_curobo_search,
                )

                with scoped_native_curobo_search(self._robot) as policy:
                    result = self._robot.left_plan_path(target)
                search_policy = dict(policy)
                if search_policy.get("factory_invocations") != 1:
                    raise RuntimeError("pinned CuRobo plan config not applied exactly once")
            else:
                result = self._robot.left_plan_path(target)
        except Exception as exc:  # vendor backend boundary
            return self._failure(
                started,
                MotionFailureCode.BACKEND_UNAVAILABLE,
                f"RoboTwin planner raised {type(exc).__name__}: {exc}",
                recoverable=True,
            )

        if not isinstance(result, Mapping):
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                "RoboTwin planner returned a non-mapping result",
                recoverable=True,
            )

        status = str(result.get("status", "missing"))
        if status != "Success":
            return self._failure(
                started,
                MotionFailureCode.OPTIMIZATION_FAILED,
                f"RoboTwin planner status={status}",
                recoverable=True,
                details={"vendor_status": status, "native_search_policy": search_policy},
            )

        try:
            positions = _numeric_rows(result["position"], label="planner position")
        except (KeyError, ValueError) as exc:
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                f"invalid RoboTwin planner result: {exc}",
                recoverable=True,
            )

        prepended_start = positions[0] != start_state.positions
        if prepended_start:
            positions = (start_state.positions, *positions)

        velocities: tuple[tuple[float, ...], ...] | None = None
        if "velocity" in result:
            try:
                parsed_velocities = _numeric_rows(result["velocity"], label="planner velocity")
                if prepended_start and len(parsed_velocities) == len(positions) - 1:
                    velocities = ((0.0,) * len(PANTHERA_JOINT_NAMES), *parsed_velocities)
                elif len(parsed_velocities) == len(positions):
                    velocities = parsed_velocities
            except ValueError:
                velocities = None

        path_metadata: dict[str, Any] = {
            "scene_revision": context.scene_revision,
            "transform_revision": context.transform_revision,
            "target_pose_robotwin_wxyz": target,
            "vendor_status": status,
            "native_search_policy": search_policy,
            "vendor_sample_count": len(positions),
            "vendor_interpolation_dt_s": 1.0 / 250.0,
            "vendor_velocity_available": velocities is not None,
            "vendor_velocity_samples_rad_s": velocities,
        }
        if world_sync is not None:
            path_metadata["curobo_world_sync"] = world_sync

        path = JointPath(
            joint_names=PANTHERA_JOINT_NAMES,
            positions=positions,
            source=self._planner_id,
            metadata=path_metadata,
        )
        return PlannerResult(
            path=path,
            failure=None,
            planning_time_s=max(0.0, self._clock() - started),
            planner_id=self._planner_id,
        )


    def _failure(
        self,
        started: float,
        code: MotionFailureCode,
        message: str,
        *,
        recoverable: bool,
        details: dict[str, Any] | None = None,
    ) -> PlannerResult:
        return PlannerResult(
            path=None,
            failure=MotionFailure(
                code=code,
                message=message,
                recoverable=recoverable,
                details=details or {},
            ),
            planning_time_s=max(0.0, self._clock() - started),
            planner_id=self._planner_id,
        )


def _extract_panthera_rows(values: Any, robot: Any, *, label: str) -> tuple[tuple[float, ...], ...]:
    """Extract joint1..joint6 from an MPlib result that may include gripper joints."""

    try:
        raw_rows = tuple(tuple(float(item) for item in row) for row in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a 2D numeric sequence") from exc
    if not raw_rows:
        raise ValueError(f"{label} must not be empty")

    if all(len(row) == len(PANTHERA_JOINT_NAMES) for row in raw_rows):
        rows = raw_rows
    else:
        active_names = tuple(
            joint.get_name() for joint in robot.left_entity.get_active_joints()
        )
        try:
            indices = tuple(active_names.index(name) for name in PANTHERA_JOINT_NAMES)
        except ValueError as exc:
            raise ValueError("Panthera arm joints missing from active joint order") from exc
        if any(max(indices) >= len(row) for row in raw_rows):
            raise ValueError(f"{label} rows do not match active joint order")
        rows = tuple(tuple(row[index] for index in indices) for row in raw_rows)

    if not all(isfinite(item) for row in rows for item in row):
        raise ValueError(f"{label} contains non-finite values")
    return rows


class PantheraRoboTwinMPlibPlanner:
    """Source-audited MPlib RRT fallback; native compatibility remains unvalidated."""

    def __init__(
        self,
        robot: Any,
        *,
        planner_id: str = "robotwin-mplib-rrt",
        start_state_tolerance_rad: float = 0.02,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not planner_id.strip():
            raise ValueError("planner_id must be non-empty")
        if start_state_tolerance_rad < 0 or not isfinite(start_state_tolerance_rad):
            raise ValueError("start_state_tolerance_rad must be finite and non-negative")
        self._robot = robot
        self._planner_id = planner_id
        self._start_state_tolerance_rad = start_state_tolerance_rad
        self._clock = clock

    def plan(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: Any,
    ) -> PlannerResult:
        del scene
        started = self._clock()


        if start_state.joint_names != PANTHERA_JOINT_NAMES:
            return self._failure(
                started,
                MotionFailureCode.INVALID_GOAL,
                "start_state joint order does not match Panthera joint1..joint6",
                recoverable=False,
            )

        try:
            target = pose6d_to_robotwin_target(goal)
            measured = _measured_arm_state(self._robot)
        except ValueError as exc:
            return self._failure(
                started,
                MotionFailureCode.INVALID_GOAL,
                str(exc),
                recoverable=False,
            )

        max_start_error = max(
            abs(expected - actual)
            for expected, actual in zip(start_state.positions, measured, strict=True)
        )
        if max_start_error > self._start_state_tolerance_rad:
            return self._failure(
                started,
                MotionFailureCode.EXECUTION_STALE,
                "planner start_state is stale relative to simulator articulation qpos",
                recoverable=True,
                details={"max_start_error_rad": max_start_error},
            )


        try:
            end_link_pose = self._robot._trans_from_gripper_to_endlink(target, arm_tag="left")
            full_qpos = self._robot.left_entity.get_qpos()
            result = self._robot.left_mplib_planner.plan_pose(
                full_qpos,
                end_link_pose,
                arms_tag="left",
                try_times=10,
                log=False,
            )
        except Exception as exc:  # vendor backend boundary
            return self._failure(
                started,
                MotionFailureCode.BACKEND_UNAVAILABLE,
                f"RoboTwin MPlib fallback raised {type(exc).__name__}: {exc}",
                recoverable=True,
            )

        if not isinstance(result, Mapping):
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                "RoboTwin MPlib returned a non-mapping result",
                recoverable=True,
            )
        status = str(result.get("status", "missing"))
        if status != "Success":
            return self._failure(
                started,
                MotionFailureCode.OPTIMIZATION_FAILED,
                f"RoboTwin MPlib status={status}",
                recoverable=True,
                details={"vendor_status": status},
            )


        try:
            positions = _extract_panthera_rows(
                result["position"],
                self._robot,
                label="MPlib position",
            )
        except (KeyError, ValueError) as exc:
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                f"invalid RoboTwin MPlib result: {exc}",
                recoverable=True,
            )

        prepended_start = positions[0] != start_state.positions
        if prepended_start:
            positions = (start_state.positions, *positions)

        velocities: tuple[tuple[float, ...], ...] | None = None
        if "velocity" in result:
            try:
                parsed = _extract_panthera_rows(
                    result["velocity"],
                    self._robot,
                    label="MPlib velocity",
                )
                if prepended_start and len(parsed) == len(positions) - 1:
                    velocities = ((0.0,) * len(PANTHERA_JOINT_NAMES), *parsed)
                elif len(parsed) == len(positions):
                    velocities = parsed
            except ValueError:
                velocities = None


        path = JointPath(
            joint_names=PANTHERA_JOINT_NAMES,
            positions=positions,
            source=self._planner_id,
            metadata={
                "scene_revision": context.scene_revision,
                "transform_revision": context.transform_revision,
                "target_pose_robotwin_wxyz": target,
                "vendor_status": status,
                "vendor_sample_count": len(positions),
                "vendor_interpolation_dt_s": 1.0 / 250.0,
                "vendor_velocity_available": velocities is not None,
                "vendor_velocity_samples_rad_s": velocities,
                "fallback_backend": True,
            },
        )
        return PlannerResult(
            path=path,
            failure=None,
            planning_time_s=max(0.0, self._clock() - started),
            planner_id=self._planner_id,
        )

    def _failure(
        self,
        started: float,
        code: MotionFailureCode,
        message: str,
        *,
        recoverable: bool,
        details: dict[str, Any] | None = None,
    ) -> PlannerResult:
        return PlannerResult(
            path=None,
            failure=MotionFailure(
                code=code,
                message=message,
                recoverable=recoverable,
                details=details or {},
            ),
            planning_time_s=max(0.0, self._clock() - started),
            planner_id=self._planner_id,
        )


class PantheraRoboTwinMPlibScrewPlanner(PantheraRoboTwinMPlibPlanner):
    """Target-aware straight screw-motion planner for bounded grasp acquisition."""

    def __init__(
        self,
        robot: Any,
        *,
        planner_id: str = "robotwin-mplib-screw",
        start_state_tolerance_rad: float = 0.02,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        super().__init__(
            robot,
            planner_id=planner_id,
            start_state_tolerance_rad=start_state_tolerance_rad,
            clock=clock,
        )
        self.last_provenance: dict[str, Any] | None = None

    def plan(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: Any,
    ) -> PlannerResult:
        del scene
        started = self._clock()
        self.last_provenance = {
            "strategy": "mplib_screw_acquisition",
            "wrapper_method": "left_mplib_planner.plan_screw",
            "target_aware_planning_world_required": True,
            "plan_success": False,
        }

        if start_state.joint_names != PANTHERA_JOINT_NAMES:
            return self._failure(
                started,
                MotionFailureCode.INVALID_GOAL,
                "start_state joint order does not match Panthera joint1..joint6",
                recoverable=False,
            )

        try:
            target = pose6d_to_robotwin_target(goal)
            measured = _measured_arm_state(self._robot)
        except ValueError as exc:
            return self._failure(
                started,
                MotionFailureCode.INVALID_GOAL,
                str(exc),
                recoverable=False,
            )

        max_start_error = max(
            abs(expected - actual)
            for expected, actual in zip(start_state.positions, measured, strict=True)
        )
        if max_start_error > self._start_state_tolerance_rad:
            return self._failure(
                started,
                MotionFailureCode.EXECUTION_STALE,
                "planner start_state is stale relative to simulator articulation qpos",
                recoverable=True,
                details={"max_start_error_rad": max_start_error},
            )

        try:
            end_link_pose = self._robot._trans_from_gripper_to_endlink(
                target,
                arm_tag="left",
            )
            full_qpos = self._robot.left_entity.get_qpos()
            plan_screw = getattr(self._robot.left_mplib_planner, "plan_screw", None)
            if not callable(plan_screw):
                raise AttributeError("left_mplib_planner.plan_screw is unavailable")
            result = plan_screw(
                full_qpos,
                end_link_pose,
                arms_tag="left",
                log=False,
            )
        except Exception as exc:  # vendor/native boundary
            self.last_provenance["backend_error"] = f"{type(exc).__name__}: {exc}"
            return self._failure(
                started,
                MotionFailureCode.BACKEND_UNAVAILABLE,
                f"RoboTwin MPlib screw planner raised {type(exc).__name__}: {exc}",
                recoverable=True,
            )

        if not isinstance(result, Mapping):
            self.last_provenance["vendor_status"] = "non-mapping"
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                "RoboTwin MPlib screw planner returned a non-mapping result",
                recoverable=True,
            )

        status = str(result.get("status", "missing"))
        self.last_provenance["vendor_status"] = status
        if status != "Success":
            return self._failure(
                started,
                MotionFailureCode.OPTIMIZATION_FAILED,
                f"RoboTwin MPlib screw planner status={status}",
                recoverable=True,
                details={"vendor_status": status},
            )

        try:
            positions = _extract_panthera_rows(
                result["position"],
                self._robot,
                label="MPlib screw position",
            )
        except (KeyError, ValueError) as exc:
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                f"invalid RoboTwin MPlib screw result: {exc}",
                recoverable=True,
            )

        prepended_start = positions[0] != start_state.positions
        if prepended_start:
            positions = (start_state.positions, *positions)

        velocities: tuple[tuple[float, ...], ...] | None = None
        if "velocity" in result:
            try:
                parsed = _extract_panthera_rows(
                    result["velocity"],
                    self._robot,
                    label="MPlib screw velocity",
                )
                if prepended_start and len(parsed) == len(positions) - 1:
                    velocities = ((0.0,) * len(PANTHERA_JOINT_NAMES), *parsed)
                elif len(parsed) == len(positions):
                    velocities = parsed
            except ValueError:
                velocities = None

        self.last_provenance.update(
            {
                "plan_success": True,
                "vendor_status": status,
                "selected_planner_id": self._planner_id,
                "straight_screw_motion": True,
                "independent_collision_recheck_required": True,
            }
        )

        path = JointPath(
            joint_names=PANTHERA_JOINT_NAMES,
            positions=positions,
            source=self._planner_id,
            metadata={
                "scene_revision": context.scene_revision,
                "transform_revision": context.transform_revision,
                "target_pose_robotwin_wxyz": target,
                "vendor_status": status,
                "vendor_sample_count": len(positions),
                "vendor_interpolation_dt_s": 1.0 / 250.0,
                "vendor_velocity_available": velocities is not None,
                "vendor_velocity_samples_rad_s": velocities,
                "acquisition_strategy": "mplib_screw_acquisition",
                "target_aware_planning_world_required": True,
                "independent_collision_recheck_required": True,
            },
        )
        return PlannerResult(
            path=path,
            failure=None,
            planning_time_s=max(0.0, self._clock() - started),
            planner_id=self._planner_id,
        )


class PantheraRoboTwinMPlibAcquisitionPlanner:
    """Target-aware grasp acquisition: screw first, MPlib RRT only as safe fallback."""

    def __init__(
        self,
        robot: Any,
        *,
        screw_planner: PantheraRoboTwinMPlibScrewPlanner | None = None,
        rrt_planner: PantheraRoboTwinMPlibPlanner | None = None,
    ) -> None:
        self._screw = screw_planner or PantheraRoboTwinMPlibScrewPlanner(robot)
        self._rrt = rrt_planner or PantheraRoboTwinMPlibPlanner(robot)
        self.last_provenance: dict[str, Any] | None = None

    def plan(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: Any,
    ) -> PlannerResult:
        primary = self._screw.plan(start_state, goal, context, scene)
        primary_policy = getattr(self._screw, "last_provenance", None)
        primary_failure = primary.failure
        primary_vendor_status = None
        if primary_failure is not None:
            primary_vendor_status = primary_failure.details.get("vendor_status")

        provenance: dict[str, Any] = {
            "strategy": "mplib_target_aware_acquisition",
            "target_aware_planning_world_required": True,
            "independent_collision_recheck_required": True,
            "primary_planner_id": "robotwin-mplib-screw",
            "primary_success": primary.succeeded,
            "primary_failure_code": (
                primary_failure.code.value if primary_failure is not None else None
            ),
            "primary_vendor_status": primary_vendor_status,
            "primary_policy": (
                dict(primary_policy) if isinstance(primary_policy, dict) else None
            ),
            "fallback_allowed": False,
            "fallback_attempted": False,
            "fallback_planner_id": None,
            "fallback_wrapper_method": "left_mplib_planner.plan_pose",
            "fallback_target_aware_planning_world_required": True,
            "fallback_success": None,
            "selected_planner_id": primary.planner_id,
            "plan_success": primary.succeeded,
        }
        if primary.succeeded:
            self.last_provenance = provenance
            return primary

        fallback_allowed = (
            primary_failure is not None
            and primary_failure.code is MotionFailureCode.OPTIMIZATION_FAILED
            and primary_failure.recoverable
        )
        provenance["fallback_allowed"] = fallback_allowed
        if not fallback_allowed:
            self.last_provenance = provenance
            return primary

        fallback = self._rrt.plan(start_state, goal, context, scene)
        provenance.update(
            {
                "fallback_attempted": True,
                "fallback_planner_id": "robotwin-mplib-rrt",
                "fallback_success": fallback.succeeded,
                "selected_planner_id": fallback.planner_id,
                "plan_success": fallback.succeeded,
            }
        )
        if fallback.failure is not None:
            provenance["fallback_failure_code"] = fallback.failure.code.value
            provenance["fallback_vendor_status"] = fallback.failure.details.get(
                "vendor_status"
            )
        self.last_provenance = provenance
        return fallback
