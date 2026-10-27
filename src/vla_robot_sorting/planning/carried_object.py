"""Project-owned CuRobo carried-object planning support.

This module never edits the pinned RoboTwin/CuRobo configuration on disk. It
clones the Panthera robot configuration in memory, reserves collision spheres for
an attached object, and builds a dedicated MotionGen instance only on native Linux.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from math import isfinite
from pathlib import Path
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
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    CylinderGeometry,
    SceneSnapshot,
)
from vla_robot_sorting.planning.panthera_robotwin import (
    ROBOTWIN_WORLD_FRAME,
    _measured_arm_state,
    _normalized_wxyz,
    _numeric_rows,
    _rotation_matrix_wxyz,
    _transpose_rotate,
    pose6d_to_robotwin_target,
)
from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES

ATTACHED_OBJECT_LINK = "attached_object"
ATTACHED_OBJECT_JOINT = "attached_object_joint"
DEFAULT_ATTACHED_SPHERES = 100
DEFAULT_ATTACHMENT_CONTACT_LINKS = ("gripper_L_link", "gripper_R_link")


def _obb_z_extent_m(
    dims_xyz_m: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
) -> float:
    rotation = _rotation_matrix_wxyz(orientation_wxyz)
    half = tuple(value / 2.0 for value in dims_xyz_m)
    return sum(abs(rotation[2][index]) * half[index] for index in range(3))


def carried_support_contact_diagnostic(
    provenance: Mapping[str, Any],
    *,
    surface_sphere_radius_m: float,
) -> dict[str, Any]:
    """Compute conservative carried-object/table vertical separation in base frame."""

    if not isfinite(surface_sphere_radius_m) or surface_sphere_radius_m <= 0:
        raise ValueError("surface_sphere_radius_m must be finite and positive")
    object_pose = tuple(float(v) for v in provenance["object_base_pose_wxyz"])
    object_dims = tuple(float(v) for v in provenance["object_dims_xyz_m"])
    table_pose = tuple(float(v) for v in provenance["table_base_pose_wxyz"])
    table_dims = tuple(float(v) for v in provenance["table_dims_xyz_m"])
    if len(object_pose) != 7 or len(table_pose) != 7:
        raise ValueError("object/table base poses must contain seven values")
    if len(object_dims) != 3 or len(table_dims) != 3:
        raise ValueError("object/table dimensions must contain three values")

    object_extent_z = _obb_z_extent_m(
        object_dims,
        _normalized_wxyz(object_pose[3:], label="object base orientation"),
    )
    table_extent_z = _obb_z_extent_m(
        table_dims,
        _normalized_wxyz(table_pose[3:], label="table base orientation"),
    )
    object_bottom_z = object_pose[2] - object_extent_z
    table_top_z = table_pose[2] + table_extent_z
    signed_gap_m = object_bottom_z - table_top_z
    required_lift_to_positive_gap_m = max(0.0, -signed_gap_m)
    required_lift_to_sphere_clearance_m = max(
        0.0,
        surface_sphere_radius_m - signed_gap_m,
    )
    return {
        "object_bottom_z_base_m": object_bottom_z,
        "table_top_z_base_m": table_top_z,
        "signed_gap_m": signed_gap_m,
        "surface_sphere_radius_m": surface_sphere_radius_m,
        "required_lift_to_positive_gap_m": required_lift_to_positive_gap_m,
        "required_lift_to_sphere_clearance_m": required_lift_to_sphere_clearance_m,
        "conservative_overlap": signed_gap_m <= 0.0,
        "within_surface_sphere_radius": signed_gap_m <= surface_sphere_radius_m,
        "diagnostic_only": True,
    }


def _native_json_value(value: Any) -> Any:
    """Convert native CuRobo/Torch scalar diagnostics into JSON-safe values."""

    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _native_json_value(enum_value)
    detach = getattr(value, "detach", None)
    if callable(detach):
        tensor = detach().cpu()
        numel = getattr(tensor, "numel", None)
        if callable(numel) and int(numel()) == 1:
            return float(tensor.item())
        reshape = getattr(tensor, "reshape", None)
        tolist = getattr(reshape(-1) if callable(reshape) else tensor, "tolist", None)
        if callable(tolist):
            return tolist()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _motion_gen_result_summary(native_result: Any) -> dict[str, Any]:
    fields = (
        "valid_query",
        "status",
        "attempts",
        "trajopt_attempts",
        "used_graph",
        "position_error",
        "rotation_error",
        "cspace_error",
        "solve_time",
        "ik_time",
        "graph_time",
        "trajopt_time",
        "finetune_time",
        "total_time",
    )
    return {
        field: _native_json_value(getattr(native_result, field, None))
        for field in fields
    }


def prepare_carried_robot_config(
    raw_config: Mapping[str, Any],
    *,
    config_path: str | Path,
    attachment_parent_link: str,
    attachment_contact_links: tuple[str, ...] = DEFAULT_ATTACHMENT_CONTACT_LINKS,
    sphere_count: int = DEFAULT_ATTACHED_SPHERES,
    asset_path_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Clone a CuRobo robot YAML and reserve attached-object collision spheres."""

    if sphere_count < 1:
        raise ValueError("sphere_count must be positive")
    attachment_parent_link = attachment_parent_link.strip()
    if not attachment_parent_link:
        raise ValueError("attachment_parent_link must be non-empty")
    contact_links = tuple(
        dict.fromkeys(link.strip() for link in attachment_contact_links if link.strip())
    )
    if not contact_links:
        raise ValueError("attachment_contact_links must contain at least one link")

    config = deepcopy(dict(raw_config))
    robot_cfg = config.get("robot_cfg", config)
    if not isinstance(robot_cfg, dict):
        raise ValueError("robot configuration must contain a robot_cfg mapping")
    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError("robot configuration must contain kinematics")

    base = Path(config_path).resolve().parent
    overrides = dict(asset_path_overrides or {})
    for key in ("urdf_path", "collision_spheres"):
        override = overrides.get(key)
        if override is not None:
            resolved = Path(override).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"carried-object asset override for {key} does not exist: {resolved}"
                )
            kinematics[key] = str(resolved)
            continue

        value = kinematics.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            kinematics[key] = str((base / value).resolve())

    extra_links = kinematics.get("extra_links")
    if extra_links is None:
        extra_links = {}
    if not isinstance(extra_links, dict):
        raise ValueError("extra_links must be a mapping or null")
    extra_links = deepcopy(extra_links)
    desired_link = {
        "parent_link_name": attachment_parent_link,
        "link_name": ATTACHED_OBJECT_LINK,
        "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "joint_type": "FIXED",
        "joint_name": ATTACHED_OBJECT_JOINT,
    }
    existing_link = extra_links.get(ATTACHED_OBJECT_LINK)
    if existing_link is not None:
        if not isinstance(existing_link, dict):
            raise ValueError("existing attached_object extra link must be a mapping")
        for key, expected in desired_link.items():
            actual = existing_link.get(key)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"existing attached_object extra link conflicts at {key}: "
                    f"{actual!r} != {expected!r}"
                )
        desired_link = {**desired_link, **deepcopy(existing_link)}
    extra_links[ATTACHED_OBJECT_LINK] = desired_link
    kinematics["extra_links"] = extra_links

    collision_links = kinematics.get("collision_link_names")
    if collision_links is not None:
        if not isinstance(collision_links, list):
            raise ValueError("collision_link_names must be a list or null")
        collision_links = list(collision_links)
        if ATTACHED_OBJECT_LINK not in collision_links:
            collision_links.append(ATTACHED_OBJECT_LINK)
        kinematics["collision_link_names"] = collision_links

    self_collision_ignore = kinematics.get("self_collision_ignore")
    if self_collision_ignore is None:
        self_collision_ignore = {}
    if not isinstance(self_collision_ignore, dict):
        raise ValueError("self_collision_ignore must be a mapping or null")
    self_collision_ignore = deepcopy(self_collision_ignore)
    for link_name in (attachment_parent_link, *contact_links):
        ignored = self_collision_ignore.get(link_name)
        if ignored is None:
            ignored = []
        if not isinstance(ignored, list):
            raise ValueError(
                f"self_collision_ignore[{link_name!r}] must be a list or null"
            )
        ignored = list(ignored)
        if ATTACHED_OBJECT_LINK not in ignored:
            ignored.append(ATTACHED_OBJECT_LINK)
        self_collision_ignore[link_name] = ignored
    kinematics["self_collision_ignore"] = self_collision_ignore

    self_collision_buffer = kinematics.get("self_collision_buffer")
    if self_collision_buffer is None:
        self_collision_buffer = {}
    if not isinstance(self_collision_buffer, dict):
        raise ValueError("self_collision_buffer must be a mapping or null")
    self_collision_buffer = deepcopy(self_collision_buffer)
    self_collision_buffer.setdefault(ATTACHED_OBJECT_LINK, 0.0)
    kinematics["self_collision_buffer"] = self_collision_buffer

    extras = kinematics.get("extra_collision_spheres")
    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise ValueError("extra_collision_spheres must be a mapping or null")
    extras = deepcopy(extras)
    existing = extras.get(ATTACHED_OBJECT_LINK)
    if existing is not None and int(existing) < sphere_count:
        raise ValueError(
            "existing attached_object sphere capacity is below requested capacity"
        )
    extras[ATTACHED_OBJECT_LINK] = max(int(existing or 0), sphere_count)
    kinematics["extra_collision_spheres"] = extras
    return config


def _world_pose_to_base(
    robot: Any,
    position_world: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    origin = getattr(robot, "left_entity_origion_pose", None)
    if origin is None:
        raise ValueError("Panthera base pose is unavailable")
    base_position = tuple(float(value) for value in origin.p)
    base_q = _normalized_wxyz(origin.q, label="Panthera base orientation")
    if len(base_position) != 3 or not all(isfinite(value) for value in base_position):
        raise ValueError("Panthera base position must contain three finite values")

    relative = tuple(
        position_world[index] - base_position[index] for index in range(3)
    )
    rotation = _rotation_matrix_wxyz(base_q)
    position_base = _transpose_rotate(rotation, relative)

    bw, bx, by, bz = base_q
    ow, ox, oy, oz = orientation_wxyz
    conjugate = (bw, -bx, -by, -bz)
    cw, cx, cy, cz = conjugate
    q_base = _normalized_wxyz(
        (
            cw * ow - cx * ox - cy * oy - cz * oz,
            cw * ox + cx * ow + cy * oz - cz * oy,
            cw * oy - cx * oz + cy * ow + cz * ox,
            cw * oz + cx * oy - cy * ox + cz * ow,
        ),
        label="object base orientation",
    )
    return position_base, q_base


def carried_world_from_scene(
    robot: Any,
    scene: SceneSnapshot,
    *,
    object_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build table + carried-object CuRobo world in Panthera base coordinates."""

    if scene.frame_id != ROBOTWIN_WORLD_FRAME:
        raise ValueError("carried-object scene must use world frame")
    table = scene.metadata.get("curobo_table_collision")
    if not isinstance(table, Mapping):
        raise ValueError("scene is missing curobo_table_collision metadata")

    try:
        table_center = tuple(float(value) for value in table["center_xyz_m"])
        table_dims = tuple(float(value) for value in table["dims_xyz_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid table collision metadata") from exc
    table_q = _normalized_wxyz(
        table.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0)),
        label="table orientation",
    )
    table_position_base, table_q_base = _world_pose_to_base(
        robot,
        table_center,
        table_q,
    )

    target = scene.object(object_id)
    if target.pose.frame_id != ROBOTWIN_WORLD_FRAME:
        raise ValueError("carried object pose must use world frame")

    if isinstance(target.geometry, BoxGeometry):
        dims = tuple(float(value) for value in target.geometry.size_xyz_m)
        object_geometry = "box"
        geometry_provenance: dict[str, Any] = {
            "representation": "native_cuboid",
        }
    elif isinstance(target.geometry, CylinderGeometry):
        axis = tuple(float(value) for value in target.geometry.axis_local)
        axis_index = max(range(3), key=lambda index: abs(axis[index]))
        if (
            abs(abs(axis[axis_index]) - 1.0) > 1e-6
            or any(
                abs(axis[index]) > 1e-6
                for index in range(3)
                if index != axis_index
            )
        ):
            raise ValueError(
                "carried cylinder axis must align with one local principal axis"
            )
        diameter = 2.0 * float(target.geometry.radius_m)
        dims_list = [diameter, diameter, diameter]
        dims_list[axis_index] = float(target.geometry.height_m)
        dims = tuple(dims_list)
        object_geometry = "cylinder"
        geometry_provenance = {
            "representation": "conservative_cuboid_envelope",
            "cylinder_axis_local": axis,
            "cylinder_radius_m": float(target.geometry.radius_m),
            "cylinder_height_m": float(target.geometry.height_m),
        }
    else:
        raise ValueError("unsupported carried-object geometry")

    qx, qy, qz, qw = target.pose.orientation_xyzw
    object_q_world = _normalized_wxyz(
        (qw, qx, qy, qz),
        label="carried object orientation",
    )
    object_position_base, object_q_base = _world_pose_to_base(
        robot,
        target.pose.position_xyz,
        object_q_world,
    )
    world = {
        "cuboid": {
            "table": {
                "dims": list(table_dims),
                "pose": [*table_position_base, *table_q_base],
            },
            object_id: {
                "dims": list(dims),
                "pose": [*object_position_base, *object_q_base],
            },
        }
    }
    provenance = {
        "object_id": object_id,
        "object_geometry": object_geometry,
        "object_dims_xyz_m": dims,
        "object_geometry_provenance": geometry_provenance,
        "object_base_pose_wxyz": (*object_position_base, *object_q_base),
        "table_base_pose_wxyz": (*table_position_base, *table_q_base),
        "table_dims_xyz_m": table_dims,
        "scene_revision": scene.scene_revision,
        "transform_revision": scene.transform_revision,
    }
    return world, provenance


class PantheraRoboTwinCarriedObjectPlanner:
    """CuRobo-only planner with collision geometry for the grasped payload."""

    def __init__(
        self,
        robot: Any,
        *,
        config_path: str | Path,
        object_id: str,
        attachment_parent_link: str,
        attachment_contact_links: tuple[str, ...] = DEFAULT_ATTACHMENT_CONTACT_LINKS,
        sphere_count: int = DEFAULT_ATTACHED_SPHERES,
        surface_sphere_radius_m: float = 0.003,
        planner_id: str = "robotwin-curobo-carried-object",
        start_state_tolerance_rad: float = 0.02,
        clock: Callable[[], float] = perf_counter,
        motion_gen_factory: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
        asset_path_overrides: Mapping[str, str | Path] | None = None,
    ) -> None:
        if not object_id.strip():
            raise ValueError("object_id must be non-empty")
        if sphere_count < 1:
            raise ValueError("sphere_count must be positive")
        if not isfinite(surface_sphere_radius_m) or surface_sphere_radius_m <= 0:
            raise ValueError("surface_sphere_radius_m must be finite and positive")
        attachment_parent_link = attachment_parent_link.strip()
        if not attachment_parent_link:
            raise ValueError("attachment_parent_link must be non-empty")
        contact_links = tuple(
            dict.fromkeys(
                link.strip() for link in attachment_contact_links if link.strip()
            )
        )
        if not contact_links:
            raise ValueError("attachment_contact_links must contain at least one link")

        self._robot = robot
        self._config_path = Path(config_path)
        self._object_id = object_id
        self._attachment_parent_link = attachment_parent_link
        self._attachment_contact_links = contact_links
        self._sphere_count = sphere_count
        self._surface_sphere_radius_m = surface_sphere_radius_m
        self._planner_id = planner_id
        self._start_state_tolerance_rad = start_state_tolerance_rad
        self._clock = clock
        self._motion_gen_factory = motion_gen_factory or self._native_motion_gen_factory
        self._asset_path_overrides = {
            key: Path(value).resolve()
            for key, value in dict(asset_path_overrides or {}).items()
        }
        self._motion_gen: Any | None = None
        self._provenance: dict[str, Any] | None = None
        self._attached = False

    def initialize_attachment(
        self,
        start_state: JointState,
        scene: SceneSnapshot,
    ) -> dict[str, Any]:
        if self._attached:
            raise RuntimeError("carried object is already attached")
        self._validate_start(start_state)
        raw = self._load_yaml()
        config = prepare_carried_robot_config(
            raw,
            config_path=self._config_path,
            attachment_parent_link=self._attachment_parent_link,
            attachment_contact_links=self._attachment_contact_links,
            sphere_count=self._sphere_count,
            asset_path_overrides=self._asset_path_overrides,
        )
        world, provenance = carried_world_from_scene(
            self._robot,
            scene,
            object_id=self._object_id,
        )
        motion_gen = self._motion_gen_factory(config, world)
        attached = self._attach_native(motion_gen, start_state)
        if not attached:
            raise RuntimeError("CuRobo refused carried-object attachment")
        self._motion_gen = motion_gen
        self._attached = True
        support_contact_geometry = carried_support_contact_diagnostic(
            provenance,
            surface_sphere_radius_m=self._surface_sphere_radius_m,
        )
        self._provenance = {
            **provenance,
            "support_contact_geometry": support_contact_geometry,
            "attached_object_link": ATTACHED_OBJECT_LINK,
            "attachment_parent_link": self._attachment_parent_link,
            "attachment_contact_links": list(self._attachment_contact_links),
            "attachment_joint_name": ATTACHED_OBJECT_JOINT,
            "reserved_sphere_count": self._sphere_count,
            "surface_sphere_radius_m": self._surface_sphere_radius_m,
            "world_obstacle_removed_on_attach": True,
            "config_source": str(self._config_path),
            "asset_path_overrides": {
                key: str(value) for key, value in self._asset_path_overrides.items()
            },
            "pinned_config_mutated_on_disk": False,
        }
        return dict(self._provenance)

    def detach(self) -> None:
        if self._motion_gen is not None and self._attached:
            detach = getattr(self._motion_gen, "detach_object_from_robot", None)
            if not callable(detach):
                detach = getattr(self._motion_gen, "detach", None)
            if not callable(detach):
                raise RuntimeError("CuRobo detach API is unavailable")
            detach(link_name=ATTACHED_OBJECT_LINK)
        self._attached = False

    def plan(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: Any,
    ) -> PlannerResult:
        del scene
        started = self._clock()
        if not self._attached or self._motion_gen is None:
            return self._failure(
                started,
                MotionFailureCode.BACKEND_UNAVAILABLE,
                "carried-object planner is not initialized/attached",
            )
        try:
            self._validate_start(start_state)
            target = pose6d_to_robotwin_target(goal)
            result = self._plan_native(self._motion_gen, start_state, target)
        except ValueError as exc:
            return self._failure(
                started,
                MotionFailureCode.INVALID_GOAL,
                str(exc),
            )
        except Exception as exc:
            return self._failure(
                started,
                MotionFailureCode.BACKEND_UNAVAILABLE,
                f"carried-object CuRobo failed: {type(exc).__name__}: {exc}",
            )

        if not isinstance(result, Mapping):
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                "carried-object CuRobo returned a non-mapping result",
            )
        if str(result.get("status")) != "Success":
            native_status = result.get("native_status") or result.get("status")
            return self._failure(
                started,
                MotionFailureCode.OPTIMIZATION_FAILED,
                f"carried-object CuRobo status={native_status}",
                details={
                    "native_status": native_status,
                    "native_diagnostics": result.get("native_diagnostics"),
                },
            )
        try:
            positions = _numeric_rows(
                result["position"],
                label="carried-object position",
            )
        except (KeyError, ValueError) as exc:
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                f"invalid carried-object path: {exc}",
            )
        if any(len(row) != len(PANTHERA_JOINT_NAMES) for row in positions):
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                "carried-object path must contain six-joint rows",
            )

        prepended_start = positions[0] != start_state.positions
        if prepended_start:
            positions = (start_state.positions, *positions)

        velocities: tuple[tuple[float, ...], ...] | None = None
        if "velocity" in result:
            try:
                parsed_velocities = _numeric_rows(
                    result["velocity"],
                    label="carried-object velocity",
                )
                if prepended_start and len(parsed_velocities) == len(positions) - 1:
                    velocities = (
                        (0.0,) * len(PANTHERA_JOINT_NAMES),
                        *parsed_velocities,
                    )
                elif len(parsed_velocities) == len(positions):
                    velocities = parsed_velocities
            except ValueError:
                velocities = None
        if velocities is None:
            return self._failure(
                started,
                MotionFailureCode.INTERNAL_ERROR,
                "carried-object path is missing aligned vendor velocity samples",
                details={
                    "native_status": result.get("native_status"),
                    "native_diagnostics": result.get("native_diagnostics"),
                    "position_samples": len(positions),
                },
            )

        metadata = {
            "scene_revision": context.scene_revision,
            "transform_revision": context.transform_revision,
            "carried_object": dict(self._provenance or {}),
            "vendor_status": "Success",
            "vendor_sample_count": len(positions),
            "vendor_interpolation_dt_s": 1.0 / 250.0,
            "vendor_velocity_available": True,
            "vendor_velocity_samples_rad_s": velocities,
            "native_diagnostics": result.get("native_diagnostics"),
        }
        path = JointPath(
            joint_names=PANTHERA_JOINT_NAMES,
            positions=positions,
            source=self._planner_id,
            metadata=metadata,
        )
        return PlannerResult(
            path=path,
            failure=None,
            planning_time_s=max(0.0, self._clock() - started),
            planner_id=self._planner_id,
        )

    def _validate_start(self, start_state: JointState) -> None:
        if start_state.joint_names != PANTHERA_JOINT_NAMES:
            raise ValueError("start_state joint order does not match Panthera")
        measured = _measured_arm_state(self._robot)
        error = max(
            abs(expected - actual)
            for expected, actual in zip(start_state.positions, measured, strict=True)
        )
        if error > self._start_state_tolerance_rad:
            raise ValueError("start_state is stale relative to simulator qpos")

    def _load_yaml(self) -> dict[str, Any]:
        import yaml

        with self._config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Panthera CuRobo YAML must contain a mapping")
        return raw

    def _native_motion_gen_factory(
        self,
        config: dict[str, Any],
        world: dict[str, Any],
    ) -> Any:
        # The carried-object MotionGen is built *after* native task setup;
        # apply the same in-memory strict interior bounds here separately.
        # This does not modify the pinned YAML, URDF or actual robot bounds.
        from copy import deepcopy

        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        from vla_robot_sorting.planning.native_curobo_interior import (
            CUROBO_POSITION_LIMIT_CLIP_RAD,
            INTERIOR_RETRACT_CONFIG_RAD,
            PRIMARY_IK_SEEDS,
        )

        clipped_config = deepcopy(config)
        cspace = clipped_config["robot_cfg"]["kinematics"]["cspace"]
        expected_names = [f"joint{i}" for i in range(1, 7)]
        if cspace.get("joint_names") != expected_names:
            raise ValueError("unexpected carried CuRobo joint order")
        if cspace.get("position_limit_clip", 0.0) != 0.0:
            raise ValueError("unexpected carried CuRobo position clip")
        cspace["position_limit_clip"] = CUROBO_POSITION_LIMIT_CLIP_RAD
        if cspace.get("retract_config") not in (None, [0.0] * 6):
            raise ValueError("unexpected carried CuRobo retract config")
        cspace["retract_config"] = list(INTERIOR_RETRACT_CONFIG_RAD)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            clipped_config,
            world,
            interpolation_dt=1.0 / 250.0,
            num_ik_seeds=PRIMARY_IK_SEEDS,
            num_trajopt_seeds=1,
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()
        return motion_gen

    def _native_joint_state(self, start_state: JointState) -> Any:
        import torch
        from curobo.types.state import JointState as CuroboJointState

        q = torch.tensor(start_state.positions, dtype=torch.float32).cuda().reshape(1, -1)
        return CuroboJointState.from_position(
            q,
            joint_names=list(PANTHERA_JOINT_NAMES),
        )

    def _native_start_diagnostic(self, motion_gen: Any, native_state: Any) -> dict[str, Any]:
        check = getattr(motion_gen, "check_start_state", None)
        if not callable(check):
            return {
                "available": False,
                "valid": None,
                "status": None,
            }
        valid, status = check(native_state)
        return {
            "available": True,
            "valid": bool(valid),
            "status": _native_json_value(status),
        }

    def diagnose_start_state(self, start_state: JointState) -> dict[str, Any]:
        """Inspect post-attachment MotionGen start validity without moving the robot."""

        if not self._attached or self._motion_gen is None:
            return {
                "available": False,
                "valid": None,
                "status": "planner_not_attached",
            }
        self._validate_start(start_state)
        return self._native_start_diagnostic(
            self._motion_gen,
            self._native_joint_state(start_state),
        )

    def diagnose_plan_with_obstacle_disabled(
        self,
        start_state: JointState,
        goal: MotionGoal,
        *,
        obstacle_name: str,
    ) -> dict[str, Any]:
        """Plan for diagnosis with one world obstacle disabled; never execute the path."""

        diagnostic: dict[str, Any] = {
            "planning_only": True,
            "executed": False,
            "obstacle_name": obstacle_name,
            "obstacle_disabled": False,
            "obstacle_reenabled": False,
            "plan_success": False,
        }
        if not self._attached or self._motion_gen is None:
            diagnostic["error"] = "planner_not_attached"
            return diagnostic
        self._validate_start(start_state)
        checker = getattr(self._motion_gen, "world_coll_checker", None)
        enable_obstacle = getattr(checker, "enable_obstacle", None)
        if not callable(enable_obstacle):
            diagnostic["error"] = "world_collision_checker_enable_obstacle_unavailable"
            return diagnostic

        get_names = getattr(checker, "get_obstacle_names", None)
        if callable(get_names):
            try:
                diagnostic["world_obstacle_names"] = list(get_names())
            except Exception as exc:
                diagnostic["world_obstacle_names_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

        disabled = False
        try:
            enable_obstacle(obstacle_name, enable=False)
            disabled = True
            diagnostic["obstacle_disabled"] = True
            diagnostic["start_state"] = self.diagnose_start_state(start_state)
            target = pose6d_to_robotwin_target(goal)
            raw = self._plan_native(self._motion_gen, start_state, target)
            diagnostic["native_status"] = raw.get("native_status") or raw.get("status")
            diagnostic["native_diagnostics"] = raw.get("native_diagnostics")
            diagnostic["plan_success"] = raw.get("status") == "Success"
            positions = raw.get("position")
            diagnostic["path_samples"] = len(positions) if positions is not None else 0
        except Exception as exc:
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if disabled:
                try:
                    enable_obstacle(obstacle_name, enable=True)
                    diagnostic["obstacle_reenabled"] = True
                except Exception as exc:
                    diagnostic["reenable_error"] = f"{type(exc).__name__}: {exc}"
        return diagnostic

    def plan_support_escape(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: Any,
        *,
        obstacle_name: str,
        requested_escape_m: float,
        max_escape_m: float = 0.01,
    ) -> tuple[PlannerResult, dict[str, Any]]:
        """Plan a bounded support-contact escape with one obstacle disabled for planning.

        The obstacle is always re-enabled before the returned path may be executed.
        This method never executes motion.
        """

        started = self._clock()
        diagnostic: dict[str, Any] = {
            "planning_only": True,
            "executed": False,
            "obstacle_name": obstacle_name,
            "obstacle_disabled": False,
            "obstacle_reenabled": False,
            "requested_escape_m": float(requested_escape_m),
            "max_escape_m": float(max_escape_m),
            "bounded": False,
            "plan_success": False,
        }
        if (
            not isfinite(requested_escape_m)
            or requested_escape_m <= 0.0
            or not isfinite(max_escape_m)
            or max_escape_m <= 0.0
            or requested_escape_m > max_escape_m
        ):
            diagnostic["error"] = "escape_distance_out_of_bounds"
            return (
                self._failure(
                    started,
                    MotionFailureCode.INVALID_GOAL,
                    "support escape distance is outside the bounded policy",
                    details=diagnostic,
                ),
                diagnostic,
            )
        diagnostic["bounded"] = True

        if not self._attached or self._motion_gen is None:
            diagnostic["error"] = "planner_not_attached"
            return (
                self._failure(
                    started,
                    MotionFailureCode.BACKEND_UNAVAILABLE,
                    "carried-object planner is not initialized/attached",
                    details=diagnostic,
                ),
                diagnostic,
            )

        checker = getattr(self._motion_gen, "world_coll_checker", None)
        enable_obstacle = getattr(checker, "enable_obstacle", None)
        if not callable(enable_obstacle):
            diagnostic["error"] = "world_collision_checker_enable_obstacle_unavailable"
            return (
                self._failure(
                    started,
                    MotionFailureCode.BACKEND_UNAVAILABLE,
                    "world collision checker cannot toggle support obstacle",
                    details=diagnostic,
                ),
                diagnostic,
            )

        disabled = False
        planned: PlannerResult | None = None
        try:
            enable_obstacle(obstacle_name, enable=False)
            disabled = True
            diagnostic["obstacle_disabled"] = True
            diagnostic["start_state"] = self.diagnose_start_state(start_state)
            planned = self.plan(start_state, goal, context, scene)
            diagnostic["plan_success"] = planned.succeeded
            diagnostic["planner_id"] = planned.planner_id
            if planned.failure is not None:
                diagnostic["failure_code"] = planned.failure.code.value
                diagnostic["failure_message"] = planned.failure.message
                diagnostic["failure_details"] = dict(planned.failure.details)
            if planned.path is not None:
                diagnostic["path_samples"] = len(planned.path.positions)
        except Exception as exc:
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if disabled:
                try:
                    enable_obstacle(obstacle_name, enable=True)
                    diagnostic["obstacle_reenabled"] = True
                except Exception as exc:
                    diagnostic["reenable_error"] = f"{type(exc).__name__}: {exc}"

        if diagnostic["obstacle_reenabled"] is not True:
            return (
                self._failure(
                    started,
                    MotionFailureCode.BACKEND_UNAVAILABLE,
                    "support obstacle was not re-enabled after escape planning",
                    details=diagnostic,
                ),
                diagnostic,
            )
        if planned is None:
            return (
                self._failure(
                    started,
                    MotionFailureCode.BACKEND_UNAVAILABLE,
                    "support escape planning did not return a result",
                    details=diagnostic,
                ),
                diagnostic,
            )
        if not planned.succeeded or planned.path is None:
            return planned, diagnostic

        metadata = dict(planned.path.metadata)
        metadata["support_escape_planning"] = dict(diagnostic)
        path = JointPath(
            joint_names=planned.path.joint_names,
            positions=planned.path.positions,
            source=planned.path.source,
            metadata=metadata,
        )
        return (
            PlannerResult(
                path=path,
                failure=None,
                planning_time_s=planned.planning_time_s,
                planner_id=planned.planner_id,
            ),
            diagnostic,
        )

    def _attach_native(self, motion_gen: Any, start_state: JointState) -> bool:
        native_state = self._native_joint_state(start_state)
        return bool(
            motion_gen.attach_objects_to_robot(
                native_state,
                [self._object_id],
                surface_sphere_radius=self._surface_sphere_radius_m,
                link_name=ATTACHED_OBJECT_LINK,
                remove_obstacles_from_world_config=True,
            )
        )

    def _plan_native(
        self,
        motion_gen: Any,
        start_state: JointState,
        target_gripper_pose: tuple[float, ...],
    ) -> Mapping[str, Any]:
        import numpy as np
        from curobo.types.math import Pose as CuroboPose
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        end_link_pose = self._robot._trans_from_gripper_to_endlink(
            target_gripper_pose,
            arm_tag="left",
        )
        vendor = self._robot.left_planner
        world_base_pose = np.concatenate(
            [
                np.asarray(vendor.robot_origion_pose.p, dtype=float),
                np.asarray(vendor.robot_origion_pose.q, dtype=float),
            ]
        )
        world_target_pose = np.concatenate(
            [
                np.asarray(end_link_pose.p, dtype=float),
                np.asarray(end_link_pose.q, dtype=float),
            ]
        )
        position, quaternion = vendor._trans_from_world_to_base(
            world_base_pose,
            world_target_pose,
        )
        position = np.asarray(position, dtype=float) + np.asarray(
            vendor.frame_bias,
            dtype=float,
        )
        goal_pose = CuroboPose.from_list([*position.tolist(), *quaternion.tolist()])
        native_start = self._native_joint_state(start_state)
        start_diagnostic = self._native_start_diagnostic(motion_gen, native_start)
        native_result = motion_gen.plan_single(
            native_start,
            goal_pose,
            MotionGenPlanConfig(max_attempts=10),
        )
        native_diagnostics = _motion_gen_result_summary(native_result)
        native_diagnostics["start_state"] = start_diagnostic
        native_status = native_diagnostics.get("status")
        if not bool(native_result.success.item()):
            return {
                "status": "Fail",
                "native_status": native_status,
                "native_diagnostics": native_diagnostics,
            }
        return {
            "status": "Success",
            "native_status": native_status,
            "native_diagnostics": native_diagnostics,
            "position": native_result.interpolated_plan.position.detach().cpu().numpy(),
            "velocity": native_result.interpolated_plan.velocity.detach().cpu().numpy(),
        }

    def _failure(
        self,
        started: float,
        code: MotionFailureCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> PlannerResult:
        failure_details = {"object_id": self._object_id}
        if details is not None:
            failure_details.update(dict(details))
        return PlannerResult(
            path=None,
            failure=MotionFailure(
                code=code,
                message=message,
                recoverable=True,
                details=failure_details,
            ),
            planning_time_s=max(0.0, self._clock() - started),
            planner_id=self._planner_id,
        )
