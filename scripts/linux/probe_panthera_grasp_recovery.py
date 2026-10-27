#!/usr/bin/env python3
"""Native non-actuating B5 grasp-feasibility + B7 stale-recovery probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--task-config", default="move_pillbottle_pad_panthera_single")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--runs", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xyzw_from_wxyz(q: Any) -> tuple[float, float, float, float]:
    qw, qx, qy, qz = (float(value) for value in q)
    return (qx, qy, qz, qw)


def max_radial_distance_from_vertices(
    vertices_xyz_m: Any,
    *,
    center_xyz_m: tuple[float, float, float],
    long_axis_index: int,
) -> float:
    """Return a conservative cylinder radius around one local principal axis."""

    from math import hypot, isfinite

    if long_axis_index not in (0, 1, 2):
        raise ValueError("long_axis_index must be 0, 1, or 2")
    radial_indices = tuple(index for index in range(3) if index != long_axis_index)
    radius = 0.0
    seen = False
    for raw in vertices_xyz_m:
        vertex = tuple(float(value) for value in raw)
        if len(vertex) != 3 or not all(isfinite(value) for value in vertex):
            raise ValueError("mesh vertices must contain three finite coordinates")
        seen = True
        radius = max(
            radius,
            hypot(
                vertex[radial_indices[0]] - center_xyz_m[radial_indices[0]],
                vertex[radial_indices[1]] - center_xyz_m[radial_indices[1]],
            ),
        )
    if not seen or radius <= 0.0 or not isfinite(radius):
        raise ValueError("mesh vertices did not produce a positive radial radius")
    return radius


def actor_box_from_collision_mesh(task: Any, robotwin_root: Path) -> dict[str, Any]:
    import numpy as np
    import transforms3d as t3d
    import trimesh
    from envs.utils.create_actor import get_glb_or_obj_file

    model_id = int(task.pillbottle_id)
    modeldir = robotwin_root / "assets" / "objects" / "080_pillbottle"
    collision_dir = modeldir / "collision"
    source_dir = collision_dir if collision_dir.is_dir() else modeldir
    collision_path = Path(get_glb_or_obj_file(source_dir, model_id)).resolve()
    if not collision_path.is_file():
        raise FileNotFoundError(collision_path)

    config = task.pillbottle.config
    if not isinstance(config, dict) or "scale" not in config:
        raise RuntimeError("pillbottle model_data scale is required for B5 provenance")
    scale = np.asarray(config["scale"], dtype=float)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise RuntimeError(f"invalid model_data scale: {scale}")

    with collision_path.open("rb") as handle:
        loaded = trimesh.load(handle, file_type=collision_path.suffix.lstrip("."))
    if isinstance(loaded, trimesh.Scene):
        try:
            collision_mesh = loaded.dump(concatenate=True)
        except TypeError:
            collision_mesh = trimesh.util.concatenate(tuple(loaded.dump()))
    else:
        collision_mesh = loaded
    bounds = np.asarray(collision_mesh.bounds, dtype=float)
    vertices = np.asarray(collision_mesh.vertices, dtype=float)
    if bounds.shape != (2, 3):
        raise RuntimeError(f"unexpected collision mesh bounds shape: {bounds.shape}")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise RuntimeError(f"unexpected collision mesh vertices shape: {vertices.shape}")
    local_center = (bounds[0] + bounds[1]) * 0.5 * scale
    extents = (bounds[1] - bounds[0]) * scale
    long_axis_index = int(np.argmax(extents))
    scaled_vertices = vertices * scale
    mesh_max_radial_distance_m = max_radial_distance_from_vertices(
        scaled_vertices,
        center_xyz_m=tuple(float(value) for value in local_center),
        long_axis_index=long_axis_index,
    )

    actor_pose = task.pillbottle.get_pose()
    rotation = t3d.quaternions.quat2mat(actor_pose.q)
    world_center = np.asarray(actor_pose.p, dtype=float) + rotation @ local_center
    return {
        "model_id": model_id,
        "collision_mesh": str(collision_path),
        "collision_mesh_sha256": sha256_file(collision_path),
        "model_scale": scale.tolist(),
        "mesh_bounds_unscaled": bounds.tolist(),
        "mesh_vertex_count": len(vertices),
        "mesh_long_axis_local_index": long_axis_index,
        "mesh_max_radial_distance_m": mesh_max_radial_distance_m,
        "local_center_scaled_m": local_center.tolist(),
        "size_xyz_m": extents.tolist(),
        "world_center_m": world_center.tolist(),
        "actor_pose_wxyz": [
            *[float(v) for v in actor_pose.p],
            *[float(v) for v in actor_pose.q],
        ],
    }


def symmetric_parallel_jaw_aperture_from_joint_scale(
    scale: object,
) -> tuple[float, float]:
    """Convert Panthera per-finger base-joint travel to relative jaw aperture.

    RoboTwin drives gripper_L_joint over [0, 0.04] m while gripper_R_joint
    mirrors it with multiplier -1. The relative finger-to-finger travel is
    therefore 2*q, i.e. [0, 0.08] m for the canonical Panthera model.
    """
    if not isinstance(scale, list) or len(scale) != 2:
        raise ValueError("Panthera gripper_scale must contain two values")
    joint_min, joint_max = (float(scale[0]), float(scale[1]))
    if joint_min < 0.0 or joint_max <= joint_min:
        raise ValueError(f"invalid Panthera gripper_scale: {scale!r}")
    return 2.0 * joint_min, 2.0 * joint_max


def load_gripper_proxy(
    robotwin_root: Path,
    task_args: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import yaml

    from vla_robot_sorting.contracts.grasping import ParallelJawGripper

    embodiment = task_args["left_embodiment_config"]
    scale = embodiment.get("gripper_scale")
    if not isinstance(scale, list) or len(scale) != 2:
        raise RuntimeError("Panthera gripper_scale must contain closed/open metres")
    per_finger_min, per_finger_max = (float(scale[0]), float(scale[1]))
    min_width, max_width = symmetric_parallel_jaw_aperture_from_joint_scale(scale)

    # Pinned RoboTwin Base_Task.get_grasp_pose() defines the task gripper frame
    # 120 mm behind the marked contact point. Panthera grasp_actor() replaces the
    # default target_dis=0 with PANTHERA_DEFAULT_GRASP_DIS=-0.02 m, so the final
    # gripper-frame target remains 100 mm behind the contact center. The task's
    # pre_grasp_dis=0.06 m is measured before applying target_dis, therefore the
    # actual pregrasp-to-grasp target separation is 0.06 - (-0.02) = 0.08 m.
    task_grasp_frame_to_contact_m = 0.12
    task_pre_grasp_dis_m = 0.06
    panthera_default_grasp_dis_m = -0.02
    contact_center_offset_m = (
        task_grasp_frame_to_contact_m + panthera_default_grasp_dis_m
    )
    pregrasp_standoff_m = (
        task_pre_grasp_dis_m - panthera_default_grasp_dis_m
    )

    sphere_path = robotwin_root / "assets/embodiments/panthera-6dof/collision_panthera.yml"
    with sphere_path.open(encoding="utf-8") as handle:
        spheres = yaml.safe_load(handle)["collision_spheres"]
    finger_radii = [
        float(item["radius"])
        for link in ("gripper_L_link", "gripper_R_link")
        for item in spheres[link]
    ]
    finger_cross_section = 2.0 * max(finger_radii)

    gripper = ParallelJawGripper(
        min_width_m=min_width,
        max_width_m=max_width,
        finger_thickness_m=finger_cross_section,
        finger_height_m=finger_cross_section,
        pregrasp_standoff_m=pregrasp_standoff_m,
        jaw_clearance_m=0.0,
        contact_center_offset_m=contact_center_offset_m,
    )
    provenance = {
        "min_width_m": min_width,
        "max_width_m": max_width,
        "finger_proxy_diameter_m": finger_cross_section,
        "task_pre_grasp_dis_m": task_pre_grasp_dis_m,
        "pregrasp_standoff_m": pregrasp_standoff_m,
        "contact_center_offset_m": contact_center_offset_m,
        "task_grasp_frame_to_contact_m": task_grasp_frame_to_contact_m,
        "panthera_default_grasp_dis_m": panthera_default_grasp_dis_m,
        "per_finger_joint_scale_m": [per_finger_min, per_finger_max],
        "relative_aperture_multiplier": 2.0,
        "width_source": (
            "Panthera symmetric prismatic jaws: relative aperture = "
            "2 * base-joint travel; right joint mimics left with -1 multiplier"
        ),
        "finger_proxy_source": str(sphere_path),
        "standoff_source": (
            "canonical move_pillbottle_pad pre_grasp_dis=0.06 minus "
            "Panthera target_dis=-0.02 gives 0.08m target separation"
        ),
        "contact_offset_source": (
            "pinned RoboTwin Base_Task.get_grasp_pose contact-0.12m and "
            "Panthera PANTHERA_DEFAULT_GRASP_DIS=-0.02m"
        ),
    }
    return gripper, provenance


def collision_bounds_geometry_proxy(
    size_xyz_m: tuple[float, float, float],
    *,
    mesh_max_radial_distance_m: float | None = None,
    mesh_long_axis_local_index: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Choose a conservative primitive from collision-mesh measurements."""

    from math import isfinite

    from vla_robot_sorting.contracts.perception import BoxGeometry, CylinderGeometry

    size = tuple(float(value) for value in size_xyz_m)
    if len(size) != 3 or any(not isfinite(value) or value <= 0.0 for value in size):
        raise ValueError("collision-mesh AABB size must contain three positive values")

    long_axis_index = max(range(3), key=lambda index: size[index])
    if (
        mesh_long_axis_local_index is not None
        and int(mesh_long_axis_local_index) != long_axis_index
    ):
        raise ValueError("mesh long-axis provenance disagrees with collision AABB")
    radial_indices = tuple(index for index in range(3) if index != long_axis_index)
    height_m = size[long_axis_index]
    radial_sizes = tuple(size[index] for index in radial_indices)
    radial_max = max(radial_sizes)
    radial_min = min(radial_sizes)
    elongation_ratio = height_m / radial_max
    radial_aspect_ratio = radial_max / radial_min
    mesh_radius = (
        float(mesh_max_radial_distance_m)
        if mesh_max_radial_distance_m is not None
        else None
    )
    mesh_radius_valid = (
        mesh_radius is not None and isfinite(mesh_radius) and mesh_radius > 0.0
    )

    if (
        elongation_ratio >= 1.5
        and radial_aspect_ratio <= 1.15
        and mesh_radius_valid
    ):
        axis = [0.0, 0.0, 0.0]
        axis[long_axis_index] = 1.0
        geometry = CylinderGeometry(
            radius_m=mesh_radius,
            height_m=height_m,
            axis_local=tuple(axis),
        )
        return geometry, {
            "primitive": "cylinder",
            "fit_rule": (
                "elongation>=1.5 and radial_aspect<=1.15 with "
                "mesh-vertex radial envelope"
            ),
            "long_axis_local_index": long_axis_index,
            "axis_local": tuple(axis),
            "radius_m": mesh_radius,
            "height_m": height_m,
            "elongation_ratio": elongation_ratio,
            "radial_aspect_ratio": radial_aspect_ratio,
            "conservative_radius_rule": (
                "maximum scaled collision-mesh vertex radial distance "
                "from AABB-center principal axis"
            ),
        }

    return BoxGeometry(size), {
        "primitive": "box",
        "fit_rule": (
            "fallback collision-mesh AABB; cylinder requires elongated near-round "
            "AABB plus mesh-vertex radial envelope"
        ),
        "size_xyz_m": size,
        "long_axis_local_index": long_axis_index,
        "elongation_ratio": elongation_ratio,
        "radial_aspect_ratio": radial_aspect_ratio,
        "mesh_max_radial_distance_m": mesh_radius,
    }


def make_scene_snapshot(
    task: Any,
    *,
    revision: str,
    geometry_info: dict[str, Any],
    transform_revision: str,
) -> Any:
    import transforms3d as t3d

    from vla_robot_sorting.contracts.models import Pose6D
    from vla_robot_sorting.contracts.perception import (
        DestinationRegion,
        GeometrySource,
        SceneObject,
        SceneSnapshot,
    )

    actor_pose = task.pillbottle.get_pose()
    pad_pose = task.pad.get_pose()
    table_pose = task.table.get_pose()
    table_local_center = (0.0, 0.0, -0.025)
    table_rotation = t3d.quaternions.quat2mat(table_pose.q)
    table_world_center = tuple(
        float(table_pose.p[index])
        + sum(
            float(table_rotation[index, axis]) * table_local_center[axis]
            for axis in range(3)
        )
        for index in range(3)
    )
    geometry, geometry_proxy = collision_bounds_geometry_proxy(
        tuple(float(v) for v in geometry_info["size_xyz_m"]),
        mesh_max_radial_distance_m=float(
            geometry_info["mesh_max_radial_distance_m"]
        ),
        mesh_long_axis_local_index=int(
            geometry_info["mesh_long_axis_local_index"]
        ),
    )
    target = SceneObject(
        object_id="pillbottle",
        pose=Pose6D(
            tuple(float(v) for v in geometry_info["world_center_m"]),
            xyzw_from_wxyz(actor_pose.q),
            "world",
        ),
        geometry=geometry,
        geometry_source=GeometrySource.SIM_GROUND_TRUTH,
        semantic_labels=("pill bottle", "pillbottle"),
        identity_confidence=1.0,
        metadata={
            "model_id": geometry_info["model_id"],
            "collision_mesh": geometry_info["collision_mesh"],
            "collision_mesh_sha256": geometry_info["collision_mesh_sha256"],
            "model_scale": geometry_info["model_scale"],
            "mesh_bounds_unscaled": geometry_info["mesh_bounds_unscaled"],
            "geometry_method": "primitive fit from scaled local collision-mesh AABB",
            "geometry_proxy": geometry_proxy,
        },
    )
    destination = DestinationRegion(
        destination_id="pad",
        pose=Pose6D(
            tuple(float(v) for v in pad_pose.p),
            xyzw_from_wxyz(pad_pose.q),
            "world",
        ),
        tolerance_xyz_m=(0.04, 0.04, 0.005),
        semantic_labels=("blue pad", "pad"),
    )
    return SceneSnapshot(
        scene_revision=revision,
        transform_revision=transform_revision,
        frame_id="world",
        objects=(target,),
        destinations=(destination,),
        table_height_m=float(table_pose.p[2]),
        metadata={
            "source": "canonical RoboTwin runtime ground truth",
            "curobo_table_collision": {
                "source": (
                    "pinned RoboTwin create_table_and_wall: "
                    "length=1.2,width=0.7,thickness=0.05"
                ),
                "frame_id": "world",
                "center_xyz_m": table_world_center,
                "orientation_wxyz": tuple(float(value) for value in table_pose.q),
                "dims_xyz_m": (1.2, 0.7, 0.05),
            },
        },
    )


def candidate_dict(candidate: Any) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "object_id": candidate.object_id,
        "grasp_position_xyz": candidate.grasp_pose.position_xyz,
        "grasp_orientation_xyzw": candidate.grasp_pose.orientation_xyzw,
        "pregrasp_position_xyz": candidate.pregrasp_pose.position_xyz,
        "pregrasp_orientation_xyzw": candidate.pregrasp_pose.orientation_xyzw,
        "approach_direction": candidate.approach_direction,
        "closing_direction": candidate.closing_direction,
        "jaw_width_m": candidate.jaw_width_m,
        "source": candidate.source,
        "geometry_source": candidate.geometry_source,
        "scene_revision": candidate.scene_revision,
        "transform_revision": candidate.transform_revision,
        "raw_quality": candidate.raw_quality,
        "metadata": candidate.metadata,
    }


def current_arm_state(task: Any) -> Any:
    from vla_robot_sorting.contracts import JointState
    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES

    actual = tuple(float(v) for v in task.robot.get_left_arm_real_jointState())
    if len(actual) != 7:
        raise RuntimeError("expected six arm joints plus gripper")
    return JointState(joint_names=PANTHERA_JOINT_NAMES, positions=actual[:6])


def expand_arm_qpos(task: Any, position: Any) -> Any:
    import numpy as np

    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES

    active_names = tuple(joint.get_name() for joint in task.robot.left_entity.get_active_joints())
    indices = tuple(active_names.index(name) for name in PANTHERA_JOINT_NAMES)
    state = np.asarray(task.robot.left_entity.get_qpos(), dtype=float).copy()
    for joint_index, value in zip(indices, position, strict=True):
        state[joint_index] = float(value)
    return state


def _planning_failure_payload(plan: Any) -> dict[str, Any] | None:
    failure = getattr(plan, "failure", None)
    if failure is None:
        return None
    code = getattr(failure, "code", None)
    code_value = getattr(code, "value", code)
    return {
        "code": str(code_value) if code_value is not None else "missing",
        "message": str(getattr(failure, "message", "missing")),
    }


def plan_primary_fallback(
    primary_planner: Any,
    fallback_planner: Any,
    start_state: Any,
    goal: Any,
    context: Any,
    *,
    scene: Any,
) -> tuple[Any, dict[str, Any]]:
    """Plan once with the audited CuRobo-primary/MPlib-fallback policy."""

    primary = primary_planner.plan(start_state, goal, context, scene=scene)
    fallback = None
    selected = primary
    if not primary.succeeded:
        fallback = fallback_planner.plan(start_state, goal, context, scene=scene)
        selected = fallback

    planning_time_s = float(primary.planning_time_s)
    if fallback is not None:
        planning_time_s += float(fallback.planning_time_s)

    provenance: dict[str, Any] = {
        "primary_planner_id": primary.planner_id,
        "primary_success": bool(primary.succeeded),
        "fallback_attempted": fallback is not None,
        "fallback_planner_id": fallback.planner_id if fallback is not None else None,
        "fallback_success": bool(fallback.succeeded) if fallback is not None else None,
        "selected_planner_id": selected.planner_id,
        "plan_success": bool(selected.succeeded),
        "planning_time_s": planning_time_s,
    }
    primary_failure = _planning_failure_payload(primary)
    if primary_failure is not None:
        provenance["primary_failure"] = primary_failure
    if fallback is not None:
        fallback_failure = _planning_failure_payload(fallback)
        if fallback_failure is not None:
            provenance["fallback_failure"] = fallback_failure
    selected_path = getattr(selected, "path", None)
    selected_metadata = getattr(selected_path, "metadata", None)
    if isinstance(selected_metadata, dict):
        world_sync = selected_metadata.get("curobo_world_sync")
        if world_sync is not None:
            provenance["selected_curobo_world_sync"] = world_sync
    return selected, provenance


def plan_panthera_primary_fallback(
    task: Any,
    start_state: Any,
    goal: Any,
    context: Any,
    *,
    scene: Any,
) -> tuple[Any, dict[str, Any]]:
    from vla_robot_sorting.planning import (
        PantheraRoboTwinMotionPlanner,
        PantheraRoboTwinMPlibPlanner,
    )

    return plan_primary_fallback(
        PantheraRoboTwinMotionPlanner(task.robot),
        PantheraRoboTwinMPlibPlanner(task.robot),
        start_state,
        goal,
        context,
        scene=scene,
    )


_COLLISION_ATTRIBUTE_NAMES = (
    "name",
    "link_name",
    "link_name1",
    "link_name2",
    "body_name",
    "body_name1",
    "body_name2",
    "object_name",
    "object_name1",
    "object_name2",
    "entity_name",
    "entity_name1",
    "entity_name2",
    "actor_name",
    "actor_name1",
    "actor_name2",
    "collision_name1",
    "collision_name2",
    "link1",
    "link2",
    "body1",
    "body2",
    "object1",
    "object2",
    "entity1",
    "entity2",
    "actor1",
    "actor2",
)


def _collision_json_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= 2:
        return repr(value)[:1000]
    if isinstance(value, dict):
        return {
            str(key): _collision_json_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _collision_json_value(item, depth=depth + 1)
            for item in value
        ]

    attributes: dict[str, Any] = {}
    for name in _COLLISION_ATTRIBUTE_NAMES:
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if callable(item):
            continue
        attributes[name] = _collision_json_value(item, depth=depth + 1)
    payload: dict[str, Any] = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value)[:1000],
    }
    if attributes:
        payload["attributes"] = attributes
    return payload


def collision_hit_payload(hit: Any) -> dict[str, Any]:
    value = _collision_json_value(hit)
    if isinstance(value, dict):
        return value
    return {
        "type": f"{type(hit).__module__}.{type(hit).__qualname__}",
        "repr": repr(hit)[:1000],
        "value": value,
    }


def _entity_descriptor(value: Any) -> dict[str, Any]:
    actor = getattr(value, "actor", None)
    direct_entity = getattr(value, "entity", None)
    actor_entity = getattr(actor, "entity", None) if actor is not None else None
    candidates = [
        item
        for item in (value, actor, direct_entity, actor_entity)
        if item is not None
    ]
    result: dict[str, Any] = {
        "type": type(value).__name__,
        "resolved_types": [type(item).__name__ for item in candidates],
    }
    for item in candidates:
        get_name = getattr(item, "get_name", None)
        if callable(get_name):
            with suppress(Exception):
                name = str(get_name())
                if name:
                    result["name"] = name
                    break
    for item in candidates:
        get_global_id = getattr(item, "get_global_id", None)
        if callable(get_global_id):
            with suppress(Exception):
                result["global_id"] = int(get_global_id())
                break
    return result


def collision_scene_entity_registry(task: Any) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "target": _entity_descriptor(task.pillbottle),
        "pad": _entity_descriptor(task.pad),
        "table": _entity_descriptor(task.table),
        "robot_links": [],
    }
    with suppress(Exception):
        registry["robot_links"] = [
            _entity_descriptor(link)
            for link in task.robot.left_entity.get_links()
        ]
    return registry


def _matched_collision_roles(
    payloads: list[dict[str, Any]],
    registry: dict[str, Any],
) -> list[str]:
    text = json.dumps(payloads, sort_keys=True).lower()
    matches: set[str] = set()
    aliases = {
        "target": ("pillbottle", "pill bottle"),
        "pad": ("pad",),
        "table": ("table",),
    }
    for role in ("target", "pad", "table"):
        descriptor = registry.get(role)
        names: list[str] = list(aliases[role])
        if isinstance(descriptor, dict) and descriptor.get("name"):
            names.append(str(descriptor["name"]).lower())
        if any(name and name in text for name in names):
            matches.add(role)

    for descriptor in registry.get("robot_links", []):
        if not isinstance(descriptor, dict) or not descriptor.get("name"):
            continue
        name = str(descriptor["name"]).lower()
        if name and name in text:
            matches.add("robot_link")
            break
    return sorted(matches)


def collision_recheck(task: Any, path: Any) -> dict[str, Any]:
    inner = task.robot.left_mplib_planner.planner
    registry = collision_scene_entity_registry(task)
    collisions: list[dict[str, Any]] = []
    for index, position in enumerate(path.positions):
        state = expand_arm_qpos(task, position)
        self_hits = inner.check_for_self_collision(state)
        env_hits = inner.check_for_env_collision(state)
        if self_hits or env_hits:
            self_payloads = [collision_hit_payload(hit) for hit in self_hits]
            env_payloads = [collision_hit_payload(hit) for hit in env_hits]
            collisions.append(
                {
                    "waypoint_index": index,
                    "self_collision_count": len(self_hits),
                    "environment_collision_count": len(env_hits),
                    "self_hits": self_payloads,
                    "environment_hits": env_payloads,
                    "matched_self_roles": _matched_collision_roles(
                        self_payloads,
                        registry,
                    ),
                    "matched_environment_roles": _matched_collision_roles(
                        env_payloads,
                        registry,
                    ),
                }
            )
    return {
        "provenance_schema": "mplib-collision-v1",
        "scene_entities": registry,
        "waypoints_checked": len(path.positions),
        "collision_free": not collisions,
        "collisions": collisions,
    }


def self_collision_recheck(task: Any, path: Any) -> dict[str, Any]:
    inner = task.robot.left_mplib_planner.planner
    registry = collision_scene_entity_registry(task)
    collisions: list[dict[str, Any]] = []
    for index, position in enumerate(path.positions):
        state = expand_arm_qpos(task, position)
        self_hits = inner.check_for_self_collision(state)
        if self_hits:
            self_payloads = [collision_hit_payload(hit) for hit in self_hits]
            collisions.append(
                {
                    "waypoint_index": index,
                    "self_collision_count": len(self_hits),
                    "self_hits": self_payloads,
                    "matched_self_roles": _matched_collision_roles(
                        self_payloads,
                        registry,
                    ),
                }
            )
    return {
        "provenance_schema": "mplib-collision-v1",
        "scene_entities": registry,
        "waypoints_checked": len(path.positions),
        "self_collision_free": not collisions,
        "collisions": collisions,
    }


def terminal_reachability(
    task: Any,
    candidate: Any,
    *,
    context: Any,
    scene: Any,
    limits: Any,
    timer: Any,
) -> dict[str, Any]:
    from vla_robot_sorting.contracts import MotionGoal
    from vla_robot_sorting.planning import PantheraRoboTwinMotionPlanner
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits

    planner = PantheraRoboTwinMotionPlanner(task.robot)
    plan = planner.plan(
        current_arm_state(task),
        MotionGoal(candidate.grasp_pose),
        context,
        scene=scene,
    )
    payload: dict[str, Any] = {
        "success": False,
        "planner_id": plan.planner_id,
        "planning_time_s": plan.planning_time_s,
        "environment_contact_policy": (
            "CuRobo terminal reachability models robot/table; target contact is allowed "
            "at the final grasp. SAPIEN environment collision is enforced to pregrasp."
        ),
    }
    if not plan.succeeded:
        payload["failure"] = {
            "code": plan.failure.code.value if plan.failure else "missing",
            "message": plan.failure.message if plan.failure else "missing",
        }
        return payload

    assert plan.path is not None
    try:
        trajectory = timer.parameterize(plan.path, limits)
        safety = validate_joint_trajectory_limits(trajectory, limits)
        self_collision = self_collision_recheck(task, plan.path)
    except Exception as exc:
        payload["validation_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    payload.update(
        {
            "path_samples": len(plan.path.positions),
            "retimed_duration_s": trajectory.duration_s,
            "dynamic_safe": safety.is_safe,
            "dynamic_violation_count": len(safety.violations),
            "self_collision_recheck": self_collision,
            "curobo_world_sync": plan.path.metadata.get("curobo_world_sync"),
            "success": safety.is_safe and self_collision["self_collision_free"],
        }
    )
    return payload


def plan_pregrasp_with_interior_rescue(
    task: Any,
    start: Any,
    goal: Any,
    context: Any,
    *,
    scene: Any,
    limits: Any,
    timer: Any,
    primary_fallback: Any = None,
    mplib_planner: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Bounded independent MPlib attempt after a CuRobo success at a joint bound.

    Preserve the vendor primary outcome. The rescue is accepted only if its own
    RAW and retimed trajectory has the exact interior reserve. Normal B5
    dynamics, FCL, geometry and runtime checks still execute after selection.
    """
    from vla_robot_sorting.planning import PantheraRoboTwinMPlibPlanner
    from vla_robot_sorting.planning.pregrasp_joint_reserve import audit_pregrasp_joint_reserve

    base_planner = primary_fallback or plan_panthera_primary_fallback
    plan, original_policy = base_planner(task, start, goal, context, scene=scene)
    policy = dict(original_policy)
    if not plan.succeeded or plan.path is None:
        return plan, policy

    try:
        primary_timed = timer.parameterize(plan.path, limits)
        primary_report = audit_pregrasp_joint_reserve(plan.path, primary_timed, limits)
    except (ValueError, TypeError) as exc:
        policy["interior_rescue"] = {
            "attempted": False,
            "reason": "primary_path_validation_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return plan, policy

    if primary_report["passed"] or plan.planner_id != "robotwin-curobo":
        return plan, policy

    fallback = mplib_planner or PantheraRoboTwinMPlibPlanner(task.robot)
    alternative = fallback.plan(start, goal, context, scene=scene)
    alternative_report = None
    alternative_validation_error = None
    if alternative.succeeded and alternative.path is not None:
        try:
            timed = timer.parameterize(alternative.path, limits)
            alternative_report = audit_pregrasp_joint_reserve(alternative.path, timed, limits)
        except (ValueError, TypeError) as exc:
            alternative_validation_error = f"{type(exc).__name__}: {exc}"

    rescue = {
        "strategy": "pregrasp_interior_mplib_rescue_v1",
        "attempted": True,
        "primary_native_success": True,
        "primary_planner_id": plan.planner_id,
        "primary_reserve": primary_report,
        "fallback_planner_id": alternative.planner_id,
        "fallback_native_success": bool(alternative.succeeded),
        "fallback_reserve": alternative_report,
        "fallback_validation_error": alternative_validation_error,
        "fallback_failure": _planning_failure_payload(alternative),
        "accepted": (
            alternative.succeeded
            and alternative_report is not None
            and alternative_report["passed"] is True
        ),
    }
    if not rescue["accepted"]:
        policy["interior_rescue"] = rescue
        return plan, policy

    rescue_policy = {
        **policy,
        "strategy": rescue["strategy"],
        "primary_native_success": True,
        "primary_reserve": primary_report,
        "primary_success": True,
        "fallback_attempted": True,
        "fallback_planner_id": alternative.planner_id,
        "fallback_success": True,
        "selected_planner_id": alternative.planner_id,
        "plan_success": True,
        "planning_time_s": (float(plan.planning_time_s) + float(alternative.planning_time_s)),
        "interior_rescue": rescue,
    }
    return alternative, rescue_policy


def evaluate_candidates(
    task: Any,
    scene: Any,
    *,
    gripper: Any,
    gate: dict[str, Any],
    context: Any,
    limits: Any,
    timer: Any,
    stop_after_feasible_height_group: bool = True,
    selected_candidate_id: str | None = None,
    preferred_candidate_id: str | None = None,
) -> dict[str, Any]:
    from vla_robot_sorting.contracts import MotionGoal
    from vla_robot_sorting.contracts.grasping import FeasibilityEvaluation, GraspRequest
    from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator
    from vla_robot_sorting.grasping.ranking import GraspRanker
    from vla_robot_sorting.planning.pregrasp_joint_reserve import audit_pregrasp_joint_reserve
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits
    from vla_robot_sorting.telemetry import trajectory_kinematics

    request = GraspRequest(
        object_id="pillbottle",
        scene_revision=scene.scene_revision,
        transform_revision=scene.transform_revision,
        gripper=gripper,
        minimum_table_clearance_m=float(gate["minimum_table_clearance_m"]),
    )
    generation = GeometryFirstGraspGenerator(
        cylinder_side_grasp_height_fractions=tuple(
            float(value) for value in gate.get("cylinder_side_grasp_height_fractions", (0.0,))
        ),
        box_side_grasp_height_fractions=tuple(
            float(value) for value in gate.get("box_side_grasp_height_fractions", (0.0,))
        ),
        box_parallel_jaw_roll_variants=bool(gate.get("box_parallel_jaw_roll_variants", False)),
    ).generate(request, scene)
    if not generation.succeeded:
        return {
            "passed": False,
            "generation_failure": generation.failure.message if generation.failure else "unknown",
            "candidate_count": 0,
            "non_top_down_count": 0,
            "candidates": [],
            "evaluations": [],
            "rejection_histogram": {"generation_failure": 1},
            "selected": None,
        }

    candidates = generation.candidates
    non_top = [
        item
        for item in candidates
        if abs(float(item.approach_direction[2])) <= float(gate["non_top_down_abs_approach_z_max"])
    ]
    provenance_complete = [
        item
        for item in candidates
        if (
            item.source == "geometry_first"
            and item.geometry_source == "sim_ground_truth"
            and item.scene_revision == scene.scene_revision
            and item.transform_revision == scene.transform_revision
            and item.grasp_pose.frame_id == "world"
            and item.pregrasp_pose.frame_id == "world"
        )
    ]

    # The same audited CuRobo-primary/MPlib-fallback helper is reused by
    # B5, B7, and the integrated pregrasp probe.
    start = current_arm_state(task)
    records: list[dict[str, Any]] = []
    evaluations: list[Any] = []
    evaluated_candidates: list[Any] = []
    accepted_pregrasp_plans: dict[str, tuple[Any, dict[str, Any]]] = {}
    reasons = Counter()

    ordered = sorted(
        non_top,
        key=lambda item: (
            float(item.metadata.get("grasp_height_fraction", 0.0)),
            -item.raw_quality,
            item.candidate_id,
        ),
    )
    current_height_fraction: float | None = None
    feasible_in_height_group = False
    for candidate in ordered[: int(gate["max_candidate_evaluations"])]:
        height_fraction = float(candidate.metadata.get("grasp_height_fraction", 0.0))
        if (
            stop_after_feasible_height_group
            and current_height_fraction is not None
            and height_fraction != current_height_fraction
            and feasible_in_height_group
        ):
            break
        if current_height_fraction != height_fraction:
            current_height_fraction = height_fraction
            feasible_in_height_group = False
        terminal = terminal_reachability(
            task,
            candidate,
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
        )
        plan, planning = plan_pregrasp_with_interior_rescue(
            task,
            start,
            MotionGoal(candidate.pregrasp_pose),
            context,
            scene=scene,
            limits=limits,
            timer=timer,
        )
        detail: dict[str, Any] = {
            "candidate": candidate_dict(candidate),
            "terminal_reachability": terminal,
            **{f"pregrasp_{key}": value for key, value in planning.items()},
        }
        dynamic_safe = False
        collision_free = False
        reserve_safe = False
        motion_cost = 10.0
        if plan.succeeded:
            assert plan.path is not None
            try:
                trajectory = timer.parameterize(plan.path, limits)
                safety = validate_joint_trajectory_limits(trajectory, limits)
                reserve = audit_pregrasp_joint_reserve(plan.path, trajectory, limits)
                reserve_safe = reserve["passed"] is True
                collision = collision_recheck(task, plan.path)
                metrics = trajectory_kinematics(trajectory)
                dynamic_safe = safety.is_safe
                collision_free = bool(collision["collision_free"])
                motion_cost = metrics.joint_path_length_rad
                detail.update(
                    {
                        "pregrasp_path_samples": len(plan.path.positions),
                        "retimed_duration_s": trajectory.duration_s,
                        "dynamic_safe": dynamic_safe,
                        "dynamic_violation_count": len(safety.violations),
                        "pregrasp_joint_reserve": reserve,
                        "collision_recheck": collision,
                        "joint_path_length_rad": motion_cost,
                    }
                )
            except Exception as exc:
                detail["pregrasp_validation_error"] = f"{type(exc).__name__}: {exc}"
        else:
            detail["pregrasp_failure"] = {
                "code": plan.failure.code.value if plan.failure else "missing",
                "message": plan.failure.message if plan.failure else "missing",
            }

        reachable = (
            bool(terminal.get("success")) and plan.succeeded and dynamic_safe and reserve_safe
        )
        eligible_collision = bool(terminal.get("success")) and collision_free
        if not terminal.get("success"):
            reasons["terminal_reachability"] += 1
        if not plan.succeeded:
            reasons["pregrasp_plan"] += 1
        if plan.succeeded and not dynamic_safe:
            reasons["dynamic_limit"] += 1
        if plan.succeeded and not collision_free:
            reasons["pregrasp_collision"] += 1
        if plan.succeeded and not reserve_safe:
            reasons["pregrasp_joint_reserve"] += 1

        clearance = max(0.0, float(candidate.metadata.get("table_clearance_m", 0.0)))
        evaluation = FeasibilityEvaluation(
            candidate_id=candidate.candidate_id,
            reachable=reachable,
            collision_free=eligible_collision,
            clearance_margin_m=clearance,
            approach_clearance_m=clearance,
            grasp_quality=candidate.raw_quality,
            motion_cost=motion_cost,
            object_identity_confidence=1.0,
        )
        evaluations.append(evaluation)
        evaluated_candidates.append(candidate)
        detail["reachable"] = reachable
        detail["collision_free"] = eligible_collision
        if reachable and eligible_collision:
            accepted_pregrasp_plans[candidate.candidate_id] = (plan, planning)
        records.append(detail)
        feasible_in_height_group = feasible_in_height_group or (reachable and eligible_collision)

    selected = None
    ranked_payload = []
    eligible_candidate_ids: list[str] = []
    if evaluated_candidates:
        ranked = GraspRanker().rank(tuple(evaluated_candidates), tuple(evaluations))
        by_id = {item.candidate_id: item for item in evaluated_candidates}
        for score in ranked.scores:
            ranked_payload.append(
                {
                    "candidate_id": score.candidate_id,
                    "total": score.total,
                    "eligible": score.eligible,
                    "rejection_reasons": score.rejection_reasons,
                    "factors": score.factors,
                }
            )
            for reason in score.rejection_reasons:
                reasons[f"rank:{reason}"] += 1
            if score.eligible:
                eligible_candidate_ids.append(score.candidate_id)
                if selected_candidate_id is None and selected is None:
                    selected = by_id[score.candidate_id]

        if selected_candidate_id is not None and selected_candidate_id in eligible_candidate_ids:
            selected = by_id[selected_candidate_id]
        elif selected_candidate_id is None and preferred_candidate_id in eligible_candidate_ids:
            selected = by_id[preferred_candidate_id]

    passed = (
        len(candidates) >= 2
        and len(non_top) >= 1
        and len(provenance_complete) == len(candidates)
        and selected is not None
    )
    return {
        "passed": passed,
        "candidate_count": len(candidates),
        "non_top_down_count": len(non_top),
        "provenance_complete_count": len(provenance_complete),
        "candidates": [candidate_dict(item) for item in candidates],
        "evaluations": records,
        "evaluated_candidate_count": len(records),
        "candidate_evaluation_policy": {
            "ordering": "height_fraction_ascending_then_quality_then_id",
            "stop_after_feasible_height_group": stop_after_feasible_height_group,
            "max_candidate_evaluations": int(gate["max_candidate_evaluations"]),
            "selected_candidate_id_override": selected_candidate_id,
            "preferred_candidate_id": preferred_candidate_id,
            "preferred_selected": (
                selected is not None and selected.candidate_id == preferred_candidate_id
            ),
            "preference_fallback_allowed": selected_candidate_id is None,
        },
        "eligible_candidate_ids": eligible_candidate_ids,
        "ranking": ranked_payload,
        "rejection_histogram": dict(reasons),
        "selected": candidate_dict(selected) if selected is not None else None,
        "_selected_object": selected,
        "_selected_pregrasp_plan": (
            accepted_pregrasp_plans.get(selected.candidate_id) if selected is not None else None
        ),
    }


class PhysicsStepCounter:
    def __init__(self) -> None:
        self.count = 0

    def on_physics_step(self, task: Any) -> None:
        del task
        self.count += 1


def sync_scaled_target_into_mplib_world(task: Any) -> dict[str, Any]:
    """Inject the late-spawned target into the existing MPlib world with scale applied."""

    import numpy as np
    from mplib import Pose
    from mplib.collision_detection.fcl import (
        Box,
        BVHModel,
        Capsule,
        CollisionObject,
        Convex,
        Cylinder,
        FCLObject,
        Sphere,
    )
    from mplib.sapien_utils.conversion import convert_object_name
    from sapien.physx import (
        PhysxCollisionShapeBox,
        PhysxCollisionShapeCapsule,
        PhysxCollisionShapeConvexMesh,
        PhysxCollisionShapeCylinder,
        PhysxCollisionShapeSphere,
        PhysxCollisionShapeTriangleMesh,
        PhysxRigidBaseComponent,
    )
    from transforms3d.euler import euler2quat

    wrapper = task.robot.left_mplib_planner
    planner = wrapper.planner
    planning_world = planner.planning_world
    actor = task.pillbottle.actor
    target_name = convert_object_name(actor)
    component = actor.find_component_by_type(PhysxRigidBaseComponent)
    if component is None:
        raise RuntimeError("pillbottle has no PhysxRigidBaseComponent")

    shapes: list[Any] = []
    shape_poses: list[Any] = []
    scaled_mesh_shape_count = 0
    non_unit_scale_shape_count = 0
    scale_vectors: list[list[float]] = []

    for shape in component.collision_shapes:
        shape_pose = Pose(shape.local_pose)
        geometry: Any

        if isinstance(shape, PhysxCollisionShapeBox):
            geometry = Box(side=np.asarray(shape.half_size, dtype=float) * 2.0)
        elif isinstance(shape, PhysxCollisionShapeCapsule):
            geometry = Capsule(radius=float(shape.radius), lz=float(shape.half_length) * 2.0)
            shape_pose *= Pose(q=euler2quat(0.0, np.pi / 2.0, 0.0))
        elif isinstance(shape, PhysxCollisionShapeCylinder):
            geometry = Cylinder(
                radius=float(shape.radius),
                lz=float(shape.half_length) * 2.0,
            )
            shape_pose *= Pose(q=euler2quat(0.0, np.pi / 2.0, 0.0))
        elif isinstance(shape, PhysxCollisionShapeSphere):
            geometry = Sphere(radius=float(shape.radius))
        elif isinstance(shape, PhysxCollisionShapeConvexMesh):
            scale = np.asarray(shape.scale, dtype=float)
            scale_vectors.append([float(value) for value in scale])
            if not np.allclose(scale, 1.0):
                non_unit_scale_shape_count += 1
            vertices = np.asarray(shape.vertices, dtype=float) * scale
            geometry = Convex(vertices=vertices, faces=np.asarray(shape.triangles))
            scaled_mesh_shape_count += 1
        elif isinstance(shape, PhysxCollisionShapeTriangleMesh):
            scale = np.asarray(shape.scale, dtype=float)
            scale_vectors.append([float(value) for value in scale])
            if not np.allclose(scale, 1.0):
                non_unit_scale_shape_count += 1
            geometry = BVHModel()
            geometry.begin_model()
            geometry.add_sub_model(
                vertices=np.asarray(shape.vertices, dtype=float) * scale,
                faces=np.asarray(shape.triangles),
            )
            geometry.end_model()
            scaled_mesh_shape_count += 1
        else:
            raise TypeError(f"unsupported target collision shape: {type(shape).__name__}")

        shapes.append(CollisionObject(geometry))
        shape_poses.append(shape_pose)

    if not shapes:
        raise RuntimeError("pillbottle target has no collision shapes")

    if planning_world.has_object(target_name):
        planning_world.remove_object(target_name)
    planning_world.add_object(
        FCLObject(target_name, actor.pose, shapes, shape_poses)
    )
    target_present = bool(planning_world.has_object(target_name))
    if not target_present:
        raise RuntimeError(
            f"MPlib PlanningWorld failed to register target {target_name!r}"
        )

    return {
        "strategy": "scaled_target_fcl_injection",
        "planner_class": type(planner).__name__,
        "planning_world_class": type(planning_world).__name__,
        "move_group": str(task.robot.left_move_group),
        "target_name": target_name,
        "target_present": target_present,
        "collision_shape_count": len(shapes),
        "scaled_mesh_shape_count": scaled_mesh_shape_count,
        "non_unit_scale_shape_count": non_unit_scale_shape_count,
        "scale_vectors": scale_vectors,
        "planner_reused": True,
        "planning_world_reused": True,
    }


def perturb_target(task: Any, distance_m: float) -> dict[str, Any]:
    import sapien

    pose = task.pillbottle.get_pose()
    before = [float(v) for v in pose.p]
    delta = distance_m if before[1] <= 0.05 else -distance_m
    after = [before[0], before[1] + delta, before[2]]
    task.pillbottle.actor.set_pose(sapien.Pose(after, pose.q))
    return {"before_xyz_m": before, "after_xyz_m": after, "delta_y_m": delta}


def main() -> int:
    args = parse_args()
    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()
    gate_path = project_root / "configs/s1.3/grasp_recovery_gates.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    runs = int(args.runs if args.runs is not None else gate["runs"])
    if runs < 1:
        raise ValueError("runs must be at least 1")

    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))

    import numpy as np
    import yaml
    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.contracts import PlanningContext
    from vla_robot_sorting.contracts.recovery import (
        RecoveryEvent,
        RecoveryState,
        RecoveryTrigger,
    )
    from vla_robot_sorting.control import (
        PantheraRoboTwinTrajectoryExecutor,
        build_execution_segment,
    )
    from vla_robot_sorting.orchestration.recovery import ClosedLoopRecoveryPolicy
    from vla_robot_sorting.perception.change_detection import SceneChangeDetector
    from vla_robot_sorting.planning import (
        PhysicsAlignedTimeDilationParameterizer,
        RoboTwinVendorTrajectoryParameterizer,
        load_joint_limits_from_urdf,
    )
    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES
    from vla_robot_sorting.simulation.render_profile import (
        apply_fast_preview_render_profile,
    )

    records: list[dict[str, Any]] = []
    for run_index in range(runs):
        seed = args.seed_start + run_index
        task = None
        record: dict[str, Any] = {"run": run_index, "seed": seed}
        try:
            task, task_args = prepare_task_and_args(args.task, args.task_config)
            task_args["render_freq"] = 0
            task_args["save_data"] = False
            task_args["collect_data"] = False
            task_args["domain_randomization"]["cluttered_table"] = False
            task_args["domain_randomization"]["random_background"] = False
            task_args["domain_randomization"]["random_light"] = False
            apply_fast_preview_render_profile(task_args)

            setup_started = perf_counter()
            task.setup_demo(now_ep_num=run_index, seed=seed, **task_args)
            record["setup_duration_s"] = perf_counter() - setup_started

            initial_mplib_world = sync_scaled_target_into_mplib_world(task)
            geometry = actor_box_from_collision_mesh(task, robotwin_root)
            gripper, gripper_provenance = load_gripper_proxy(robotwin_root, task_args)
            old_revision = f"b57-seed-{seed}-initial"
            transform_revision = "robotwin-static-v1"
            old_scene = make_scene_snapshot(
                task,
                revision=old_revision,
                geometry_info=geometry,
                transform_revision=transform_revision,
            )
            old_context = PlanningContext(old_revision, transform_revision)

            curobo_config = robotwin_root / "assets/embodiments/panthera-6dof/curobo.yml"
            with curobo_config.open(encoding="utf-8") as handle:
                cspace = yaml.safe_load(handle)["robot_cfg"]["kinematics"]["cspace"]
            urdf_path = Path(task.robot.left_urdf_path)
            if not urdf_path.is_absolute():
                urdf_path = (robotwin_root / urdf_path).resolve()
            limits = load_joint_limits_from_urdf(
                urdf_path,
                PANTHERA_JOINT_NAMES,
                max_acceleration_rad_s2=cspace["max_acceleration"],
                max_jerk_rad_s3=cspace["max_jerk"],
            )
            timing_gate = json.loads(
                (project_root / "configs/s1.2/timing_gates.json").read_text(encoding="utf-8")
            )
            timer = PhysicsAlignedTimeDilationParameterizer(
                RoboTwinVendorTrajectoryParameterizer(),
                physics_period_s=float(timing_gate["physics_period_s"]),
                max_dilation_factor=int(timing_gate["time_dilation"]["max_factor"]),
            )

            b5 = evaluate_candidates(
                task,
                old_scene,
                gripper=gripper,
                gate=gate,
                context=old_context,
                limits=limits,
                timer=timer,
            )
            selected = b5.pop("_selected_object", None)
            record["mplib_world_initial"] = initial_mplib_world
            record["geometry"] = geometry
            record["geometry_proxy"] = old_scene.object("pillbottle").metadata.get(
                "geometry_proxy"
            )
            record["gripper"] = gripper_provenance
            record["table_height_m"] = old_scene.table_height_m
            record["B5"] = b5
            if selected is None:
                record["B7"] = {"passed": False, "reason": "B5 produced no selected grasp"}
                records.append(record)
                continue

            from vla_robot_sorting.contracts import MotionGoal
            from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits

            initial_plan, initial_planning = plan_panthera_primary_fallback(
                task,
                current_arm_state(task),
                MotionGoal(selected.pregrasp_pose),
                old_context,
                scene=old_scene,
            )
            if not initial_plan.succeeded:
                record["B7"] = {
                    "passed": False,
                    "reason": "selected B5 pregrasp could not be replanned",
                    "initial_planning": initial_planning,
                }
                records.append(record)
                continue
            assert initial_plan.path is not None
            old_trajectory = timer.parameterize(initial_plan.path, limits)
            if not validate_joint_trajectory_limits(old_trajectory, limits).is_safe:
                record["B7"] = {
                    "passed": False,
                    "reason": "initial trajectory dynamic safety failed",
                    "initial_planning": initial_planning,
                }
                records.append(record)
                continue
            controller_period = float(old_trajectory.metadata["controller_period_s"])
            segment = build_execution_segment(
                old_trajectory,
                old_context,
                segment_id=f"stale-{seed}",
                requested_horizon_s=float(gate["execution_probe"]["requested_horizon_s"]),
                controller_period_s=controller_period,
            )

            qpos_before = np.asarray(task.robot.left_entity.get_qpos(), dtype=float).copy()
            perturb = perturb_target(task, float(gate["perturbation"]["translation_m"]))
            moved_mplib_world = sync_scaled_target_into_mplib_world(task)
            moved_geometry = actor_box_from_collision_mesh(task, robotwin_root)
            new_revision = f"b57-seed-{seed}-moved"
            new_scene = make_scene_snapshot(
                task,
                revision=new_revision,
                geometry_info=moved_geometry,
                transform_revision=transform_revision,
            )

            detection_started = perf_counter()
            change = SceneChangeDetector().detect(
                old_scene,
                new_scene,
                target_object_id="pillbottle",
                destination_id="pad",
            )
            detection_latency = perf_counter() - detection_started

            decision_started = perf_counter()
            decision = ClosedLoopRecoveryPolicy().decide(
                RecoveryEvent(
                    trigger=RecoveryTrigger.SCENE_CHANGE,
                    observed_at_s=detection_started,
                    scene_change=change,
                ),
                RecoveryState(),
            )
            decision_latency = perf_counter() - decision_started

            counter = PhysicsStepCounter()
            task.add_physics_step_observer(counter)
            executor = PantheraRoboTwinTrajectoryExecutor(
                task,
                scene_revision=lambda revision=new_revision: revision,
                transform_revision=lambda revision=transform_revision: revision,
                physics_period_s=float(timing_gate["physics_period_s"]),
            )
            stale_feedback = executor.execute(segment)
            task.clear_physics_step_observers()
            qpos_after = np.asarray(task.robot.left_entity.get_qpos(), dtype=float).copy()

            replan_started = perf_counter()
            new_context = PlanningContext(new_revision, transform_revision)
            replacement = evaluate_candidates(
                task,
                new_scene,
                gripper=gripper,
                gate=gate,
                context=new_context,
                limits=limits,
                timer=timer,
            )
            replacement_selected = replacement.pop("_selected_object", None)
            replacement_latency = perf_counter() - replan_started

            replacement_segment_ready = False
            replacement_collision_free = False
            replacement_planning = None
            replacement_collision = None
            replacement_segment_scene_revision = None
            replacement_segment_transform_revision = None
            if replacement_selected is not None:
                replacement_plan, replacement_planning = plan_panthera_primary_fallback(
                    task,
                    current_arm_state(task),
                    MotionGoal(replacement_selected.pregrasp_pose),
                    new_context,
                    scene=new_scene,
                )
                if replacement_plan.succeeded:
                    assert replacement_plan.path is not None
                    replacement_collision = collision_recheck(task, replacement_plan.path)
                    replacement_trajectory = timer.parameterize(replacement_plan.path, limits)
                    replacement_safe = validate_joint_trajectory_limits(
                        replacement_trajectory,
                        limits,
                    ).is_safe
                    replacement_collision_free = bool(replacement_collision["collision_free"])
                    if replacement_safe and replacement_collision_free:
                        replacement_segment = build_execution_segment(
                            replacement_trajectory,
                            new_context,
                            segment_id=f"replacement-{seed}",
                            requested_horizon_s=float(
                                gate["execution_probe"]["requested_horizon_s"]
                            ),
                            controller_period_s=float(
                                replacement_trajectory.metadata["controller_period_s"]
                            ),
                        )
                        replacement_segment_scene_revision = (
                            replacement_segment.scene_revision
                        )
                        replacement_segment_transform_revision = (
                            replacement_segment.transform_revision
                        )
                        replacement_segment_ready = True

            from vla_robot_sorting.contracts.motion import ExecutionStatus
            from vla_robot_sorting.contracts.recovery import RecoveryAction, SceneChangeKind

            stale_rejected = (
                stale_feedback.status is ExecutionStatus.CANCELLED
                and stale_feedback.replan_required
                and counter.count == 0
                and np.array_equal(qpos_before, qpos_after)
            )
            b7_pass = (
                change.kind is SceneChangeKind.TARGET_MOVED
                and decision.action is RecoveryAction.REPLAN
                and stale_rejected
                and replacement.get("passed") is True
                and replacement_segment_ready
                and replacement_collision_free
            )
            record["B7"] = {
                "passed": b7_pass,
                "perturbation": perturb,
                "mplib_world_after_perturbation": moved_mplib_world,
                "old_scene_revision": old_revision,
                "new_scene_revision": new_revision,
                "transform_revision": transform_revision,
                "scene_change_kind": change.kind.value,
                "position_delta_m": change.position_delta_m,
                "detection_latency_s": detection_latency,
                "decision_action": decision.action.value,
                "decision_latency_s": decision_latency,
                "initial_planning": initial_planning,
                "stale_segment_scene_revision": segment.scene_revision,
                "stale_segment_transform_revision": segment.transform_revision,
                "stale_feedback_status": stale_feedback.status.value,
                "stale_replan_required": stale_feedback.replan_required,
                "stale_physics_steps": counter.count,
                "qpos_unchanged": bool(np.array_equal(qpos_before, qpos_after)),
                "replacement_latency_s": replacement_latency,
                "replacement_candidate_count": replacement.get("candidate_count"),
                "replacement_passed": replacement.get("passed"),
                "replacement_selected": replacement.get("selected"),
                "replacement_planning": replacement_planning,
                "replacement_collision_recheck": replacement_collision,
                "replacement_segment_ready": replacement_segment_ready,
                "replacement_segment_scene_revision": replacement_segment_scene_revision,
                "replacement_segment_transform_revision": (
                    replacement_segment_transform_revision
                ),
                "replacement_collision_free": replacement_collision_free,
            }
            records.append(record)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record.setdefault("B5", {"passed": False})
            record.setdefault("B7", {"passed": False})
            records.append(record)
        finally:
            if task is not None:
                with suppress(Exception):
                    task.close_env()

    b5_pass_count = sum(item.get("B5", {}).get("passed") is True for item in records)
    b7_pass_count = sum(item.get("B7", {}).get("passed") is True for item in records)
    payload = {
        "schema": "panthera-native-b5-b7-v1",
        "task": args.task,
        "task_config": args.task_config,
        "gate_config": gate,
        "gate_config_path": str(gate_path),
        "runs": runs,
        "B5_pass_count": b5_pass_count,
        "B7_pass_count": b7_pass_count,
        "B5_pass": b5_pass_count == runs,
        "B7_pass": b7_pass_count == runs,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["B5_pass"] and payload["B7_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
