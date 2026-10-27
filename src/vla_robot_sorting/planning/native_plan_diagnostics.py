"""Plan-only MPlib 0.2.1 diagnostics. FCL geometry is NOT PhysX contact evidence.

Only MPlib model qpos is changed, then restored. No simulator setters,
physics stepping, contact exemptions, executor calls or motion authorization.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from vla_robot_sorting.planning.manipulation_recovery import (
    RecoveryContractError,
    digest,
    write_new_json,
)

MPLIB_PLANNER_SHA256 = "785f77233baf4159f3327b664fae2507572c189223a6075f42546c1133ddfa0e"


def finite_vector(raw: Any, size: int | None = None) -> list[float]:
    if not isinstance(raw, (list, tuple)) or not raw or (size is not None and len(raw) != size):
        raise RecoveryContractError("missing or mismatched numeric vector")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in raw
    ):
        raise RecoveryContractError("numeric vector must be finite; booleans forbidden")
    return [float(v) for v in raw]


def pose_vector(pose: Any) -> list[float]:
    return finite_vector([*[float(v) for v in pose.p], *[float(v) for v in pose.q]], 7)


def validate_distance_selftest(separated: float, overlapping: float) -> dict[str, Any]:
    """Two radius-0.1 m spheres, center separations 0.3 and 0.15 m."""
    if (
        not math.isfinite(separated)
        or not math.isfinite(overlapping)
        or abs(separated - 0.1) > 1e-5
        or abs(overlapping + 0.05) > 1e-5
    ):
        raise RecoveryContractError("native distance sign/unit selftest failed")
    return {
        "passed": True,
        "unit": "m",
        "positive_means": "separated",
        "negative_means": "FCL_penetration",
        "separated_m": separated,
        "overlapping_m": overlapping,
        "scope": "primitive API check; not validation of mesh/PhysX equivalence",
    }


def validate_fk_agreement(computed: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """Refuse incorrect root/world convention or incomplete link set."""
    if not computed or set(computed) != set(measured):
        raise RecoveryContractError("FK reference link set mismatch")
    max_position = max_rotation = 0.0
    for name, raw in computed.items():
        a, b = finite_vector(raw, 7), finite_vector(measured[name], 7)
        for q in (a[3:], b[3:]):
            if abs(sum(v * v for v in q) - 1.0) > 1e-4:
                raise RecoveryContractError("FK quaternion not normalized")
        position = math.dist(a[:3], b[:3])
        rotation = min(math.dist(a[3:], b[3:]), math.dist(a[3:], [-x for x in b[3:]]))
        max_position, max_rotation = max(max_position, position), max(max_rotation, rotation)
    if max_position > 1e-4 or max_rotation > 1e-4:
        raise RecoveryContractError("FK world transform disagrees with simulator links")
    return {
        "passed": True,
        "max_position_error_m": max_position,
        "max_quaternion_l2_error": max_rotation,
        "link_count": len(computed),
    }


def collect_plan_samples(backend: Any, rows: Any, times: Any) -> dict[str, Any]:
    """Backend API failures propagate after restoration; partial data cannot pass."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise RecoveryContractError("empty plan")
    positions = [finite_vector(row) for row in rows]
    if len({len(row) for row in positions}) != 1:
        raise RecoveryContractError("inconsistent qpos size")
    timestamps = finite_vector(times, len(positions))
    if timestamps[0] < 0 or any(b <= a for a, b in pairwise(timestamps)):
        raise RecoveryContractError("plan times must be nonnegative and strictly increasing")
    saved = backend.snapshot()
    samples = []
    try:
        for i, (qpos, timestamp) in enumerate(zip(positions, timestamps, strict=True)):
            row = backend.sample(qpos)
            _validate_sample(row)
            row.update(sample_index=i, time_from_start_s=timestamp, qpos_rad=qpos)
            samples.append(row)
    finally:
        backend.restore(saved)
        if backend.snapshot() != saved:
            raise RecoveryContractError("planner state restoration failed")
    return {
        "schema": "s1m.plan-samples.v2",
        "source_kind": backend.source_kind,
        "samples": samples,
        "waypoint_count": len(samples),
        "physics_steps_during_collection": 0,
        "execution_status": "not_executed",
        "controller_eligible": False,
        "native_acceptance": False,
        "collision_scope": "FCL/ACM geometry; not physical contact or swept-volume proof",
    }


def native_distance_request(fcl: Any) -> Any:
    """Pinned signed-distance solver; collision checks use a separate request."""
    try:
        solver = fcl.GJKSolverType.GST_INDEP
        return fcl.DistanceRequest(
            enable_nearest_points=True,
            enable_signed_distance=True,
            distance_tolerance=1e-6,
            gjk_solver_type=solver,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RecoveryContractError("unsupported pinned FCL distance request") from exc


def single_witness_distance_record(name: str, result: Any) -> dict[str, Any]:
    """MPlib 0.2.1 exposes one 3D solver point, not a two-point witness pair."""
    if not isinstance(name, str) or not name:
        raise RecoveryContractError("missing robot collision object identity")
    try:
        value = float(result.min_distance)
        raw = result.nearest_points
        if getattr(raw, "shape", None) != (3,):
            raise RecoveryContractError("FCL must report exactly one 3D point")
        point = [float(v) for v in raw]
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RecoveryContractError("invalid native FCL distance result") from exc
    if not math.isfinite(value) or abs(value) > 100 or not all(math.isfinite(v) for v in point):
        raise RecoveryContractError("nonfinite or unbounded native FCL distance")
    return {
        "robot_collision_object": name,
        "signed_distance_m": value,
        "nearest_point_world_m": point,
        "witness_contract": "one_solver_point_on_first_object",
    }


class NativeMPlibDiagnosticBackend:
    """Runtime-checked adapter for the pinned APIs documented in the runbook."""

    source_kind = "native_planner_geometry"

    def __init__(self, task: Any, helpers: Any) -> None:
        import mplib
        import mplib.planner
        from mplib.collision_detection import AllowedCollision, fcl

        versions = {name: importlib.metadata.version(name) for name in ("mplib", "sapien")}
        if versions != {"mplib": "0.2.1", "sapien": "3.0.0b1"}:
            raise RecoveryContractError(f"unsupported native versions: {versions}")
        planner_path = Path(mplib.planner.__file__)
        if digest(planner_path) != MPLIB_PLANNER_SHA256:
            raise RecoveryContractError("MPlib planner.py differs from audited v0.2.1 tag")
        self.task, self.helpers, self.fcl, self.Pose = task, helpers, fcl, mplib.Pose
        self.inner = task.robot.left_mplib_planner.planner
        self.model, self.world = self.inner.robot, self.inner.planning_world
        self.pino = self.inner.pinocchio_model
        self.acm = self.world.get_allowed_collision_matrix()
        self.target_name = helpers.sync_scaled_target_into_mplib_world(task)["target_name"]
        self.target = self.world.get_object(self.target_name)
        if self.target is None or len(self.world.get_planned_articulations()) != 1:
            raise RecoveryContractError(
                "missing target or unsupported multiple planned articulations"
            )
        self.names = list(self.inner.user_link_names)
        if not {"gripper_L_link", "gripper_R_link"}.issubset(self.names):
            raise RecoveryContractError("missing native finger FK links")
        self.joint_names = list(self.inner.user_joint_names)
        actual_names = [j.get_name() for j in task.robot.left_entity.get_active_joints()]
        if self.joint_names != actual_names:
            raise RecoveryContractError("MPlib/simulator full joint ordering mismatch")
        self.collision_objects = self.model.get_fcl_model().get_collision_objects()
        if not self.collision_objects:
            raise RecoveryContractError("no robot FCL collision geometry")
        for link in self.model.get_fcl_model().get_collision_link_names():
            if self.acm.get_allowed_collision(link, self.target_name) not in (
                None,
                AllowedCollision.NEVER,
            ):
                raise RecoveryContractError("robot-target contact exemption present")
        self.distance_request = native_distance_request(fcl)
        self.contact_request = fcl.CollisionRequest(num_max_contacts=32, enable_contact=True)
        sphere = fcl.Sphere(radius=0.1)
        a = fcl.CollisionObject(sphere, mplib.Pose(p=[0, 0, 0]))
        distances = []
        for x in (0.3, 0.15):
            b = fcl.CollisionObject(sphere, mplib.Pose(p=[x, 0, 0]))
            distances.append(float(fcl.distance(a, b, self.distance_request).min_distance))
        self.distance_selftest = validate_distance_selftest(*distances)
        # FK root pose comes from MPlib itself. Runtime comparison catches CV/GL,
        # local/world, root transform and joint-order mismatches.
        self.root_pose = self.model.get_pose()
        measured = {
            n: pose_vector(task.robot.left_entity.find_link_by_name(n).get_pose())
            for n in self.names
        }
        saved = self.snapshot()
        try:
            self.model.set_qpos(task.robot.left_entity.get_qpos(), True)
            computed = self._fk()
            self.fk_selftest = validate_fk_agreement(computed, measured)
        finally:
            self.restore(saved)
        entries = sorted(self.acm.get_all_entry_names())
        scope = {
            "objects": sorted(self.world.get_object_names()),
            "articulations": sorted(self.world.get_articulation_names()),
            "collision_links": self.model.get_fcl_model().get_collision_link_names(),
            "self_pairs": self.model.get_fcl_model().get_collision_pairs(),
            "target": self.target_name,
            "acm_defaults": {n: str(self.acm.get_default_entry(n)) for n in entries},
            "acm_entries": [
                [a, b, str(self.acm.get_entry(a, b))]
                for i, a in enumerate(entries)
                for b in entries[i:]
            ],
            "root_pose_world_wxyz": pose_vector(self.root_pose),
            "target_pose_world_wxyz": pose_vector(task.pillbottle.get_pose()),
            "target_geometry": helpers.actor_box_from_collision_mesh(task, Path.cwd()),
        }
        self.scope = scope
        self.scope_sha256 = hashlib.sha256(
            json.dumps(scope, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self.provenance = {
            "versions": versions,
            "mplib_planner_sha256": digest(planner_path),
            "mplib_extension_sha256": digest(Path(mplib.pymp.__file__)),
            "distance_selftest": self.distance_selftest,
            "distance_query": {
                "solver": "GST_INDEP",
                "distance_tolerance_m": 1e-6,
                "signed_distance": True,
                "witness_contract": "one_solver_point_on_first_object",
            },
            "fk_selftest": self.fk_selftest,
            "scope": scope,
            "scope_sha256": self.scope_sha256,
            "unit": "m",
            "contact_semantics": "FCL points/penetration, NOT PhysX impulses",
        }

    def snapshot(self) -> list[float]:
        return [float(v) for v in self.model.get_qpos()]

    def restore(self, qpos: Any) -> None:
        self.model.set_qpos(qpos, True)
        self.pino.compute_forward_kinematics(qpos)

    def _fk(self) -> dict[str, list[float]]:
        self.pino.compute_forward_kinematics(self.model.get_qpos())
        return {
            name: pose_vector(
                self.root_pose * self.pino.get_link_pose(self.inner.link_name_2_idx[name])
            )
            for name in self.names
        }

    @staticmethod
    def _pair(hit: Any) -> dict[str, str]:
        return {
            name: str(getattr(hit, name))
            for name in ("object_name1", "object_name2", "link_name1", "link_name2")
        }

    def sample(self, qpos: list[float]) -> dict[str, Any]:
        import numpy as np

        full = np.asarray(self.task.robot.left_entity.get_qpos(), dtype=float).copy()
        arm_names = [f"joint{i}" for i in range(1, 7)]
        if len(qpos) != 6:
            raise RecoveryContractError("native collector requires six arm joints")
        for name, q in zip(arm_names, qpos, strict=True):
            full[self.joint_names.index(name)] = q
        self.model.set_qpos(full, True)
        hits = self.world.check_collision(self.contact_request)
        pairs = []
        for hit in hits:
            points = [
                {
                    "position_world_m": [float(v) for v in c.pos],
                    "normal_world": [float(v) for v in c.normal],
                    "penetration_depth_m": float(c.penetration_depth),
                }
                for c in hit.res.get_contacts()
            ]
            pairs.append(
                {
                    **self._pair(hit),
                    "collision_type": str(hit.collision_type),
                    "contacts": points,
                    "contact_cap_reached": len(points) >= 32,
                }
            )
        distances = {}
        for label, fn in (("world", self.world.distance_robot), ("self", self.world.distance_self)):
            hit = fn(self.distance_request)
            value = float(hit.min_distance)
            if not math.isfinite(value) or abs(value) > 100:
                raise RecoveryContractError("distance result is missing/unbounded/invalid")
            distances[label] = {**self._pair(hit), "signed_distance_m": value}
        # Direct each-link-to-target measurements; raw geometry, NOT an ACM change.
        target_distances = []
        for obj in self.collision_objects:
            result = self.fcl.distance(obj, self.target, self.distance_request)
            target_distances.append(single_witness_distance_record(str(obj.name), result))
        return {
            "full_qpos": full.tolist(),
            "full_joint_names": self.joint_names,
            "fk_link_poses_world_wxyz": self._fk(),
            "collision_pairs": pairs,
            "distance_minima_acm_filtered": distances,
            "target_link_distances": target_distances,
            "target_signed_distance_m": min(d["signed_distance_m"] for d in target_distances),
            "scope_sha256": self.scope_sha256,
        }

    def collect(self, rows: Any, times: Any, output: Path, **metadata: Any) -> dict[str, Any]:
        before = [float(v) for v in self.task.robot.left_entity.get_qpos()]
        result = collect_plan_samples(self, rows, times)
        if before != [float(v) for v in self.task.robot.left_entity.get_qpos()]:
            raise RecoveryContractError("simulator qpos changed during plan-only collection")
        result.update(
            metadata,
            provenance=self.provenance,
            collected_at_utc=datetime.now(timezone.utc).isoformat(),
            observation_association=(
                "frozen simulator state during planner-only query; not sensor timestamp"
            ),
            simulator_full_qpos_before=before,
            simulator_full_qpos_after=[float(v) for v in self.task.robot.left_entity.get_qpos()],
        )
        write_new_json(output, result)
        return result


def audit_native_plan_trace(data: Any) -> dict[str, Any]:
    """Independent structural veto; integrity is not physical collision proof."""
    errors = []
    try:
        if not isinstance(data, dict) or data.get("schema") != "s1m.plan-samples.v2":
            raise RecoveryContractError("invalid plan trace schema")
        if (
            data.get("source_kind") != "native_planner_geometry"
            or data.get("execution_status") != "not_executed"
            or data.get("controller_eligible") is not False
            or data.get("native_acceptance") is not False
            or data.get("physics_steps_during_collection") != 0
        ):
            raise RecoveryContractError("native plan-only provenance/flags required")
        samples = data.get("samples")
        if (
            not isinstance(samples, list)
            or not samples
            or data.get("waypoint_count") != len(samples)
        ):
            raise RecoveryContractError("missing/inconsistent waypoint inventory")
        provenance = data.get("provenance")
        if not isinstance(provenance, dict):
            raise RecoveryContractError("missing native provenance")
        if provenance.get("versions") != {"mplib": "0.2.1", "sapien": "3.0.0b1"}:
            raise RecoveryContractError("unsupported native versions")
        if provenance.get("mplib_planner_sha256") != MPLIB_PLANNER_SHA256:
            raise RecoveryContractError("wrong native planner source")
        scope = provenance.get("scope")
        scope_hash = hashlib.sha256(
            json.dumps(scope, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if provenance.get("scope_sha256") != scope_hash:
            raise RecoveryContractError("geometry scope checksum mismatch")
        test = provenance.get("distance_selftest", {})
        validate_distance_selftest(test.get("separated_m"), test.get("overlapping_m"))
        if provenance.get("fk_selftest", {}).get("passed") is not True:
            raise RecoveryContractError("missing native FK comparison")
        if provenance.get("distance_query") != {
            "solver": "GST_INDEP",
            "distance_tolerance_m": 1e-6,
            "signed_distance": True,
            "witness_contract": "one_solver_point_on_first_object",
        }:
            raise RecoveryContractError("unsupported native FCL distance contract")
        previous = -1.0
        for index, sample in enumerate(samples):
            if sample.get("sample_index") != index or sample.get("scope_sha256") != scope_hash:
                raise RecoveryContractError("sample index/scope mismatch")
            _validate_sample(sample)
            finite_vector(sample.get("qpos_rad"), 6)
            timestamp = finite_vector([sample.get("time_from_start_s")], 1)[0]
            if timestamp < 0 or timestamp <= previous:
                raise RecoveryContractError("invalid trace timing")
            previous = timestamp
            distances = sample.get("target_link_distances")
            if not isinstance(distances, list) or not distances:
                raise RecoveryContractError("missing link-target distances")
            values = []
            for distance in distances:
                if not isinstance(distance.get("robot_collision_object"), str):
                    raise RecoveryContractError("missing collision object identity")
                value = finite_vector([distance.get("signed_distance_m")], 1)[0]
                if (
                    distance.get("witness_contract") != "one_solver_point_on_first_object"
                    or "nearest_points_world_m" in distance
                ):
                    raise RecoveryContractError("ambiguous FCL witness contract")
                finite_vector(distance.get("nearest_point_world_m"), 3)
                if abs(value) > 100:
                    raise RecoveryContractError("unbounded target distance")
                values.append(value)
            if min(values) != sample["target_signed_distance_m"]:
                raise RecoveryContractError("target clearance minimum mismatch")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "passed": not errors,
        "errors": errors,
        "controller_eligible": False,
        "native_acceptance": False,
        "claim": "structural veto only; not PhysX/swept-volume proof",
    }


def _validate_sample(row: Any) -> None:
    if not isinstance(row, dict):
        raise RecoveryContractError("native sample is not a mapping")
    finite_vector(row.get("full_qpos"))
    links = row.get("fk_link_poses_world_wxyz")
    if not isinstance(links, dict) or not links:
        raise RecoveryContractError("missing FK link records")
    validate_fk_agreement(links, links)
    if not isinstance(row.get("collision_pairs"), list):
        raise RecoveryContractError("missing collision records")
    distance = finite_vector([row.get("target_signed_distance_m")], 1)[0]
    if abs(distance) > 100:
        raise RecoveryContractError("invalid/unbounded target distance")
