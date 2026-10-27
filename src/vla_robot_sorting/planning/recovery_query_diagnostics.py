"""Bounded recovery research. Diagnostic results never authorize robot commands."""

from __future__ import annotations

import math
from collections import deque
from copy import deepcopy

from vla_robot_sorting.planning.manipulation_recovery import RecoveryContractError
from vla_robot_sorting.planning.native_plan_diagnostics import finite_vector, validate_fk_agreement


def _bounds(q, lower, upper):
    q, lower, upper = finite_vector(q), finite_vector(lower), finite_vector(upper)
    if (
        len(q) != len(lower)
        or len(q) != len(upper)
        or any(a >= b for a, b in zip(lower, upper, strict=True))
    ):
        raise RecoveryContractError("invalid exact joint limits")
    margins = [min(v - a, b - v) for v, a, b in zip(q, lower, upper, strict=True)]
    return {
        "within_bounds": all(m >= 0 for m in margins),
        "joint_margins_native_units": margins,
        "qpos_modified": False,
    }


def deterministic_solver_starts(current, lower, upper, arm_indices):
    current = finite_vector(current)
    if not _bounds(current, lower, upper)["within_bounds"]:
        raise RecoveryContractError("measured full state outside exact limits; no clipping")
    if (
        len(arm_indices) != 6
        or len(set(arm_indices)) != 6
        or any(type(i) is not int or i < 0 or i >= len(current) for i in arm_indices)
    ):
        raise RecoveryContractError("invalid six-arm-joint mapping")
    midpoint = current.copy()
    for i in arm_indices:
        midpoint[i] = (lower[i] + upper[i]) / 2
    halfway = [(a + b) / 2 for a, b in zip(current, midpoint, strict=True)]
    return [
        {"label": label, "full_qpos": q, "role": "solver_initialization_not_robot_start"}
        for label, q in [("measured", current.copy()), ("midpoint", midpoint), ("halfway", halfway)]
    ]


def collect_ik_diagnostics(
    backend,
    goal,
    *,
    phase: str,
    allow_rrt: bool = False,
    start_full_qpos=None,
    start_kind="live_measured",
):
    """Exactly three unclamped CLIK queries and at most one non-actuating RRT query."""
    goal = finite_vector(goal, 7)
    if abs(sum(v * v for v in goal[3:]) - 1) > 1e-6:
        raise RecoveryContractError("goal quaternion must be normalized wxyz")
    if allow_rrt and phase != "retreat":
        raise RecoveryContractError("fixed-target RRT cannot model a carried-object phase")
    saved = finite_vector(backend.snapshot())
    current = finite_vector(
        start_full_qpos
        if start_full_qpos is not None
        else getattr(backend, "measured_start", backend.snapshot)(),
        len(saved),
    )
    if start_kind not in ("live_measured", "predicted_native_pregrasp"):
        raise RecoveryContractError("unsupported query start provenance")
    start_bounds = _bounds(current, backend.lower, backend.upper)
    if not start_bounds["within_bounds"]:
        return {
            "schema": "s1m.raw-ik-query.v1",
            "source_kind": backend.source_kind,
            "query_start_full_qpos": current,
            "start_state_kind": start_kind,
            "goal_endlink_world_wxyz": goal,
            "full_joint_names": backend.joint_names,
            "rejected_start": True,
            "start_joint_bounds": start_bounds,
            "joint_coordinate_units": getattr(backend, "joint_units", None),
            "queries": [],
            "rrt": None,
            "executed": False,
            "controller_eligible": False,
            "native_acceptance": False,
            "reason": "exact measured bounds reject; zero solver calls",
        }
    starts = deterministic_solver_starts(current, backend.lower, backend.upper, backend.arm_indices)
    queries = []
    rrt = None
    try:
        for seed in starts:
            raw = backend.solve(goal, seed["full_qpos"].copy())
            q = finite_vector(raw.get("full_qpos"), len(saved))
            if any(q[i] != current[i] for i in range(len(q)) if i not in backend.arm_indices):
                raise RecoveryContractError("solver changed a masked non-arm joint")

            if type(raw.get("solver_converged")) is not bool:
                raise RecoveryContractError("missing raw solver convergence flag")
            finite_vector(raw.get("solver_error_6d"), 6)
            residual = finite_vector([raw.get("fk_goal_error")], 1)[0]
            if residual < 0:
                raise RecoveryContractError("negative FK residual")
            links = raw.get("fk_link_poses_world_wxyz")
            if not isinstance(links, dict):
                raise RecoveryContractError("missing per-solution FK")
            validate_fk_agreement(links, links)
            pairs = raw.get("collision_pairs")
            if not isinstance(pairs, list):
                raise RecoveryContractError("missing raw collision records")
            finite_vector([raw.get("target_signed_distance_m")], 1)
            bounds = _bounds(q, backend.lower, backend.upper)
            bounds["arm_minimum_margin_rad"] = min(
                bounds["joint_margins_native_units"][i] for i in backend.arm_indices
            )
            reason = (
                "solver_not_converged"
                if not raw["solver_converged"]
                else "outside_joint_bounds"
                if not bounds["within_bounds"]
                else "fk_residual_failed"
                if residual >= 1e-3
                else "collision_filtered"
                if pairs
                else "scoped_geometric_candidate"
            )
            queries.append(
                {
                    **raw,
                    "solver_seed": seed,
                    "joint_bounds": bounds,
                    "classification": reason,
                    "controller_eligible": False,
                    "native_acceptance": False,
                }
            )
            backend.restore(saved)
        candidates = [q for q in queries if q["classification"] == "scoped_geometric_candidate"]
        candidates.sort(
            key=lambda q: (
                -q["joint_bounds"]["arm_minimum_margin_rad"],
                q["fk_goal_error"],
                q["solver_seed"]["label"],
            )
        )
        if allow_rrt and candidates:
            rrt = backend.plan_qpos(
                candidates[0]["full_qpos"].copy(),
                current.copy(),
                fix_joint_limits=False,
                planning_time=5,
            )
            if not isinstance(rrt, dict):
                raise RecoveryContractError("malformed RRT diagnostics")
            rrt = {
                **rrt,
                "executed": False,
                "controller_eligible": False,
                "native_acceptance": False,
            }
    finally:
        backend.restore(saved)
        if finite_vector(backend.snapshot()) != saved:
            raise RecoveryContractError("planner state restoration failed")
    return {
        "schema": "s1m.raw-ik-query.v1",
        "source_kind": backend.source_kind,
        "phase": phase,
        "goal_endlink_world_wxyz": goal,
        "query_start_full_qpos": current,
        "start_state_kind": start_kind,
        "full_joint_names": backend.joint_names,
        "joint_coordinate_units": getattr(backend, "joint_units", None),
        "queries": queries,
        "rrt": rrt,
        "maximum_raw_ik_queries": 3,
        "maximum_rrt_queries": 1 if allow_rrt else 0,
        "collision_scope": "current frozen MPlib target; NOT a CuRobo carried-object model",
        "ranking": "larger exact joint margin, then smaller native FK residual; diagnostic only",
        "executed": False,
        "controller_eligible": False,
        "native_acceptance": False,
    }


class MotionDiagnosticCapture:
    """Bounded samples plus independent first/violation/terminal records; never a safety gate."""

    def __init__(self, *, capacity=1024, stride=20, source_kind="unverified_observation"):
        if type(capacity) is not int or capacity < 1 or type(stride) is not int or stride < 1:
            raise RecoveryContractError("invalid capture budget")
        self.capacity, self.stride, self.source_kind = capacity, stride, source_kind
        self.samples = deque(maxlen=capacity)
        self.seen = 0
        self.sampled = 0
        self.command = None
        self.initial = None
        self.first_violation = None
        self.terminal = None

    def on_command(self, index, positions, velocities, time_from_start_s):
        if type(index) is not int or index < 0:
            raise RecoveryContractError("invalid command index")
        positions, velocities = (
            finite_vector(list(positions), 6),
            finite_vector(list(velocities), 6),
        )
        timestamp = finite_vector([time_from_start_s], 1)[0]
        if timestamp < 0:
            raise RecoveryContractError("negative command time")
        self.command = {
            "waypoint_index": index,
            "positions_rad": positions,
            "velocities_rad_s": velocities,
            "time_from_start_s": timestamp,
            "association": "drive command applied before physics",
        }

    def observe(self, step, reader, *, violation=False, terminal=False):
        if type(step) is not int or step < 0 or step < self.seen:
            raise RecoveryContractError("invalid physics step")
        self.seen = step
        if self.initial is not None and step % self.stride != 0 and not violation and not terminal:
            return
        state = reader()
        if not isinstance(state, dict):
            raise RecoveryContractError("malformed live snapshot")
        finite_vector(state.get("full_qpos"))
        row = {**deepcopy(state), "physics_step": step, "command": deepcopy(self.command)}
        if self.initial is None:
            self.initial = row
        if violation and self.first_violation is None:
            self.first_violation = row
        if terminal:
            self.terminal = row
        self.samples.append(row)
        self.sampled += 1

    def payload(self):
        displacement = None
        if self.initial and self.terminal:
            a = self.initial.get("gripper_pose_world_xyzw")
            b = self.terminal.get("gripper_pose_world_xyzw")
            if a is not None and b is not None:
                displacement = math.dist(finite_vector(a, 7)[:3], finite_vector(b, 7)[:3])
        return {
            "measured_gripper_displacement_m": displacement,
            "displacement_authorizes_acceptance": False,
            "schema": "s1m.live-command-observation.v1",
            "source_kind": self.source_kind,
            "physics_steps_seen": self.seen,
            "sample_stride_steps": self.stride,
            "capacity": self.capacity,
            "sampled_records": self.sampled,
            "truncated": self.sampled > self.capacity,
            "initial": self.initial,
            "first_violation": self.first_violation,
            "terminal": self.terminal,
            "samples": list(self.samples),
            "controller_eligible": False,
            "native_acceptance": False,
        }


def read_motion_state(task, entity_id, *, source_kind="unverified_observation", link_names=None):
    """Read declared APIs only; missing/nonfinite pose/contact fields are errors."""
    from datetime import datetime, timezone

    from vla_robot_sorting.planning.native_plan_diagnostics import pose_vector

    entity = task.robot.left_entity
    names = [j.get_name() for j in entity.get_active_joints()]
    q = finite_vector(entity.get_qpos().tolist(), len(names))
    if link_names is None:
        poses = {link.get_name(): pose_vector(link.get_pose()) for link in entity.get_links()}
    else:
        if (
            not link_names
            or len(set(link_names)) != len(link_names)
            or any(not isinstance(n, str) for n in link_names)
        ):
            raise RecoveryContractError("invalid verified link-name inventory")
        poses = {
            name: pose_vector(entity.find_link_by_name(name).get_pose()) for name in link_names
        }
    validate_fk_agreement(poses, poses)
    if not {"gripper_L_link", "gripper_R_link", "tool_link"}.issubset(poses):
        raise RecoveryContractError("missing required native link poses")
    contacts = []
    all_contacts = list(task.scene.get_contacts())
    for contact in all_contacts[:64]:
        if len(contact.bodies) != 2:
            raise RecoveryContractError("unsupported contact body pair")
        ids = [entity_id(body) for body in contact.bodies]
        if any(v is None for v in ids):
            raise RecoveryContractError("unresolved native contact body identity")
        points = []
        for point in contact.points[:32]:
            points.append(
                {
                    "position_world_m": finite_vector(point.position.tolist(), 3),
                    "normal_world": finite_vector(point.normal.tolist(), 3),
                    "native_impulse": finite_vector(point.impulse.tolist(), 3),
                    "native_separation": finite_vector([float(point.separation)], 1)[0],
                }
            )
        contacts.append(
            {
                "body_ids": ids,
                "points": points,
                "point_count": len(contact.points),
                "points_truncated": len(contact.points) > 32,
            }
        )
    return {
        "full_qpos": q,
        "full_joint_names": names,
        "link_poses_world_wxyz": poses,
        "physics_contacts": contacts,
        "contact_pair_count": len(all_contacts),
        "contacts_truncated": len(all_contacts) > 64,
        "source_kind": source_kind,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller_eligible": False,
        "native_acceptance": False,
    }


def validate_diagnostic_mode(mode, automatic_acceptance):
    if mode not in ("plan-only", "physical-diagnostic") or automatic_acceptance:
        raise RecoveryContractError("additional diagnostics cannot run/open strict acceptance")
