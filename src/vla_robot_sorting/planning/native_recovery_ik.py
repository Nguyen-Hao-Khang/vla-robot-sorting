"""Runtime-checked MPlib 0.2.1 raw IK diagnostic adapter. Never executes a path."""

from __future__ import annotations

import math

import numpy as np

from vla_robot_sorting.planning.manipulation_recovery import RecoveryContractError
from vla_robot_sorting.planning.native_plan_diagnostics import (
    collect_plan_samples,
    finite_vector,
    pose_vector,
)


class NativeRecoveryIKBackend:
    source_kind = "native_planner_geometry"

    def __init__(self, geometry):
        self.geometry = geometry
        self.inner = geometry.inner
        self.pino = geometry.pino
        self.joint_names = list(geometry.joint_names)
        self.arm_indices = [self.joint_names.index(f"joint{i}") for i in range(1, 7)]
        if list(self.inner.move_group_joint_indices) != self.arm_indices:
            raise RecoveryContractError("unsupported move-group mapping")
        types = list(self.inner.joint_types)
        if len(types) != len(self.joint_names) or any(
            not t.startswith(("JointModelR", "JointModelP")) for t in types
        ):
            raise RecoveryContractError("unsupported native joint coordinate units")
        self.joint_units = ["rad" if t.startswith("JointModelR") else "m" for t in types]
        limits = np.asarray(self.inner.joint_limits, dtype=float)
        if limits.shape != (len(self.joint_names), 2) or not np.isfinite(limits).all():
            raise RecoveryContractError("unsupported native joint-limit mapping")
        self.lower, self.upper = limits[:, 0].tolist(), limits[:, 1].tolist()
        if not callable(getattr(self.pino, "compute_IK_CLIK", None)):
            raise RecoveryContractError("pinned raw CLIK API unavailable")
        self.provenance = {
            **geometry.provenance,
            "raw_ik_api": "PinocchioModel.compute_IK_CLIK (NOT clamped _JL)",
            "max_iter": 1000,
            "eps": 1e-5,
            "dt": 0.1,
            "damp": 1e-12,
        }

    def measured_start(self):
        return self.geometry.task.robot.left_entity.get_qpos().tolist()

    def snapshot(self):
        return self.geometry.snapshot()

    def restore(self, q):
        self.geometry.restore(q)

    def solve(self, goal, initial):
        world = self.geometry.Pose(p=goal[:3], q=goal[3:])
        base = self.geometry.root_pose.inv() * world
        mask = [i not in self.arm_indices for i in range(len(initial))]
        raw = self.pino.compute_IK_CLIK(
            self.inner.move_group_link_id,
            base,
            np.asarray(initial, dtype=float).copy(),
            mask=mask,
            eps=1e-5,
            max_iter=1000,
            dt=0.1,
            damp=1e-12,
        )
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise RecoveryContractError("unsupported raw CLIK return semantics")
        if not isinstance(raw[1], (bool, np.bool_)):
            raise RecoveryContractError("raw CLIK convergence semantics changed")
        q = finite_vector(np.asarray(raw[0]).reshape(-1).tolist(), len(initial))
        error = finite_vector(np.asarray(raw[2]).reshape(-1).tolist(), 6)
        measured = self.geometry.task.robot.left_entity.get_qpos().tolist()
        if any(q[i] != measured[i] for i in range(len(q)) if i not in self.arm_indices):
            raise RecoveryContractError("raw CLIK changed a masked non-arm joint")
        sample = self.geometry.sample([q[i] for i in self.arm_indices])
        self.pino.compute_forward_kinematics(np.asarray(q))
        achieved = self.pino.get_link_pose(self.inner.move_group_link_id)
        residual = float(base.distance(achieved))
        if not math.isfinite(residual):
            raise RecoveryContractError("nonfinite native FK residual")
        return {
            **sample,
            "full_qpos": q,
            "solver_converged": bool(raw[1]),
            "solver_error_6d": error,
            "fk_goal_error": residual,
            "residual_semantics": "MPlib Pose.distance; not calibrated Cartesian accuracy",
            "goal_endlink_base_wxyz": pose_vector(base),
        }

    def plan_qpos(self, goal, start, *, fix_joint_limits, planning_time):
        if fix_joint_limits is not False or planning_time != 5:
            raise RecoveryContractError("RRT budget or no-clipping contract changed")
        start_sample = self.geometry.sample([start[i] for i in self.arm_indices])
        if start_sample["collision_pairs"]:
            return {"status": "START_COLLISION_NO_RRT_CALL", "start_sample": start_sample}
        import mplib

        mplib.set_global_seed(0)
        raw = self.inner.plan_qpos(
            [np.asarray(goal, dtype=float)],
            np.asarray(start, dtype=float),
            time_step=0.004,
            planning_time=5,
            fix_joint_limits=False,
            fixed_joint_indices=[],
        )
        result = {
            "status": str(raw.get("status")),
            "random_seed": 0,
            "fix_joint_limits": False,
            "planning_time_s": 5,
            "physics_steps": 0,
        }
        if raw.get("status") == "Success":
            from vla_robot_sorting.planning.panthera_robotwin import _extract_panthera_rows

            rows = _extract_panthera_rows(
                raw["position"], self.geometry.task.robot, label="diagnostic RRT"
            )
            times = np.asarray(raw["time"]).reshape(-1).tolist()
            result["plan_trace"] = collect_plan_samples(self.geometry, rows, times)
            result["plan_trace"]["provenance"] = self.geometry.provenance
            arm_lower = [self.lower[i] for i in self.arm_indices]
            arm_upper = [self.upper[i] for i in self.arm_indices]
            result["exact_bound_violation_indices"] = [
                i
                for i, row in enumerate(rows)
                if any(
                    q < lo or q > hi for q, lo, hi in zip(row, arm_lower, arm_upper, strict=True)
                )
            ]
            result["collision_sample_indices"] = [
                i
                for i, sample in enumerate(result["plan_trace"]["samples"])
                if sample["collision_pairs"]
            ]
            result["scoped_sample_checks_pass"] = (
                not result["exact_bound_violation_indices"]
                and not result["collision_sample_indices"]
            )

        return result
