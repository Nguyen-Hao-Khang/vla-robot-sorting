#!/usr/bin/env python3
"""Native non-actuating Sprint-1 A-007 planner-failure fail-closed probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--task-config", default="move_pillbottle_pad_panthera_single")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@dataclass(slots=True)
class InjectedFailurePlanner:
    """Typed project-boundary fault injection; never returns a path."""

    code: Any

    def plan(self, start_state: Any, goal: Any, context: Any, scene: Any) -> Any:
        del start_state, goal, context, scene
        from vla_robot_sorting.contracts import MotionFailure, PlannerResult

        return PlannerResult(
            path=None,
            failure=MotionFailure(
                code=self.code,
                message=f"controlled A-007 fault injection: {self.code.value}",
                recoverable=True,
                details={"fault_injection": True},
            ),
            planning_time_s=0.0,
            planner_id="a007-controlled-fault-injection",
        )


@dataclass(slots=True)
class CountingParameterizer:
    calls: int = 0

    def parameterize(self, path: Any, limits: Any) -> Any:
        del path, limits
        self.calls += 1
        raise AssertionError("parameterizer must not run after planner failure")


@dataclass(slots=True)
class CountingSafety:
    path_calls: int = 0
    trajectory_calls: int = 0

    def validate_path(self, path: Any, context: Any, scene: Any) -> Any:
        del path, context, scene
        self.path_calls += 1
        raise AssertionError("path safety must not run after planner failure")

    def validate_trajectory(
        self,
        trajectory: Any,
        limits: Any,
        context: Any,
        scene: Any,
    ) -> Any:
        del trajectory, limits, context, scene
        self.trajectory_calls += 1
        raise AssertionError("trajectory safety must not run after planner failure")


@dataclass(slots=True)
class CountingExecutor:
    delegate: Any
    calls: int = 0

    def execute(self, segment: Any) -> Any:
        self.calls += 1
        return self.delegate.execute(segment)


@dataclass(slots=True)
class PhysicsStepCounter:
    count: int = 0

    def on_physics_step(self, task: Any) -> None:
        del task
        self.count += 1


def main() -> int:
    args = parse_args()
    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()

    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))

    import numpy as np
    import yaml
    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.contracts import (
        JointState,
        MotionFailureCode,
        MotionGoal,
        PlanningContext,
        Pose6D,
    )
    from vla_robot_sorting.control import PantheraRoboTwinTrajectoryExecutor
    from vla_robot_sorting.orchestration import BoundedMotionController, MotionCycleStatus
    from vla_robot_sorting.planning import load_joint_limits_from_urdf
    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES
    from vla_robot_sorting.simulation.render_profile import apply_fast_preview_render_profile

    task, task_args = prepare_task_and_args(args.task, args.task_config)
    task_args["render_freq"] = 0
    task_args["save_data"] = False
    task_args["collect_data"] = False
    task_args["domain_randomization"]["cluttered_table"] = False
    task_args["domain_randomization"]["random_background"] = False
    task_args["domain_randomization"]["random_light"] = False
    apply_fast_preview_render_profile(task_args)

    initialized = False
    try:
        task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        initialized = True

        actual = tuple(float(v) for v in task.robot.get_left_arm_real_jointState())
        if len(actual) != 7:
            raise RuntimeError("expected six Panthera arm joints plus gripper")
        start_state = JointState(
            joint_names=PANTHERA_JOINT_NAMES,
            positions=actual[:6],
        )
        ee = tuple(float(v) for v in task.robot.get_left_ee_pose())
        if len(ee) != 7:
            raise RuntimeError("expected xyz + wxyz end-effector pose")
        goal = MotionGoal(
            Pose6D(
                position_xyz=ee[:3],
                orientation_xyzw=(ee[4], ee[5], ee[6], ee[3]),
                frame_id="world",
            )
        )
        context = PlanningContext(
            scene_revision=f"a007-seed-{args.seed}",
            transform_revision="robotwin-static-v1",
        )

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

        cases: list[dict[str, Any]] = []
        for code in (MotionFailureCode.TIMEOUT, MotionFailureCode.COLLISION):
            qpos_before = np.asarray(task.robot.left_entity.get_qpos(), dtype=float).copy()
            physics = PhysicsStepCounter()
            task.add_physics_step_observer(physics)

            parameterizer = CountingParameterizer()
            safety = CountingSafety()
            native_executor = PantheraRoboTwinTrajectoryExecutor(
                task,
                scene_revision=lambda revision=context.scene_revision: revision,
                transform_revision=lambda revision=context.transform_revision: revision,
                physics_period_s=1.0 / 250.0,
            )
            executor = CountingExecutor(native_executor)
            controller = BoundedMotionController(
                planner=InjectedFailurePlanner(code),
                parameterizer=parameterizer,
                safety=safety,
                executor=executor,
                limits=limits,
                controller_period_s=1.0 / 250.0,
                horizon_s=0.2,
            )
            result = controller.run(
                start_state,
                goal,
                context,
                scene={},
                segment_id=f"a007-{code.value}",
            )
            task.clear_physics_step_observers()
            qpos_after = np.asarray(task.robot.left_entity.get_qpos(), dtype=float).copy()

            failure_code = result.plan.failure.code.value if result.plan.failure else None
            passed = (
                result.status is MotionCycleStatus.PLANNING_FAILED
                and failure_code == code.value
                and result.trajectory is None
                and result.execution is None
                and parameterizer.calls == 0
                and safety.path_calls == 0
                and safety.trajectory_calls == 0
                and executor.calls == 0
                and physics.count == 0
                and np.array_equal(qpos_before, qpos_after)
            )
            cases.append(
                {
                    "fault": code.value,
                    "passed": passed,
                    "motion_cycle_status": result.status.value,
                    "planner_id": result.plan.planner_id,
                    "planner_failure_code": failure_code,
                    "trajectory_created": result.trajectory is not None,
                    "execution_created": result.execution is not None,
                    "parameterizer_calls": parameterizer.calls,
                    "path_safety_calls": safety.path_calls,
                    "trajectory_safety_calls": safety.trajectory_calls,
                    "executor_calls": executor.calls,
                    "physics_steps": physics.count,
                    "qpos_unchanged": bool(np.array_equal(qpos_before, qpos_after)),
                }
            )

        payload = {
            "schema": "panthera-native-safe-failure-v1",
            "scenario_id": "S1-A-007",
            "task": args.task,
            "task_config": args.task_config,
            "seed": args.seed,
            "actual_joint_positions_rad": list(start_state.positions),
            "cases": cases,
            "pass": all(item["passed"] for item in cases) and len(cases) == 2,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 0 if payload["pass"] else 2
    finally:
        if initialized:
            task.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
