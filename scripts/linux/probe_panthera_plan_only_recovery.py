#!/usr/bin/env python3
"""Bounded non-actuating candidate plans on the canonical frozen initial scene.

RoboTwin setup may settle physics. Afterwards a physics-step observer refuses
stepping. Only planner model states are predicted; no pregrasp is executed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vla_robot_sorting.planning.manipulation_recovery import (
    TASK_CONFIG,
    UPSTREAM_SHA,
    RecoveryContractError,
    _git,
    verify_source_lock,
    write_new_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    root, upstream = args.project_root.resolve(), args.robotwin_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "s1m.initial-plan-diagnostic.v1",
        "seed": args.seed,
        "task_config": TASK_CONFIG,
        "classification": "PLAN_ONLY_DIAGNOSTIC_NOT_ACCEPTED",
        "diagnostic_complete": False,
        "native_acceptance": False,
        "controller_eligible": False,
        "executed": False,
        "trials": [],
        "physics_scope": "setup settling excluded; no steps permitted during planning",
    }
    task = None
    try:
        verify_source_lock(root, json.loads(args.source_lock.read_text(encoding="utf-8")))
        if _git(upstream, "rev-parse", "HEAD") != UPSTREAM_SHA:
            raise RecoveryContractError("RoboTwin Git pin mismatch")
        record["project_git_sha"] = _git(root, "rev-parse", "HEAD")
        record["robotwin_git_sha"] = UPSTREAM_SHA
        os.chdir(upstream)
        sys.path.insert(0, str(upstream))
        import mplib
        from script.collect_data import prepare_task_and_args

        from vla_robot_sorting.contracts.grasping import GraspRequest
        from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator
        from vla_robot_sorting.planning.native_initial_home import (
            initialize_native_task_with_home,
        )
        from vla_robot_sorting.planning.native_plan_diagnostics import NativeMPlibDiagnosticBackend
        from vla_robot_sorting.simulation.render_profile import apply_fast_preview_render_profile

        spec = importlib.util.spec_from_file_location(
            "plan_only_helpers", root / "scripts/linux/probe_panthera_grasp_recovery.py"
        )
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        task, task_args = prepare_task_and_args("move_pillbottle_pad", TASK_CONFIG)
        task_args.update(render_freq=0, save_data=False, collect_data=False)
        for key in ("cluttered_table", "random_background", "random_light"):
            task_args["domain_randomization"][key] = False
        apply_fast_preview_render_profile(task_args)
        task_args, record["initial_home"] = initialize_native_task_with_home(
            task, task_args, seed=args.seed
        )
        if record["initial_home"]["measured"]["passed"] is not True:
            raise RecoveryContractError("native initial joint bounds/reserve failed")

        class NoPhysicsAfterSetup:
            def on_physics_step(self, task):
                raise RecoveryContractError("unexpected physics step during plan-only diagnostic")

        task.add_physics_step_observer(NoPhysicsAfterSetup())
        backend = NativeMPlibDiagnosticBackend(task, helpers)
        record["api_checks"] = backend.provenance
        geometry = helpers.actor_box_from_collision_mesh(task, upstream)
        gripper, provenance = helpers.load_gripper_proxy(upstream, task_args)
        scene = helpers.make_scene_snapshot(
            task,
            revision=f"frozen-initial-seed-{args.seed}",
            geometry_info=geometry,
            transform_revision="robotwin-static-v1",
        )
        gate = json.loads(
            (root / "configs/s1.3/grasp_recovery_gates.json").read_text(encoding="utf-8")
        )
        generator = GeometryFirstGraspGenerator(
            cylinder_side_grasp_height_fractions=tuple(
                gate["cylinder_side_grasp_height_fractions"]
            ),
            box_side_grasp_height_fractions=tuple(
                gate.get("box_side_grasp_height_fractions", (0.0,))
            ),
            box_parallel_jaw_roll_variants=bool(gate.get("box_parallel_jaw_roll_variants", False)),
        )
        generation = generator.generate(
            GraspRequest(
                object_id="pillbottle",
                scene_revision=scene.scene_revision,
                transform_revision=scene.transform_revision,
                gripper=gripper,
                minimum_table_clearance_m=float(gate["minimum_table_clearance_m"]),
            ),
            scene,
        )
        if not generation.succeeded:
            raise RecoveryContractError("geometry-first candidate generation failed")
        ids = {
            0: ("pillbottle:cyl:side:1:s-1",),
            1: ("pillbottle:cyl:side:0:s-1:up75",),
            2: ("pillbottle:cyl:side:2:s-1", "pillbottle:cyl:side:3:s-1"),
        }
        candidates = {c.candidate_id: c for c in generation.candidates}
        if any(name not in candidates for name in ids[args.seed]):
            raise RecoveryContractError(
                "historical candidate absent from current canonical geometry"
            )
        initial_q = task.robot.left_entity.get_qpos().copy()
        initial_model_q = backend.snapshot()
        record["gripper_geometry"] = provenance
        record["object_geometry"] = geometry
        for index, name in enumerate(ids[args.seed]):
            mplib.set_global_seed(args.seed)
            candidate = candidates[name]
            trial = {
                "candidate": helpers.candidate_dict(candidate),
                "stages": [],
                "start_kind": "predicted_planner_state_not_executed_pregrasp",
            }
            q = initial_q.copy()
            for stage, pose in (
                ("pregrasp", candidate.pregrasp_pose),
                ("approach", candidate.grasp_pose),
            ):
                target = task.robot._trans_from_gripper_to_endlink(
                    [*pose.position_xyz, pose.orientation_xyzw[3], *pose.orientation_xyzw[:3]],
                    arm_tag="left",
                )
                try:
                    if stage == "pregrasp":
                        native = task.robot.left_mplib_planner.plan_pose(
                            q.copy(), target, log=False
                        )
                    else:
                        native = task.robot.left_mplib_planner.plan_screw(
                            q.copy(), target, log=False
                        )
                    from vla_robot_sorting.planning.native_plan_diagnostics import pose_vector
                    from vla_robot_sorting.planning.native_recovery_ik import (
                        NativeRecoveryIKBackend,
                    )
                    from vla_robot_sorting.planning.recovery_query_diagnostics import (
                        collect_ik_diagnostics,
                    )

                    raw_backend = NativeRecoveryIKBackend(backend)
                    query = collect_ik_diagnostics(
                        raw_backend,
                        pose_vector(target),
                        phase=stage,
                        start_full_qpos=q.tolist(),
                        start_kind="predicted_native_pregrasp"
                        if stage == "approach"
                        else "live_measured",
                    )
                    query.update(
                        provenance=raw_backend.provenance,
                        project_git_sha=record["project_git_sha"],
                        seed=args.seed,
                        candidate_id=name,
                        scene_revision=scene.scene_revision,
                    )
                    query_file = f"ik-{index}-{stage}.json"
                    write_new_json(output / query_file, query)
                    summary = {
                        "raw_ik_diagnostic_file": query_file,
                        "stage": stage,
                        "native_status": str(native.get("status")),
                        "plan_success": native.get("status") == "Success",
                        "executed": False,
                    }
                    if native.get("status") == "Success":
                        from vla_robot_sorting.planning.panthera_robotwin import (
                            _extract_panthera_rows,
                        )

                        rows = _extract_panthera_rows(
                            native["position"], task.robot, label="diagnostic plan"
                        )
                        times = [float(t) for t in native["time"].reshape(-1)]
                        artifact = f"candidate-{index}-{stage}.json"
                        samples = backend.collect(
                            rows,
                            times,
                            output / artifact,
                            candidate_id=name,
                            stage=stage,
                            source_time="MPlib TOPP time",
                            scene_revision=scene.scene_revision,
                            transform_revision=scene.transform_revision,
                            contact_policy={"allow_robot_target_contact": False},
                        )
                        summary.update(
                            artifact=artifact,
                            waypoint_count=len(rows),
                            collision_waypoint_count=sum(
                                bool(s["collision_pairs"]) for s in samples["samples"]
                            ),
                            minimum_target_signed_distance_m=min(
                                s["target_signed_distance_m"] for s in samples["samples"]
                            ),
                        )
                        q = helpers.expand_arm_qpos(task, rows[-1])
                    else:
                        # Record the failed query's start state, not an invented path.
                        sample = tuple(
                            float(q[backend.joint_names.index(f"joint{i}")]) for i in range(1, 7)
                        )
                        backend.collect(
                            [sample],
                            [0.0],
                            output / f"candidate-{index}-{stage}-failed-start.json",
                            candidate_id=name,
                            stage=stage,
                            plan_success=False,
                        )
                    trial["stages"].append(summary)
                finally:
                    backend.restore(initial_model_q)
                if native.get("status") != "Success":
                    break
            record["trials"].append(trial)
        if initial_q.tolist() != task.robot.left_entity.get_qpos().tolist():
            raise RecoveryContractError("simulator qpos changed during planning")
        record["diagnostic_complete"] = True
        record["physics_steps_during_planning"] = 0
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        if task is not None:
            try:
                task.clear_physics_step_observers()
                task.close()
            except Exception as exc:
                record["close_error"] = repr(exc)
                record["diagnostic_complete"] = False
        write_new_json(output / "diagnostic.json", record)
    print(
        json.dumps(
            {
                "diagnostic_complete": record["diagnostic_complete"],
                "classification": record["classification"],
                "error": record.get("error"),
            }
        )
    )
    return 0 if record["diagnostic_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
