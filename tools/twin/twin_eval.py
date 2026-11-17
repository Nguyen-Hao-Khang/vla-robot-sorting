#!/usr/bin/env python3
"""Seeded simulated evaluation of the scripted expert or a learned policy (plan item 1.8).

Episode ``i`` of a run with ``--seed S`` uses the episode seed ``S * 100000 + i``. The scene, the
object layout and the command structure (target, destination, references) depend only on that
seed, so runs that differ in language, command source or executor see the same scenes.
Each episode is judged with the success criteria of docs/task-spec.md section 6 and written
as one JSON line; ``summarise`` turns the JSONL into the tables of docs/evaluation-protocol.md.

  # simulator (RoboTwin conda env), expert or policy
  python tools/twin/twin_eval.py run --executor expert --episodes 30 --seed 1 --language en \
      --out results/expert_en.jsonl
  python tools/twin/twin_eval.py run --executor policy --policy-server 127.0.0.1:5555 \
      --checkpoint smolvla_v1_step15000 --episodes 30 --seed 1 --language vi \
      --command-source generated --out results/smolvla_vi.jsonl
  # pipeline check: the expert runs once, its 15 Hz joint targets are replayed through the
  # policy path (safety filter, take_action) on the same scene
  python tools/twin/twin_eval.py run --executor replay --episodes 10 --seed 9 \
      --out results/replay.jsonl
  # anywhere
  python tools/twin/twin_eval.py summarise results/*.jsonl --by executor,language,command_kind

``--record-dir`` (expert executor) also saves each episode in the raw training format of
twin_record.py, so demonstrations can be recorded on exactly the evaluation scenes (for example
to check that a policy reproduces demonstrations it was trained on). With seeds other than the
evaluation seed it records one demonstration per fresh scene.

``--executor dagger`` records corrective demonstrations (DAgger-style relabelling by the expert):
the served policy drives the arm for a seeded random 1-10 s, the gripper opens, and the scripted
expert then completes the command from wherever the policy left the arm and the objects. Only the
expert part is saved (``--record-dir``), so the data shows how to recover from the states the
policy actually reaches:

  python tools/twin/twin_eval.py run --executor dagger --policy-server 127.0.0.1:5555 \
      --checkpoint smolvla_blocks_v1 --episodes 100 --seed 41 --language en \
      --command-source generated --record-dir /mnt/data/raw_dagger --out results/dagger_en.jsonl

``--no-safety`` bypasses the safety filter (its decisions are still counted). It is a diagnostic
only: protocol results run with the filter, and the record says which was used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import eval_metrics as em
import sorting_task as st
import twin_scene as ts

from vla_robot_sorting.safety import ActionSafetyFilter, SafetyConfig, Status

ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = 1
TIME_LIMIT_S = 60.0
LIFT_DZ = 0.03  # an object counts as lifted 3 cm above its resting height
SAFETY_CONFIG = ROOT / "configs" / "safety.json"
COMMAND_SOURCES = ("builtin", "generated", "human")


def episode_seed(seed: int, index: int) -> int:
    return seed * 100_000 + index


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    scene: st.Scene
    command: st.Command
    layout_rng_seed: str
    expert_target: str


def plan_episode(
    seed: int,
    language: str,
    class_share: float,
    bank: dict | None,
    bank_share: float,
    family: str = "blocks",
) -> EpisodePlan:
    """Scene, command and expert target for one episode; independent of the language."""
    scene = st.sample_scene(random.Random(seed), family=family)
    command_rng = random.Random(f"{seed}/command")
    command = st.make_command(
        scene, command_rng, language, class_share, bank=bank, bank_share=bank_share
    )
    if command is None:  # a fresh scene always has free objects and containers
        raise RuntimeError(f"no command for seed {seed}")
    expert_target = random.Random(f"{seed}/expert").choice(sorted(command.targets))
    return EpisodePlan(seed, scene, command, f"{seed}/layout", expert_target)


def command_criterion(command: st.Command) -> str:
    """Criterion a class command sorts by, or the attributes an object command names."""
    return "+".join(attr for attr, _value in command.object_reference)


def load_bank(source: str, generated: Path) -> tuple[dict, float]:
    """Template bank and the share of commands drawn from it, for builtin or generated."""
    if source == "builtin":
        return {}, 0.0
    if not generated.is_file():
        raise SystemExit(f"command source 'generated' needs {generated}")
    bank = st.load_template_bank(generated)
    if not bank:
        raise SystemExit(f"{generated} holds no templates")
    return bank, 1.0


def load_human_plans(
    path: Path, class_share: float, reviewed_only: bool = True
) -> list[EpisodePlan]:
    """Episodes of the held-out command file (tools/data/draft_test_commands.py, CSV).

    Each row names its episode seed, language, allowed objects and destination. The plan is
    rebuilt from the seed and must agree with the row, then the row's text replaces the
    command text.
    """
    if not path.is_file():
        raise SystemExit(f"command source 'human' needs an existing --human-commands file ({path})")
    plans = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if reviewed_only and row["Reviewed"].strip().lower() != "yes":
                continue
            text = row["Command"].strip()
            if not text:
                continue
            family = row.get("Family") or "blocks"
            plan = plan_episode(
                int(row["Scene seed"]), row["Language"], class_share, {}, 0.0, family
            )
            targets = tuple(t.strip() for t in row["Allowed objects"].split("|"))
            command = plan.command
            if (
                command.kind != row["Command kind"]
                or set(command.targets) != set(targets)
                or command.destination != row["Destination"]
            ):
                raise SystemExit(
                    f"{path}: row for seed {row['Scene seed']} does not match its scene "
                    f"(check --class-share, now {class_share})"
                )
            plans.append(replace(plan, command=replace(command, text=text)))
    if not plans:
        raise SystemExit(
            f"{path} has no usable rows" + (" marked Reviewed=yes" if reviewed_only else "")
        )
    return plans


def episode_plans(args) -> list[EpisodePlan]:
    if args.command_source == "human":
        return load_human_plans(args.human_commands, args.class_share, not args.allow_unreviewed)
    bank, bank_share = load_bank(args.command_source, args.templates)
    return [
        plan_episode(
            episode_seed(args.seed, i),
            args.language,
            args.class_share,
            bank,
            bank_share,
            args.family,
        )
        for i in range(args.episodes)
    ]


def locations(twin) -> dict[str, str]:
    xyz = twin.objects_xyz()
    return {name: ts.location_of(twin.level, p, twin.table_top) for name, p in xyz.items()}


class LiftTracker:
    """Order in which objects first rise LIFT_DZ above their resting height."""

    def __init__(self, rest: dict[str, list[float]]) -> None:
        self.rest_z = {name: p[2] for name, p in rest.items()}
        self.order: list[str] = []

    def update(self, xyz: dict[str, list[float]]) -> None:
        for name, p in xyz.items():
            if name not in self.order and p[2] > self.rest_z[name] + LIFT_DZ:
                self.order.append(name)


class StepClock:
    """Counts physics steps (simulated time) through a hook on the task's step function."""

    def __init__(self, task, dt: float) -> None:
        self.dt, self.steps, self.tick = dt, 0, None
        original = task._step_scene

        def stepped() -> None:
            original()
            self.steps += 1
            if self.tick is not None:
                self.tick()

        task._step_scene = stepped

    def seconds(self) -> float:
        return self.steps * self.dt


def run_expert(twin, plan: EpisodePlan, clock: StepClock) -> dict:
    tracker = LiftTracker(twin.objects_xyz())
    clock.tick = lambda: tracker.update(twin.objects_xyz()) if clock.steps % 10 == 0 else None
    start = clock.steps
    outcome = twin.pick_and_place(
        plan.expert_target, plan.command.destination, echo=lambda _m: None
    )
    clock.tick = None
    phase = None
    if not outcome.ok:
        failed = next((s for s in reversed(outcome.steps) if not s.ok), None)
        phase = em.failure_phase(failed.name if failed else None)
    return {
        "lifted": tracker.order,
        "episode_time_s": round((clock.steps - start) * clock.dt, 2),
        "safety_interventions": 0,
        "safety_stop": False,
        "executor_phase": phase,
        "summary": outcome.summary,
    }


def reason_key(reason: str) -> str:
    """Safety reason without its numbers, so equal causes count together."""
    return re.sub(r"-?\d+(\.\d+)?", "#", reason)


def run_actions(
    twin,
    plan: EpisodePlan,
    act: Callable[[object, dict], list[float] | None],
    fps: float,
    size,
    max_actions: int,
    safety_on: bool = True,
    with_images: bool = True,
    recover: bool = True,
) -> dict:
    """Drive the arm with one joint target per step from ``act(state, images)`` until it
    returns None, the target objects rest in the destination, the safety filter stops, or
    ``max_actions`` is reached. Every target passes the safety filter; with ``safety_on``
    False the raw target is sent and the filter only counts. With ``recover`` False the arm
    stays where the actions left it (a DAgger prefix) instead of opening and going home."""
    import numpy as np
    from envs.utils import ArmTag

    safety = ActionSafetyFilter(SafetyConfig.load(SAFETY_CONFIG, "simulation"))
    reasons: Counter[str] = Counter()
    tracker = LiftTracker(twin.objects_xyz())
    dest = twin.level.container(plan.command.destination)
    actions = 0
    error = None
    stopped = False
    for step in range(1, max_actions + 1):
        if with_images:
            state, images = twin.record_frame(size)
        else:
            state, images = np.asarray(twin.task.robot.get_left_arm_jointState(), np.float32), {}
        try:
            action = act(state, images)
        except RuntimeError as exc:
            error = str(exc)
            break
        if action is None:
            break
        actions = step
        decision = safety.filter(action, [float(v) for v in state], step / fps)
        reasons.update(reason_key(r) for r in decision.reasons)
        if safety_on:
            if decision.status is Status.STOPPED:
                stopped = True
                break
            command = state if decision.action is None else decision.action  # rejected: hold
        else:
            command = action
        twin.drive_left_arm(command)
        xyz = twin.objects_xyz()
        tracker.update(xyz)
        if step % 15 == 0 and any(
            ts.is_inside(dest, xyz[t], twin.table_top) and xyz[t][2] <= tracker.rest_z[t] + 0.02
            for t in plan.command.targets
        ):
            break
    if recover:
        twin._recover(ArmTag("left"))
        twin._settle(60)
    return {
        "lifted": tracker.order,
        "episode_time_s": round(actions / fps, 2),
        "safety_interventions": safety.interventions,
        "safety_stop": stopped,
        "safety_reasons": dict(reasons.most_common()),
        "executor_phase": None,
        "summary": error or f"{actions} actions",
    }


def run_policy(
    twin, client, plan: EpisodePlan, fps: float, size, max_actions: int, safety_on: bool = True
) -> dict:
    client.reset()
    return run_actions(
        twin,
        plan,
        lambda state, images: client.act(state, images, plan.command.text),
        fps,
        size,
        max_actions,
        safety_on,
    )


def dagger_prefix_actions(plan: EpisodePlan, low: int, high: int) -> int:
    """Seeded number of policy actions before the expert takes over (same for every language)."""
    return random.Random(f"{plan.seed}/dagger").randint(low, high)


def run_dagger_prefix(twin, client, plan: EpisodePlan, fps: float, size, actions: int) -> dict:
    """Let the policy drive for ``actions`` steps through the safety filter, then open the
    gripper and leave the arm where it is, so the expert starts from the policy's state."""
    from envs.utils import ArmTag

    client.reset()
    result = run_actions(
        twin,
        plan,
        lambda state, images: client.act(state, images, plan.command.text),
        fps,
        size,
        actions,
        True,
        recover=False,
    )
    task = twin.task
    task.plan_success = True
    task.move(task.open_gripper(ArmTag("left")))
    twin._settle(40)
    return result


def record_expert_targets(twin, plan: EpisodePlan, clock: StepClock, fps: float):
    """Run the expert and sample its joint state at ``fps``; the targets are the next samples,
    exactly as twin_record.py builds training actions."""
    import numpy as np
    from twin_record import FrameRecorder

    recorder = FrameRecorder(
        lambda: (np.asarray(twin.task.robot.get_left_arm_jointState(), np.float32), {}),
        twin.step_dt(),
        fps,
    )
    recorder.attach(twin.task)
    recorder.start()
    result = run_expert(twin, plan, clock)
    recorder.stop()
    _states, targets, _times = recorder.arrays()
    return result, targets


def scene_meta(scene: st.Scene) -> dict:
    """The scene description twin_record.py stores in meta.json."""
    return {
        "objects": [
            {
                "name": o.name,
                "shape": o.shape,
                "colour": o.colour,
                "size": o.size,
                "item": o.item,
                "category": o.category,
            }
            for o in scene.objects
        ],
        "containers": [
            {
                "name": c.name,
                "colour": c.colour,
                "outline": c.outline,
                "depth": c.depth,
                "position": c.position,
            }
            for c in scene.containers
        ],
    }


def record_meta(
    plan: EpisodePlan, result: dict, fps: float, spawns, size, prefix: dict | None = None
) -> dict:
    command = plan.command
    return {
        "task": command.text,
        "language": command.language,
        "command_kind": command.kind,
        "family": plan.scene.family,
        "object_reference": dict(command.object_reference),
        "container_reference": list(command.container_reference),
        "targets": list(command.targets),
        "object": plan.expert_target,
        "destination": command.destination,
        "scene": scene_meta(plan.scene),
        "success": result["executor_phase"] is None,
        "summary": result["summary"],
        "fps": fps,
        "seed": plan.seed,
        "episode_key": f"eval-{plan.seed}",
        "spawns": {k: list(v) for k, v in (spawns or {}).items()},
        "image_size": list(size),
        "source": "robotwin_twin_scripted_expert (twin_eval)",
        **({"dagger_prefix": prefix} if prefix else {}),
    }


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def episode_record(run: dict, index: int, plan: EpisodePlan, before, after, result) -> dict:
    command = plan.command
    timed_out = result["episode_time_s"] > TIME_LIMIT_S
    verdict = em.judge_episode(
        before, after, command.targets, command.destination, result["lifted"], result["safety_stop"]
    )
    if timed_out:
        verdict["success"] = False
    phase = None
    if not verdict["success"]:
        if result["safety_stop"]:
            phase = "safety stop"
        elif timed_out:
            phase = "timeout"
        else:
            in_container = any(
                after.get(o) not in ("table", "off table", before.get(o)) for o in command.targets
            )
            phase = result["executor_phase"] or em.policy_failure_phase(
                result["lifted"], command.targets, in_container
            )
    return {
        **run,
        "episode": index,
        "seed": plan.seed,
        "language": command.language,
        "command_kind": command.kind,
        "family": plan.scene.family,
        "criterion": command_criterion(command),
        "text": command.text,
        "targets": list(command.targets),
        "destination": command.destination,
        "container_reference": list(command.container_reference),
        "n_objects": len(plan.scene.objects),
        "n_containers": len(plan.scene.containers),
        "before": before,
        "after": after,
        **verdict,
        "timed_out": timed_out,
        "lifted": result["lifted"],
        "episode_time_s": result["episode_time_s"],
        "safety_interventions": result["safety_interventions"],
        "safety_reasons": result.get("safety_reasons", {}),
        "failure_phase": phase,
        "summary": result["summary"],
    }


def run(args) -> int:
    from twin_console import Twin, record_size
    from twin_record import seed_everything

    plans = episode_plans(args)
    run_meta = {
        "format": FORMAT_VERSION,
        "executor": args.executor,
        "checkpoint": {
            "policy": args.checkpoint,
            "dagger": f"scripted expert after {args.checkpoint}",
            "replay": "expert replay",
        }.get(args.executor, "scripted expert"),
        "safety": "off" if args.no_safety else "on",
        "command_source": args.command_source,
        "commit": git_commit(),
        "run_seed": args.seed,
        "class_share": args.class_share,
    }
    client = None
    if args.executor in ("policy", "dagger"):
        from policy_bridge import PolicyClient

        client = PolicyClient(args.policy_server)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.is_file():  # resume: keep finished episodes of the same run
        for line in args.out.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["episode"])
    twin = Twin(args.robotwin_root.resolve(), args.seed, 0, "free", 1, headless=True)
    size = record_size()
    started = time.perf_counter()
    for index, plan in enumerate(plans):
        if index in done:
            continue
        twin.custom_level = st.to_level(plan.scene)
        twin.setup()
        clock = StepClock(twin.task, twin.step_dt())
        spawns = twin.reset_layout(random.Random(plan.layout_rng_seed))
        before = locations(twin)
        seed_everything(f"eval/{plan.seed}")
        if args.executor == "expert" and args.record_dir:
            from twin_record import FrameRecorder, cv2_saver, next_episode_index, write_episode

            recorder = FrameRecorder(lambda: twin.record_frame(size), twin.step_dt(), args.fps)
            recorder.attach(twin.task)
            recorder.start()
            result = run_expert(twin, plan, clock)
            recorder.stop()
            states, targets, times = recorder.arrays()
            args.record_dir.mkdir(parents=True, exist_ok=True)
            write_episode(
                args.record_dir,
                next_episode_index(args.record_dir),
                states,
                targets,
                times,
                recorder.images,
                record_meta(plan, result, args.fps, spawns, size),
                cv2_saver(),
            )
        elif args.executor == "dagger":
            from twin_record import FrameRecorder, cv2_saver, next_episode_index, write_episode

            steps = dagger_prefix_actions(plan, args.dagger_min_actions, args.dagger_max_actions)
            prefix = run_dagger_prefix(twin, client, plan, args.fps, size, steps)
            recorder = FrameRecorder(lambda: twin.record_frame(size), twin.step_dt(), args.fps)
            recorder.attach(twin.task)
            recorder.start()
            result = run_expert(twin, plan, clock)
            recorder.stop()
            states, targets, times = recorder.arrays()
            info = {
                "policy": args.checkpoint,
                "actions": steps,
                "lifted": prefix["lifted"],
                "safety_stop": prefix["safety_stop"],
            }
            args.record_dir.mkdir(parents=True, exist_ok=True)
            write_episode(
                args.record_dir,
                next_episode_index(args.record_dir),
                states,
                targets,
                times,
                recorder.images,
                record_meta(plan, result, args.fps, spawns, size, info),
                cv2_saver(),
            )
            result["summary"] = f"policy {steps} actions, then expert: {result['summary']}"
        elif args.executor == "expert":
            result = run_expert(twin, plan, clock)
        elif args.executor == "policy":
            result = run_policy(
                twin, client, plan, args.fps, size, int(TIME_LIMIT_S * args.fps), not args.no_safety
            )
        else:  # replay: expert once, then its targets on the same scene through the policy path
            expert, targets = record_expert_targets(twin, plan, clock, args.fps)
            twin.setup()
            clock = StepClock(twin.task, twin.step_dt())
            twin.reset_layout(random.Random(plan.layout_rng_seed))
            before = locations(twin)
            seed_everything(f"eval/{plan.seed}")
            queue = iter(targets.tolist())
            result = run_actions(
                twin,
                plan,
                lambda _state, _images, q=queue: next(q, None),
                args.fps,
                size,
                len(targets),
                not args.no_safety,
                with_images=False,
            )
            expert_ok = expert["executor_phase"] is None
            result["summary"] = (
                f"expert {'ok' if expert_ok else 'failed'} ({len(targets)} targets); replay "
                + result["summary"]
            )
        record = episode_record(run_meta, index, plan, before, locations(twin), result)
        with args.out.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{index + 1}/{len(plans)}] {'ok  ' if record['success'] else 'FAIL'} "
            f"{record['text']} ({record['failure_phase'] or record['summary']})",
            flush=True,
        )
    print(f"EVAL_DONE out={args.out} elapsed_s={time.perf_counter() - started:.0f}")
    return 0


def read_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def summarise_cmd(args, write: Callable[[str], None] = print) -> int:
    by = [f.strip() for f in args.by.split(",") if f.strip()]
    rows = em.summarise(read_records(args.results), by)
    write(em.format_table(rows, by))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run seeded episodes in the simulator")
    r.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path.home() / "worktrees/Panthera-HT_RoboTwin/RoboTwin",
    )
    r.add_argument(
        "--executor", choices=("expert", "policy", "replay", "dagger"), required=True
    )
    r.add_argument("--policy-server", default="127.0.0.1:5555")
    r.add_argument("--checkpoint", default="", help="label of the served checkpoint (logged)")
    r.add_argument("--episodes", type=int, default=30)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument(
        "--language",
        choices=("en", "vi"),
        default="en",
        help="command language (the human source takes it from each row)",
    )
    r.add_argument("--class-share", type=float, default=0.3)
    r.add_argument("--family", choices=st.FAMILIES, default="blocks", help="object family")
    r.add_argument("--command-source", choices=COMMAND_SOURCES, default="builtin")
    r.add_argument("--templates", type=Path, default=ROOT / "configs" / "command_templates.json")
    r.add_argument(
        "--human-commands",
        type=Path,
        default=ROOT / "configs" / "test_commands_v1.csv",
        help="held-out commands for --command-source human (episodes come from this file)",
    )
    r.add_argument(
        "--allow-unreviewed", action="store_true", help="also use rows not marked Reviewed=yes"
    )
    r.add_argument("--fps", type=float, default=15.0, help="policy action rate")
    r.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="expert executor: also save every episode in the raw training format (twin_record.py)",
    )
    r.add_argument(
        "--dagger-min-actions", type=int, default=15, help="dagger: shortest policy prefix"
    )
    r.add_argument(
        "--dagger-max-actions", type=int, default=150, help="dagger: longest policy prefix"
    )
    r.add_argument(
        "--no-safety",
        action="store_true",
        help="diagnostic: send targets without the safety filter (decisions are still counted)",
    )
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarise", help="tables from result JSONL files (no simulator needed)")
    s.add_argument("results", type=Path, nargs="+")
    s.add_argument("--by", default="executor,language,command_kind")
    s.add_argument("--json", type=Path, default=None, help="also write the rows as JSON")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        if args.record_dir and args.executor not in ("expert", "dagger"):
            parser.error("--record-dir needs --executor expert or dagger")
        if args.executor in ("policy", "dagger") and not args.checkpoint:
            parser.error(f"--checkpoint is required with --executor {args.executor}")
        if args.executor == "dagger" and not args.record_dir:
            parser.error("--executor dagger needs --record-dir")
        if not 1 <= args.dagger_min_actions <= args.dagger_max_actions:
            parser.error("need 1 <= --dagger-min-actions <= --dagger-max-actions")
        return run(args)
    return summarise_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
