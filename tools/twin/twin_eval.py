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
  # anywhere
  python tools/twin/twin_eval.py summarise results/*.jsonl --by executor,language,command_kind
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_metrics as em
import sorting_task as st
import twin_scene as ts

ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = 1
TIME_LIMIT_S = 60.0
LIFT_DZ = 0.03  # an object counts as lifted 3 cm above its resting height
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


class SafetyFilter:
    """Simulation stand-in for the safety layer: joint-limit and step-size checks on actions.

    An arm joint target outside its limits is clamped and counted as an intervention; a jump
    larger than ``max_step`` rad from the current state stops the episode.
    """

    def __init__(self, limits: list[tuple[float, float]], max_step: float = 0.35) -> None:
        self.limits = limits
        self.max_step = max_step
        self.interventions = 0
        self.stopped = False

    def check(self, action: list[float], state: list[float]) -> list[float]:
        out = list(action)
        changed = False
        for i, (lo, hi) in enumerate(self.limits):
            if i >= len(out):
                break
            clamped = min(max(out[i], lo), hi)
            if clamped != out[i]:
                out[i], changed = clamped, True
            if i < len(state) and abs(out[i] - state[i]) > self.max_step:
                self.stopped = True
        self.interventions += changed
        return out


def arm_limits(twin) -> list[tuple[float, float]]:
    """Joint limits of the left arm joints, in the order of the joint state."""
    entity = twin.task.robot.left_entity
    arm = {j.get_name() for j in twin.task.robot.left_arm_joints}
    limits = []
    for joint in entity.get_active_joints():
        if joint.get_name() in arm:
            lo, hi = (float(v) for v in joint.get_limits()[0])
            limits.append((lo, hi))
    return limits


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


def run_policy(twin, client, plan: EpisodePlan, fps: float, size, max_actions: int) -> dict:
    import numpy as np
    from envs.utils import ArmTag

    safety = SafetyFilter(arm_limits(twin))
    tracker = LiftTracker(twin.objects_xyz())
    dest = twin.level.container(plan.command.destination)
    client.reset()
    actions = 0
    error = None
    for actions in range(1, max_actions + 1):
        state, images = twin.record_frame(size)
        try:
            action = client.act(state, images, plan.command.text)
        except RuntimeError as exc:
            error = str(exc)
            break
        action = safety.check(action, [float(v) for v in state])
        if safety.stopped:
            break
        twin.task.take_action(np.asarray(action, dtype=np.float64), action_type="qpos")
        xyz = twin.objects_xyz()
        tracker.update(xyz)
        if actions % 15 == 0 and any(
            ts.is_inside(dest, xyz[t], twin.table_top) and xyz[t][2] <= tracker.rest_z[t] + 0.02
            for t in plan.command.targets
        ):
            break
    twin._recover(ArmTag("left"))
    twin._settle(60)
    return {
        "lifted": tracker.order,
        "episode_time_s": round(actions / fps, 2),
        "safety_interventions": safety.interventions,
        "safety_stop": safety.stopped,
        "executor_phase": None,
        "summary": error or f"{actions} policy actions",
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
        "checkpoint": args.checkpoint if args.executor == "policy" else "scripted expert",
        "command_source": args.command_source,
        "commit": git_commit(),
        "run_seed": args.seed,
        "class_share": args.class_share,
    }
    client = None
    if args.executor == "policy":
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
        twin.reset_layout(random.Random(plan.layout_rng_seed))
        before = locations(twin)
        seed_everything(f"eval/{plan.seed}")
        if args.executor == "expert":
            result = run_expert(twin, plan, clock)
        else:
            result = run_policy(twin, client, plan, args.fps, size, int(TIME_LIMIT_S * args.fps))
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
    r.add_argument("--executor", choices=("expert", "policy"), required=True)
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
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarise", help="tables from result JSONL files (no simulator needed)")
    s.add_argument("results", type=Path, nargs="+")
    s.add_argument("--by", default="executor,language,command_kind")
    s.add_argument("--json", type=Path, default=None, help="also write the rows as JSON")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        if args.executor == "policy" and not args.checkpoint:
            parser.error("--checkpoint is required with --executor policy")
        return run(args)
    return summarise_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
