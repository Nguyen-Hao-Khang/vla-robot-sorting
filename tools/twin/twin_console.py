#!/usr/bin/env python3
# ruff: noqa: E501  # long operator-facing status lines
"""Interactive digital twin: type a sorting command, a model plans it, Panthera-HT executes it.

One persistent RoboTwin scene stays open in the SAPIEN viewer, with a second window that
streams the wrist (gripper) camera and the workspace cameras. Seven scene levels:
  1 one cube, one tray            2 three coloured cubes, three coloured trays
  3 cube / cylinder / triangular prism, wide shallow tray + narrow deep bin
  4 three cylinders of different sizes   5 cylinders in two colours x two sizes
  6 look-alike packages (fresh milk vs soy milk carton, box 1 vs box 2)
  7 look-alike balls and bottles (tennis ball vs volleyball, water vs juice bottle)
Commands are typed in the terminal in any language (e.g. Vietnamese or English). The model
router (twin_models.py: Flash Lite, Robotics-ER only for refusals/invalid answers) turns each command into an
ordered plan of up to MAX_STEPS pick-and-place steps using ONLY the names in the scene; a plan
that fails validation is passed to the next model. The robot executes the steps in order with
RoboTwin's planners: hover above the object at a safe height, grasp, straight lift, short
horizontal hops, alignment in the air, straight descent, low release, straight retreat. After a failed step or a change the plan did not cause
(simulator check + workspace camera), the workspace is rescanned and the planner is asked for
the remaining work (at most MAX_REPLANS rounds). Recovery: an object whose motion failed
(knocked over, pushed out, missed) is planned again only through a fresh rescan and a new
validated plan from its CURRENT pose, at most MAX_OBJECT_ATTEMPTS times; an object stuck in
the gripper is never re-attempted; a placed object that is later knocked out of its
destination is put back. `perceive` runs the VLM detector on the workspace camera and scores it against the
simulator truth; `source vlm` plans from the camera detections instead of the simulator list.

Scope: demo / visual-inspection mode. Execution uses upstream RoboTwin grasp/place
primitives; it is a demonstration and data-collection tool, not an evaluated controller. Failures are reported with per-step
diagnostics; a failed motion is never repeated blindly (see Recovery above).
Run with the RoboTwin conda python in a graphical session:
  python tools/twin/twin_console.py --level 3
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import inspect
import json
import math
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import twin_grasp as tg  # noqa: E402
import twin_kinematics as tk  # noqa: E402
import twin_obstacles as tob  # noqa: E402
import twin_scene as ts  # noqa: E402
from twin_perception import (  # noqa: E402
    change_map,
    change_threshold,
    changed_fraction,
    dilate,
    workspace_roi,
)

TASK = "blocks_ranking_rgb"
TASK_CONFIG = "blocks_ranking_rgb_panthera_single"
UPSTREAM_BLOCKS = ("block1", "block2", "block3")
TABLE_Z = 0.74
MIN_LIFT_M = 0.03
MAX_STEPS = 6
TWIN_VERSION = "v15"  # written into every session log (setup event)
PLACE_CONSTRAINS = ("free", "align", "auto")
CACHE_DIR = Path.home() / ".cache" / "vla_robot_sorting"

SYSTEM = (
    "You plan robot pick-and-place work on a table for one robot arm. The operator command may "
    "be in any language (often Vietnamese or English). Return an ordered list of steps; each "
    "step moves ONE object (copy an exact object name) into ONE destination (copy an exact "
    f"destination name). Use at most {MAX_STEPS} steps and only as many as the command needs; "
    "skip objects that are already where they should be. Names in parentheses after 'also "
    "called' are synonyms the operator may use, but always answer with the exact name. Positions "
    "are in metres: x<0 is the operator's left, x>0 the right, smaller y is closer to the robot. "
    "Sizes, shapes, colours, current locations, free room in each destination and objects that "
    "cannot be picked are given in the scene list; respect them. Objects marked 'lying on its "
    "side' CAN be picked; only objects marked NOT GRASPABLE NOW or CANNOT cannot. To swap A (in "
    "X) and B (in Y): A -> table, B -> X, A -> Y; the 'table' destination means a free spot in "
    "the object row (use a staging pad instead only if the scene lists one). Never put more "
    "objects into a destination than its free room, "
    "counting objects that earlier steps move in or out. Objects already inside a destination "
    "(even fallen ones) only use up its room; a destination with free room can still take "
    "objects. Generic words name a category, not one object: 'chai nước' / 'bottle' = any "
    "bottle, 'hộp sữa' = any milk carton, 'hộp carton' / 'thùng' = any cardboard box, 'quả "
    "bóng' = any ball; a count ('2 chai nước', 'two balls') means that many objects of the "
    "category. 'bên trái' / left and 'bên phải' / right refer to x (left = smaller x). Tolerate "
    "typos and missing Vietnamese accents (e.g. 'khai' = 'khay'). If only part of the command "
    "is possible (an object is NOT GRASPABLE NOW, unreachable or missing), set decision to 'execute' for "
    "the possible part and say in 'reason' what was skipped and why. Set decision to 'refuse' "
    "only if nothing in the command can be done or it is negated or unsafe; set 'clarify' if it "
    "is ambiguous (several objects match and the command does not say which, and the count "
    "does not settle it). Always write 'reason' in short English."
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["execute", "refuse", "clarify"]},
        "language": {"type": "string", "enum": ["vi", "en", "mixed", "other"]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "object": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["object", "destination"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["decision", "language", "steps", "reason"],
}
CONTROL_HELP = (
    "Control words: scene, level <1-7>, reset (new random layout), models (router status), "
    "perceive (workspace camera only), scan (workspace + wrist sweep, fused), "
    "source sim|vlm, help, quit."
)


def help_text(level: ts.Level) -> str:
    levels = "\n".join(f"  level {n}: {lv.title}" for n, lv in ts.LEVELS.items())
    examples = "\n".join(f"  {e}" for e in level.examples)
    return (
        f"Levels:\n{levels}\nExamples for level {level.number}:\n{examples}\n"
        "Multi-step commands are allowed (e.g. '... then swap A and B').\n" + CONTROL_HELP
    )


@dataclass(frozen=True)
class NameTable:
    """Accepted spellings (casefolded) -> canonical object / destination names."""

    objects: dict[str, str]
    places: dict[str, str]

    @classmethod
    def from_level(cls, level: ts.Level, object_names: list[str] | None = None) -> NameTable:
        objects: dict[str, str] = {}
        if object_names is None:
            for obj in level.objects:
                objects[obj.name.casefold()] = obj.name
        else:
            objects = {n.casefold(): n for n in object_names}
        places: dict[str, str] = {}
        for c in level.containers:
            places[c.name.casefold()] = c.name
        for c in level.containers:
            for alias in c.aliases:
                places.setdefault(alias.casefold(), c.name)
        return cls(objects, places)

    @property
    def object_names(self) -> list[str]:
        return list(dict.fromkeys(self.objects.values()))

    @property
    def destination_names(self) -> list[str]:
        return list(dict.fromkeys(self.places.values()))


def object_poses(
    objects_xyz: dict[str, list[float]],
    tilts: dict[str, float] | None = None,
    orientations: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Orientation quaternion per object; without full orientations a tilt about x is assumed."""
    if orientations:
        return {n: list(orientations[n]) for n in objects_xyz if n in orientations}
    out = {}
    for name in objects_xyz:
        half = math.radians((tilts or {}).get(name, 0.0)) / 2
        out[name] = [math.cos(half), math.sin(half), 0.0, 0.0]
    return out


def grasp_choice(
    level: ts.Level,
    obj_name: str,
    objects_xyz: dict[str, list[float]],
    poses: dict[str, list[float]],
    table_top: float,
    finger_axis: str | None = None,
    dest: ts.ContainerSpec | None = None,
    allow_tilt: bool = False,
    ref_quat=None,
) -> tg.GraspChoice:
    """Best top-down grasp of the object in its current pose, considering its neighbours."""
    obj = level.object(obj_name)
    where = ts.location_of(level, objects_xyz[obj_name], table_top)
    source = None if where in ("table", "off table") else level.container(where)
    frac = ts.grasp_fraction(obj, source)
    if dest is not None:
        frac = max(frac, ts.release_grasp_fraction(obj, dest))
    neighbours = [
        (level.object(n), xyz, poses.get(n, [1.0, 0.0, 0.0, 0.0]))
        for n, xyz in objects_xyz.items()
        if n != obj_name
    ]
    return tg.plan_grasp(
        obj,
        objects_xyz[obj_name],
        poses.get(obj_name, [1.0, 0.0, 0.0, 0.0]),
        neighbours,
        finger_axis,
        source,
        frac,
        allow_tilt,
        ref_quat,
    )


def scene_listing(
    level: ts.Level,
    objects_xyz: dict[str, list[float]],
    table_top: float,
    tilts: dict[str, float] | None = None,
    orientations: dict[str, list[float]] | None = None,
    finger_axis: str | None = None,
    allow_tilt: bool = False,
) -> list[str]:
    poses = object_poses(objects_xyz, tilts, orientations)
    rows = ["Objects:"]
    for obj in level.objects:
        x, y, z = objects_xyz[obj.name]
        where = ts.location_of(level, (x, y, z), table_top)
        place = {"table": "on the table", "off table": "OFF THE TABLE (unreachable)"}.get(
            where, f"in the {where}"
        )
        note = ""
        choice = grasp_choice(
            level, obj.name, objects_xyz, poses, table_top, finger_axis, None, allow_tilt
        )
        vh = choice.vertical_half or None
        if where not in ("table", "off table") and not ts.graspable_in(
            obj, level.container(where), vh
        ):
            note = "; CANNOT be picked back out (top is not above the walls)"
        elif where != "off table":
            if not choice.ok:
                note = f"; NOT GRASPABLE NOW ({choice.reason})"
            elif choice.lying:
                note = "; lying on its side (can still be grasped)"
        rows.append(f"- {obj.name}: {obj.describe()}; at x={x:+.2f}, y={y:+.2f}, {place}{note}")
    rows += destination_rows(level, objects_xyz, table_top)
    return rows


def destination_rows(
    level: ts.Level, objects_xyz: dict[str, list[float]], table_top: float
) -> list[str]:
    rows = ["Destinations:"]
    for c in level.containers:
        held = ts.occupants(level, c, objects_xyz, table_top)
        free = max(0, c.capacity - len(held))
        holds = f"holds {', '.join(held)}" if held else "empty"
        rows.append(
            f"- {c.name}: {c.describe()}; at x={c.xy[0]:+.2f}, y={c.xy[1]:+.2f}; {holds}; free room {free}"
        )
    return rows


def build_prompt(command: str, names: NameTable, listing: list[str]) -> str:
    return (
        f"Exact object names: {', '.join(names.object_names)}\n"
        f"Exact destination names: {', '.join(names.destination_names)}\n"
        "Scene:\n" + "\n".join(listing) + f"\n\nOperator command:\n{command}"
    )


def resolve_plan(payload: dict, names: NameTable | ts.Level) -> list[tuple[str, str]] | str:
    """Return [(object, destination), ...] for an executable plan, else a refusal message."""
    if isinstance(names, ts.Level):
        names = NameTable.from_level(names)
    decision = payload.get("decision")
    reason = str(payload.get("reason") or "").strip()
    if decision != "execute":
        return reason or "The model returned no executable plan."
    steps = payload.get("steps") or []
    if not steps:
        return reason or "The model returned an empty plan."
    if len(steps) > MAX_STEPS:
        return f"Plan has {len(steps)} steps; the limit is {MAX_STEPS}."
    plan = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return f"Step {index} is malformed."
        obj = names.objects.get(str(step.get("object") or "").strip().casefold())
        dest = names.places.get(str(step.get("destination") or "").strip().casefold())
        if obj is None:
            return f"Step {index}: object '{step.get('object')}' is not in the scene."
        if dest is None:
            return f"Step {index}: destination '{step.get('destination')}' is not in the scene."
        plan.append((obj, dest))
    return plan


def simulate_plan(
    level: ts.Level,
    objects_xyz: dict[str, list[float]],
    table_top: float,
    plan: list[tuple[str, str]],
) -> str | None:
    """Check a plan against capacities and reachability, step by step, before any motion."""
    where = {n: ts.location_of(level, xyz, table_top) for n, xyz in objects_xyz.items()}
    for index, (obj, dest) in enumerate(plan, start=1):
        here = where.get(obj)
        if here == dest:
            continue
        if here == "off table":
            return f"step {index}: {obj} is off the table and cannot be reached"
        if here not in (None, "table") and not ts.graspable_in(
            level.object(obj), level.container(here)
        ):
            return f"step {index}: {obj} cannot be picked out of the {here}"
        container = level.container(dest)
        inside = [n for n, w in where.items() if w == dest]
        if len(inside) >= container.capacity:
            return f"step {index}: the {dest} would be full (holds {', '.join(inside)}); use another destination"
        where[obj] = dest
    return None


def plan_problem(payload: dict, names: NameTable, scene: tuple | None = None) -> str | None:
    """Router validator: None when the payload is a usable answer (plan, refusal or question).

    ``scene`` = (level, objects_xyz, table_top) adds a step-by-step capacity/reachability check,
    so a plan that would overfill a container is passed to the next model instead of executed.
    """
    decision = payload.get("decision")
    if decision in ("refuse", "clarify"):
        return None
    if decision != "execute":
        return f"unknown decision {decision!r}"
    resolved = resolve_plan(payload, names)
    if isinstance(resolved, str):
        return resolved
    if scene is not None:
        level, objects_xyz, table_top = scene
        if all(o in objects_xyz for o, _ in resolved):
            return simulate_plan(level, objects_xyz, table_top, resolved)
    return None


def lifted_ok(z_before: float, z_after: float) -> bool:
    return z_after - z_before >= MIN_LIFT_M


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class StepReport:
    name: str
    ok: bool
    detail: str = ""
    object_xyz_after: list[float] = field(default_factory=list)


@dataclass
class Outcome:
    ok: bool
    summary: str
    steps: list[StepReport]
    skipped: bool = False
    moved: bool = True


class Planner:
    def __init__(self, router) -> None:
        self.router = router

    def plan(
        self, prompt: str, names: NameTable, scene: tuple | None = None, escalate: bool = True
    ):
        # Replanning rounds do not escalate: "nothing more to do" is a normal answer there.
        return self.router.call(
            SYSTEM,
            prompt,
            SCHEMA,
            validate=lambda p: plan_problem(p, names, scene),
            escalate_when=(lambda p: p.get("decision") in ("refuse", "clarify"))
            if escalate
            else None,
        )


class StepError(RuntimeError):
    """A step that must stop the plan (reported with its own message)."""


NO_GRASP = "no reachable grasp"
BAD_GRASP_RADIUS = 0.015  # a frame that failed is skipped while the object stays this close
INTERIOR_HOME = (3.2 / 64, 4.0 / 64)  # joint 2/3 home (rad), as in native_initial_home
TALL_OBSTACLE = 0.08  # objects this tall (cartons, bottles) stay when the world is relaxed


def _rotz(deg: float):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def lift_options(full: float, minimum: float) -> list[float]:
    """Straight-lift heights to try, highest first (full transport height down to the
    smallest lift that clears what is right around the object)."""
    minimum = min(minimum, full)
    out = [full]
    for frac in (0.75, 0.5):
        dz = minimum + frac * (full - minimum)
        if dz < out[-1] - 0.01:
            out.append(dz)
    if minimum < out[-1] - 0.005:
        out.append(minimum)
    return out


def best_lift(tool_pose, options) -> float | None:
    """Highest lift in ``options`` that the arm reaches holding this tool orientation."""
    r = tk.quat_matrix(tool_pose[3:7])
    for dz in options:
        if tk.ik([tool_pose[0], tool_pose[1], tool_pose[2] + dz], r) is not None:
            return dz
    return None


def minimal_lift(
    level: ts.Level,
    objects_xyz: dict,
    obj_name: str,
    table_top: float,
    holder: set[str],
    goal_xy=None,
) -> float:
    """Smallest straight lift: the object's bottom clears the holding container's rim and,
    by 2 cm, the top of every object within 12 cm that lies towards ``goal_xy`` (the way
    the transit leaves; the transit then climbs to the transport height)."""
    obj = level.object(obj_name)
    xy = objects_xyz[obj_name][:2]

    def ahead(n: str) -> bool:
        if goal_xy is None:
            return True
        d = (objects_xyz[n][0] - xy[0], objects_xyz[n][1] - xy[1])
        g = (goal_xy[0] - xy[0], goal_xy[1] - xy[1])
        return d[0] * g[0] + d[1] * g[1] > 0

    tops = [
        objects_xyz[n][2] + level.object(n).half_height
        for n in objects_xyz
        if n != obj_name and math.dist(objects_xyz[n][:2], xy) < 0.12 and ahead(n)
    ]
    tops += [table_top + level.container(h).wall_height for h in holder]
    bottom = objects_xyz[obj_name][2] - obj.half_height
    return max(ts.MIN_LIFT_DZ, max(tops, default=0.0) + 0.02 - bottom)


def hover_height(pre: list[float], top: float) -> float | None:
    """Highest height (top, then 5 / 10 cm lower, then just above the pre-grasp pose) at
    which the arm can hold the grasp orientation above the pre-grasp point."""
    r = tk.quat_matrix(pre[3:7])
    start = max(top, pre[2])
    for z in (start, start - 0.05, start - 0.10, pre[2] + 0.03):
        if z >= pre[2] - 1e-9 and tk.ik([pre[0], pre[1], z], r) is not None:
            return z
    return None


def plan_pick(
    pre: list[float],
    target: list[float],
    centre,
    slot,
    release_centre_z: float,
    full_lift: float,
    min_lift: float,
    safe_z: float,
) -> dict | None:
    """Kinematic plan of a whole pick for one grasp frame, or None if the arm cannot reach
    some part of it (Panthera model, joint limits). Poses: wrist setting above the object,
    pre-grasp, grasp, straight lift (the highest reachable between the local minimum and
    the transport height), above the slot at transport height, release. Above the slot the
    gripper is turned about the vertical so a tilted approach leans away from the robot
    there too (the object stays upright)."""
    quat = list(target[3:7])
    hover_z = hover_height(pre, safe_z)
    if hover_z is None:
        return None
    if not tk.reachable_chain([[pre[0], pre[1], hover_z, *quat], pre, target]):
        return None
    lift = best_lift(target, lift_options(full_lift, min_lift))
    if lift is None:
        return None
    turn, quat2 = tg.yaw_to_lean(quat, (slot[0] - BASE_XY[0], slot[1] - BASE_XY[1]))
    if abs(turn) < SLOT_TURN_MIN_DEG:
        turn, quat2 = 0.0, quat
    off = _rotz(turn) @ np.array([centre[k] - target[k] for k in range(3)])
    ox, oy = slot[0] - off[0], slot[1] - off[1]
    over = [ox, oy, target[2] + full_lift, *quat2]
    release = [ox, oy, release_centre_z - off[2], *quat2]
    if not tk.reachable_chain([over, release]):
        return None
    return {"hover_z": hover_z, "lift": lift, "turn": turn, "over": over, "release": release}


def grasp_key(frame) -> str:
    return json.dumps([[round(float(v), 3) + 0.0 for v in row] for row in frame])  # no -0.0


STUCK_TEXT = "stuck in the gripper"  # the one failure that is never re-attempted
DRAG_TOL = 0.015
GRASP_SEARCH_LIMIT = 72  # candidate frames considered (ranked by the kinematic model)
ROBOTWIN_FALLBACK_LIMIT = 8  # frames handed to RoboTwin's own grasp when all else fails
STRAIGHT_LABELS = (
    "lift",
    "descend",
    "final approach",
    "retreat",
    "put back",
    "lower to the pre-grasp pose",
)  # short straight moves
PRE_GRASP_DIS = 0.09  # straight final approach along the gripper axis (m)
VERTICAL_TOL_DEG = 2.0  # a frame this close to straight down is executed by the twin itself
BASE_XY = (tk.BASE_XYZ[0], tk.BASE_XYZ[1])
SLOT_TURN_MIN_DEG = 5.0  # re-aim a tilted grasp about the vertical when it is this far off
GRASP_CLEAR_RADIUS = 0.11  # neighbours this close are left out of CuRobo during the grasp


class NoMotionError(StepError):
    """A step refused before anything moved (e.g. no reachable grasp)."""


def run_steps(
    steps: list[tuple[str, Callable[[], object]]],
    execute: Callable[[object], None],
    plan_ok: Callable[[], bool],
    object_xyz: Callable[[], list[float]],
    echo: Callable[[str], None],
    notes: dict[str, str] | None = None,
    guard: Callable[[str], str | None] | None = None,
) -> list[StepReport]:
    """Execute named motion steps in order; stop at the first failure, missed grasp or collision.

    A step builder may return None (nothing to do, e.g. no alignment needed); ``notes`` lets a
    builder attach a detail string to its own report; ``guard(step)`` returns a message when an
    object that should stay still has moved (collision), which stops the sequence.
    """
    notes = {} if notes is None else notes
    reports: list[StepReport] = []
    z_start = object_xyz()[2]
    for index, (name, build) in enumerate(steps, start=1):
        try:
            action = build()
            if action is not None:
                execute(action)
        except StepError as exc:
            reports.append(StepReport(name, False, str(exc), object_xyz()))
            echo(f"    [{index}/{len(steps)}] {name}: FAILED ({exc})")
            return reports
        except Exception as exc:  # RoboTwin planner boundary (e.g. UnStableError)
            reports.append(StepReport(name, False, f"{type(exc).__name__}: {exc}", object_xyz()))
            echo(f"    [{index}/{len(steps)}] {name}: FAILED ({type(exc).__name__})")
            return reports
        xyz = object_xyz()
        if not plan_ok():
            reports.append(StepReport(name, False, "planner found no safe trajectory", xyz))
            echo(f"    [{index}/{len(steps)}] {name}: FAILED (no safe trajectory)")
            return reports
        if name == "lift" and not lifted_ok(z_start, xyz[2]):
            detail = f"grasp missed: object rose {xyz[2] - z_start:+.3f} m (< {MIN_LIFT_M} m)"
            reports.append(StepReport(name, False, detail, xyz))
            echo(f"    [{index}/{len(steps)}] {name}: FAILED ({detail})")
            return reports
        problem = guard(name) if guard else None
        if problem:
            reports.append(StepReport(name, False, problem, xyz))
            echo(f"    [{index}/{len(steps)}] {name}: STOPPED ({problem})")
            return reports
        detail = notes.get(name, "")
        reports.append(StepReport(name, True, detail, xyz))
        extra = f"; {detail}" if detail else ""
        echo(
            f"    [{index}/{len(steps)}] {name}: ok (object at x={xyz[0]:+.3f} y={xyz[1]:+.3f} z={xyz[2]:.3f}{extra})"
        )
    return reports


def run_plan(
    plan: list[tuple[str, str]],
    do_step: Callable[[str, str, Callable[[str], None]], Outcome],
    echo: Callable[[str], None],
) -> list[Outcome]:
    """Run plan steps in order (a failed step is not repeated here; recovery is a new plan).

    A motion failure stops the plan. A step refused by the pre-checks before any motion
    (moved=False: fallen over, too close to a neighbour, destination full) changes nothing in
    the scene, so the remaining steps still run; the replanner deals with the blocked one.
    """
    outcomes: list[Outcome] = []
    for index, (obj, dest) in enumerate(plan, start=1):
        echo(f"  step {index}/{len(plan)}: {obj} -> {dest}")
        outcome = do_step(obj, dest, echo)
        outcomes.append(outcome)
        if outcome.ok:
            echo("  ok: " + outcome.summary)
        elif outcome.moved:
            echo("  FAILED: " + outcome.summary)
            break
        else:
            echo("  BLOCKED (no motion, continuing with the next step): " + outcome.summary)
    return outcomes


def precheck(
    level: ts.Level,
    obj_name: str,
    dest_name: str,
    objects_xyz: dict[str, list[float]],
    table_top: float,
    tilts: dict[str, float] | None = None,
    orientations: dict[str, list[float]] | None = None,
    finger_axis: str | None = None,
    allow_tilt: bool = False,
) -> Outcome | None:
    """Executor checks before any motion: already there, full, unreachable, no grasp."""
    obj, dest = level.object(obj_name), level.container(dest_name)
    xyz = objects_xyz[obj_name]
    where = ts.location_of(level, xyz, table_top)
    if where == "off table":
        return Outcome(
            False,
            f"{obj_name} is off the table (x={xyz[0]:+.2f}, y={xyz[1]:+.2f}, z={xyz[2]:.2f}); the robot did not move",
            [],
            moved=False,
        )
    if ts.in_destination(level, dest, xyz, table_top):
        return Outcome(
            True,
            f"{obj_name} is already in the {dest_name}; skipped",
            [],
            skipped=True,
            moved=False,
        )
    held = [n for n in ts.occupants(level, dest, objects_xyz, table_top) if n != obj_name]
    if len(held) >= dest.capacity:
        return Outcome(
            False,
            f"the {dest_name} is full (holds {', '.join(held)}); the robot did not move",
            [],
            moved=False,
        )
    poses = object_poses(objects_xyz, tilts, orientations)
    choice = grasp_choice(
        level, obj_name, objects_xyz, poses, table_top, finger_axis, dest, allow_tilt
    )
    if where != "table" and not ts.graspable_in(
        obj, level.container(where), choice.vertical_half or None
    ):
        return Outcome(
            False,
            f"{obj_name} cannot be picked out of the {where} (its top is not above the walls); the robot did not move",
            [],
            moved=False,
        )
    if not choice.ok:
        return Outcome(False, f"{choice.reason}; the robot did not move", [], moved=False)
    return None


def _pose_of(link):
    for getter in ("get_entity_pose", "get_pose"):
        if hasattr(link, getter):
            return getattr(link, getter)()
    return link.entity.get_pose()


def _link_position(joint) -> list[float]:
    """World position of a joint's child link (SAPIEN 3 accessor names vary by version)."""
    link = joint.get_child_link() if hasattr(joint, "get_child_link") else joint.child_link
    for getter in ("get_entity_pose", "get_pose"):
        if hasattr(link, getter):
            return list(getattr(link, getter)().p)
    return list(link.entity.get_pose().p)


class Twin:
    def __init__(
        self,
        robotwin: Path,
        seed: int,
        render_freq: int,
        place_constrain: str,
        level: int,
        headless: bool = False,
    ) -> None:
        self.robotwin, self.seed, self.render_freq = robotwin, seed, render_freq
        self.place_constrain = place_constrain
        self.level_number = level
        self.level: ts.Level = ts.LEVELS[level]
        self.headless = headless
        self.task = None
        self.actors: dict[str, object] = {}
        self.table_top = TABLE_Z
        self.workspace_camera = None
        self.camera_spec = None
        self.closed_loop = False
        self.finger_axis: str | None = None
        self.finger_links: set[str] = set()
        self.obstacles_enabled = False
        self.planner_status = "table only"
        self.world_state = "table"
        self.world_debug: dict = {}
        self.relaxed_this_pick: list[str] = []
        self.relaxed_streak = 0
        self.motion_stats: dict[str, int] = {"straight": 0, "hops": 0, "vertical_grasps": 0}
        self.finger_axis_source = "none"
        # grasp frames that already failed on an object at (about) its current position
        self.bad_grasps: dict[str, list[tuple[str, tuple]]] = {}
        self.world_boxes: list = []
        self.home_pose: list[float] | None = None

    def setup(self) -> None:
        os.chdir(self.robotwin)
        if str(self.robotwin) not in sys.path:
            sys.path.insert(0, str(self.robotwin))
        from script.collect_data import prepare_task_and_args
        from twin_view import force_raster_shaders

        if self.task is not None:
            self.task.close_env()
        cache = "disabled (VRS_PLANNER_OBSTACLES=0)"
        if os.environ.get("VRS_PLANNER_OBSTACLES", "1") != "0":
            cache = tob.enlarge_collision_cache()
        task, args = prepare_task_and_args(TASK, TASK_CONFIG)
        args.update(
            render_freq=0 if self.headless else self.render_freq,
            save_data=False,
            collect_data=False,
            eval_video_log=False,
        )
        restore = force_raster_shaders()
        try:
            task.setup_demo(now_ep_num=0, seed=self.seed, **args)
        finally:
            for undo in restore:
                undo()
        if not self.headless and getattr(task, "viewer", None) is None:
            raise RuntimeError("RoboTwin did not create a viewer window.")
        self.task = task
        self.table_top = TABLE_Z + float(getattr(task, "table_z_bias", 0.0))
        self.closed_loop = "is_open" in inspect.signature(task.place_actor).parameters
        self._replace_upstream_blocks()
        self._add_workspace_camera()
        self._settle()
        self.finger_axis = self._probe_finger_axis()
        self.prepare_interior_home()
        try:
            from envs.utils import ArmTag

            self.home_pose = self.ee_pose(ArmTag("left"))
        except Exception:
            self.home_pose = None
        self.obstacles_enabled = (
            cache.startswith("collision cache") or cache == "already patched"
        ) and tob.planner_of(task) is not None
        if self.obstacles_enabled:
            self.planner_status = "scene obstacles"
        elif tob.planner_of(task) is None:
            self.planner_status = "table only (no in-process CuRobo planner)"
        else:
            self.planner_status = f"table only ({cache})"
        if self.obstacles_enabled:
            self.sync_planner_world(set())

    def _probe_finger_axis(self) -> str | None:
        """Which gripper-frame axis the fingers open along.

        Closes and re-opens the gripper once and measures how the robot links moved (only the
        finger links move); falls back to the first two gripper joints' child links.
        """
        try:
            import numpy as np
            from envs.utils import ArmTag

            arm = ArmTag("left")
            task = self.task
            links = list(task.robot.left_entity.get_links())
            pose = self.ee_pose(arm)
            if pose is None or len(pose) < 7:
                return None

            def snapshot():
                return {link.get_name(): np.asarray(_pose_of(link).p, float) for link in links}

            opened = snapshot()
            task.plan_success = True
            task.move(task.close_gripper(arm))
            closed = snapshot()
            task.move(task.open_gripper(arm))
            task.plan_success = True
            moved = {
                n: closed[n] - opened[n]
                for n in opened
                if np.linalg.norm(closed[n] - opened[n]) > 0.003
            }
            self.finger_links = set(moved)
            axis = tg.finger_axis_from_displacements(list(moved.values()), pose[3:7])
            if axis is not None:
                self.finger_axis_source = "probe"
                return axis
        except Exception:
            pass
        try:
            joints = [j for j, *_ in self.task.robot.left_gripper][:2]
            if len(joints) < 2:
                return None
            p0, p1 = (_link_position(j) for j in joints)
            pose = self.ee_pose("left")
            if pose is None or len(pose) < 7:
                return None
            delta = [b - a for a, b in zip(p0, p1, strict=True)]
            self.finger_axis_source = "joints"
            return tg.finger_axis_from_vectors(delta, pose[3:7])
        except Exception:
            return None

    def sync_planner_world(
        self,
        exclude: set[str],
        drop: set[str] = frozenset(),
        near: tuple[float, float] | None = None,
        radius: float = 0.0,
    ) -> str:
        """Load the scene (minus ``exclude`` objects / ``drop`` containers) into CuRobo.

        With ``near``/``radius`` the objects closer than ``radius`` (horizontally) are left out
        too: the grasp itself is checked by the twin's finger-room test, while CuRobo's
        collision margin around close neighbours would otherwise reject the open fingers.
        """
        if not self.obstacles_enabled:
            self.world_state = "table"
            return self.planner_status
        if near is not None and radius > 0:
            exclude = set(exclude) | {
                n for n, xyz in self.objects_xyz().items() if math.dist(xyz[:2], near) < radius
            }
        level = self.level
        if drop:
            level = ts.Level(
                level.number,
                level.title,
                level.objects,
                tuple(c for c in level.containers if c.name not in drop),
                level.examples,
            )
        boxes = tob.scene_boxes(
            level, self.objects_xyz(), self.orientations(), self.table_top, exclude
        )
        status = tob.load_world(self.task, boxes)
        self.world_boxes = list(boxes)
        if not self.world_debug:
            try:
                self.world_debug = tob.describe_world(self.task, boxes)
            except Exception as exc:  # diagnostics only
                self.world_debug = {"error": f"{type(exc).__name__}: {exc}"}
        if "obstacle box" not in status:
            self.obstacles_enabled = False  # fall back to table-only planning for the session
            self.planner_status = f"table only ({status})"
            self.world_state = "table"
        else:
            self.world_state = "scene"
        return status

    def relax_world(self, label: str) -> bool:
        """Drop the scene obstacles (table-only world) after a refusal.

        Returns False when the world was already table-only. The scene world is loaded again
        for the next pick. It is never switched off for the whole session, because the
        table-only world lets the arm sweep through a tall carton.
        """
        # A short straight move (lift, descend, final approach, retreat) cannot sweep: it may
        # use the table-only world (keeping a carton already in the deep bin as an obstacle makes
        # the descent of a second carton impossible). Any other motion keeps the tall objects,
        # because in the table-only world a free motion can sweep the cartons in the bin.
        if label in STRAIGHT_LABELS:
            if self.world_state not in ("scene", "relaxed"):
                return False
            tob.load_world(self.task, [])
            self.world_state = "table"
        else:
            if self.world_state != "scene":
                return False
            keep = [
                b
                for b in self.world_boxes
                if b.dims[2] >= TALL_OBSTACLE and " wall " not in f" {b.name} "
            ]
            if len(keep) == len(self.world_boxes):
                return False  # nothing left to drop
            tob.load_world(self.task, keep)
            self.world_state = "relaxed"
        self.relaxed_this_pick.append(label)
        return True

    def restore_world(self, state: str) -> None:
        """Reload the world that was active before a straight move relaxed it to table-only
        (that relaxation is for that one motion only)."""
        if self.world_state != "table" or state == "table":
            return
        if state == "scene":
            tob.load_world(self.task, self.world_boxes)
        else:
            tob.load_world(
                self.task,
                [
                    b
                    for b in self.world_boxes
                    if b.dims[2] >= TALL_OBSTACLE and " wall " not in f" {b.name} "
                ],
            )
        self.world_state = state

    def finger_contact_axis(self, obj_name: str) -> tuple[float, float] | None:
        """Horizontal axis through the finger contacts on the held object (None if unknown)."""
        if not self.finger_links:
            return None
        try:
            target = self.actors[obj_name].actor.get_name()
            points = []
            for contact in self.task.scene.get_contacts():
                names = [b.entity.name for b in contact.bodies]
                if target in names and any(n in self.finger_links for n in names):
                    points += [list(p.position) for p in contact.points]
            return tg.contact_axis(points)
        except Exception:
            return None

    def _replace_upstream_blocks(self) -> None:
        """Swap the upstream task's blocks for this level's objects and containers."""
        import sapien

        task, scene = self.task, self.task.scene
        for attr in UPSTREAM_BLOCKS:
            block = getattr(task, attr, None)
            if block is None:
                continue
            entity = getattr(block, "actor", block)
            try:
                scene.remove_entity(entity)
            except Exception:  # park it far away if removal is not supported
                entity.set_pose(sapien.Pose([0.0, 3.0, 0.05]))
        self.level = ts.LEVELS[self.level_number]
        for container in self.level.containers:
            ts.build_container(scene, container, self.table_top)
        self.actors = {}
        for obj in self.level.objects:
            x, y = self.level.spawn_xy()[obj.name]
            xyz = (x, y, self.table_top + obj.half_height + 0.002)
            self.actors[obj.name] = ts.build_object(scene, obj, xyz, CACHE_DIR)

    def _add_workspace_camera(self) -> None:
        from twin_geometry import add_sapien_camera, load_camera_spec

        self.camera_spec = load_camera_spec()
        try:
            self.workspace_camera = add_sapien_camera(self.task.scene, self.camera_spec)
        except Exception as exc:  # display/perception only
            self.workspace_camera = None
            print(f"WARNING: workspace camera unavailable ({type(exc).__name__}: {exc})")

    def _settle(self, steps: int = 200) -> None:
        for _ in range(steps):
            self.task._step_scene()
        if not self.headless:
            self.task.viewer.render()

    def step_dt(self) -> float:
        scene = self.task.scene
        value = getattr(scene, "timestep", None)
        if value is None and hasattr(scene, "get_timestep"):
            value = scene.get_timestep()
        return float(value or 1.0 / 250.0)

    # ----------------------------------------------------------------- state and images
    def object_xyz(self, name: str) -> list[float]:
        p = self.actors[name].get_pose().p
        return [round(float(v), 4) for v in p]

    def objects_xyz(self) -> dict[str, list[float]]:
        return {name: self.object_xyz(name) for name in self.actors}

    def orientations(self) -> dict[str, list[float]]:
        return {name: list(actor.get_pose().q) for name, actor in self.actors.items()}

    def listing(self) -> list[str]:
        return scene_listing(
            self.level,
            self.objects_xyz(),
            self.table_top,
            orientations=self.orientations(),
            finger_axis=self.finger_axis,
            allow_tilt=True,
        )

    def workspace_rgb(self):
        from twin_geometry import read_rgb

        if self.workspace_camera is None:
            return None
        self.task.scene.update_render()
        return read_rgb(self.workspace_camera)

    def record_frame(self, size: tuple[int, int]):
        """(state, {camera: image}) for the data recorder: arm joints + gripper, wrist + workspace."""
        import cv2
        import numpy as np
        from twin_cameras import label_for

        task = self.task
        task._update_render()
        task.cameras.update_picture()
        images = {}
        for name, data in task.cameras.get_rgb().items():
            if "rgb" in data and label_for(name) == "wrist camera" and "right" not in name.lower():
                images["wrist"] = data["rgb"]
                break
        if self.workspace_camera is not None:
            from twin_geometry import read_rgb

            images["workspace"] = read_rgb(self.workspace_camera)
        images = {
            k: cv2.resize(np.asarray(v)[:, :, :3], size, interpolation=cv2.INTER_AREA)
            for k, v in images.items()
        }
        state = np.asarray(task.robot.get_left_arm_jointState(), np.float32)
        return state, images

    def reset_layout(self, rng) -> dict[str, tuple[float, float, float]]:
        """Re-randomise object poses on the same scene (no rebuild) and send the arm home."""
        import sapien

        spawns = ts.jittered_spawns(self.level, rng)
        for name, actor in self.actors.items():
            obj = self.level.object(name)
            x, y, yaw = spawns[name]
            entity = actor.actor
            entity.set_pose(
                sapien.Pose(
                    [x, y, self.table_top + obj.half_height + 0.002], list(ts.yaw_quat(yaw))
                )
            )
            body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if body is not None:
                body.set_linear_velocity([0.0, 0.0, 0.0])
                body.set_angular_velocity([0.0, 0.0, 0.0])
        from envs.utils import ArmTag

        self._recover(ArmTag("left"))
        self._settle(120)
        return spawns

    def wrist_view(self):
        """(image, CameraModel) of the wrist camera at its current pose, or (None, None)."""
        from twin_cameras import label_for
        from twin_geometry import CameraModel

        task = self.task
        task._update_render()
        task.cameras.update_picture()
        images = task.cameras.get_rgb()
        try:
            configs = task.cameras.get_config()
        except Exception:
            configs = {}
        for name, data in images.items():
            if "rgb" not in data or label_for(name) != "wrist camera" or "right" in name.lower():
                continue
            image = data["rgb"]
            cfg = configs.get(name, {})
            if "intrinsic_cv" in cfg and "cam2world_gl" in cfg:
                h, w = image.shape[:2]
                return image, CameraModel.from_gl_matrices(
                    name, cfg["intrinsic_cv"], cfg["cam2world_gl"], w, h
                )
        return None, None

    def scan_views(self, wrist: bool, echo: Callable[[str], None]) -> list:
        """Workspace view plus (optionally) wrist views from a short sweep; ends at home."""
        from envs.utils import ArmTag
        from twin_geometry import CameraModel

        views = []
        image = self.workspace_rgb()
        if image is not None:
            views.append(("workspace", image, CameraModel.from_spec(self.camera_spec)))
        if not wrist:
            return views
        arm = ArmTag("left")
        try:
            for name, move in WRIST_SCAN_MOVES:
                if move is not None:
                    self.task.plan_success = True
                    self.task.move(
                        self.task.move_by_displacement(arm_tag=arm, x=move[0], y=move[1], z=move[2])
                    )
                    if self.task.plan_success is False:
                        echo(f"  {name}: no safe trajectory, wrist sweep stopped")
                        break
                image, model = self.wrist_view()
                if image is None:
                    echo(
                        "  wrist camera calibration not available; using the workspace camera only"
                    )
                    break
                views.append((name, image, model))
        finally:
            self.task.plan_success = True
            with contextlib.suppress(Exception):
                self.task.move(self.task.back_to_origin(arm))
            self._settle(40)
        return views

    # ----------------------------------------------------------------- execution
    def ee_pose(self, arm) -> list[float] | None:
        """World pose [x, y, z, qw, qx, qy, qz] of the end effector, or None if unavailable."""
        for getter in (
            lambda: self.task.get_arm_pose(arm),
            lambda: self.task.robot.get_left_ee_pose(),
        ):
            try:
                pose = [float(v) for v in getter()]
            except Exception:
                continue
            if len(pose) >= 3:
                return pose
        return None

    def tilts(self) -> dict[str, float]:
        return {name: ts.tilt_deg(list(actor.get_pose().q)) for name, actor in self.actors.items()}

    def guard_moves(self, allowed: set[str]) -> Callable[[str], str | None]:
        """Collision monitor (simulator truth): objects other than ``allowed`` must not move."""
        before = self.objects_xyz()
        before_q = self.orientations()

        def check(_step: str) -> str | None:
            moved = ts.unexpected_moves(before, self.objects_xyz(), allowed)
            tipped = tg.unexpected_tilts(before_q, self.orientations(), allowed | set(moved))
            if not moved and not tipped:
                return None
            return "collision: " + ", ".join(
                [f"{n} moved {d * 100:.1f} cm" for n, d in moved.items()]
                + [f"{n} tipped {a:.0f} deg" for n, a in tipped.items()]
            )

        return check

    def move_with_fallback(self, build, label: str):
        """Execute a RoboTwin action; if the obstacle world makes the planner refuse it, retry
        once with the table-only world. Returns None (already executed)."""
        task = self.task
        task.plan_success = True
        task.move(build())
        if task.plan_success is False and self.relax_world(label):
            task.plan_success = True
            task.move(build())
        if task.plan_success is False:
            raise StepError(f"no safe trajectory for the {label} motion")
        return None

    def orient_above(
        self, arm, pre_grasp: list[float], safe_z: float, guard, high: float | None = None
    ) -> str | None:
        """Turn the gripper to the grasp orientation high above the object, then go straight
        down to the pre-grasp pose: the wrist (and its camera) never turns near the objects
        and the gripper axis stays vertical while it descends. None if the turn is refused."""
        task = self.task
        pose = self.ee_pose(arm)
        if pose is None or len(pose) < 7:
            return "orientation set by RoboTwin (no end-effector pose)"
        if high is None:
            high = max(safe_z, pose[2], pre_grasp[2])
        hover = [pre_grasp[0], pre_grasp[1], high, *pre_grasp[3:7]]
        try:
            from envs.utils import Action
        except Exception:
            return "orientation set by RoboTwin (no Action class)"
        task.plan_success = True
        task.move((arm, [Action(arm, "move", target_pose=hover)]))
        if task.plan_success is False and self.relax_world("orient"):
            task.plan_success = True
            task.move((arm, [Action(arm, "move", target_pose=hover)]))
        if task.plan_success is False:
            task.plan_success = True
            return None
        problem = guard("orient")
        if problem:
            raise StepError(problem)
        drop = pre_grasp[2] - high
        how = self.linear_move(arm, 0.0, 0.0, drop, guard, "lower to the pre-grasp pose")
        return f"wrist set at z={high:.2f} m, then straight down {-drop * 100:.0f} cm ({how})"

    def turn_above(
        self, arm, pre_grasp: list[float], safe_z: float, guard, high: float | None = None
    ) -> str | None:
        """``orient_above`` that backs out (None, nothing touched) when the descent is refused."""
        try:
            return self.orient_above(arm, pre_grasp, safe_z, guard, high)
        except StepError as exc:
            if str(exc).startswith("collision"):
                raise
            self.rise_to(arm, safe_z)
            return None

    def reachable_grasp(self, actor, arm, frame) -> list[float] | None:
        """RoboTwin's pre-grasp pose for this contact frame, or None if the planner cannot
        reach it (nothing moves)."""
        task = self.task
        saved = actor.config.get("contact_points_pose")
        actor.config["contact_points_pose"] = [frame]
        try:
            task.plan_success = True
            pose = task.get_grasp_pose(actor, arm, contact_point_id=0, pre_dis=0.09)
            if pose is not None and len(pose) >= 7 and float(pose[0]) != -1.0:
                return [float(v) for v in pose]
            return None
        except Exception:
            return None
        finally:
            task.plan_success = True
            actor.config["contact_points_pose"] = saved

    def pose_reachable(self, pose: list[float]) -> bool:
        """Can CuRobo plan from the current joints to this exact end-effector pose?

        Uses RoboTwin's ``left_plan_path`` (no motion). True when the check is unavailable:
        the motion itself is then the test (a refused motion does not move the arm).
        """
        robot = getattr(self.task, "robot", None)
        plan = getattr(robot, "left_plan_path", None)
        if plan is None:
            return True
        try:
            result = plan(pose)
        except Exception:
            return True
        status = result.get("status") if isinstance(result, dict) else None
        return status is None or str(status) == "Success"

    def rise_to(self, arm, z: float) -> None:
        """Best effort: straight up to height ``z`` (used to back out of a refused approach)."""
        pose = self.ee_pose(arm)
        if pose is None or z - pose[2] < 0.005:
            return
        task = self.task
        task.plan_success = True
        with contextlib.suppress(Exception):
            task.move(task.move_by_displacement(arm_tag=arm, z=z - pose[2]))
        task.plan_success = True

    def prepare_interior_home(self) -> bool:
        """Move the home posture off the joint limits once at setup.

        Panthera's homestate is all zeros, the lower limit of joints 2 and 3; RoboTwin's
        back_to_origin plans to the tool pose of that posture, so every return home ends on
        (or a hair past) a limit, a start state CuRobo may reject (the project's native
        stack moved its home inside the limits for the same reason: native_initial_home).
        Here the same interior home is driven once and becomes the 'origin' pose."""
        robot = getattr(self.task, "robot", None)
        state = getattr(robot, "left_homestate", None)
        if robot is None or state is None or len(state) < 3:
            return False
        try:
            state = list(state)
            state[1] = max(float(state[1]), INTERIOR_HOME[0])
            state[2] = max(float(state[2]), INTERIOR_HOME[1])
            robot.left_homestate = state
            robot.move_to_homestate()
            self._settle(200)
            robot.left_original_pose = robot.get_left_ee_pose()
        except Exception:
            return False
        return True

    def nudge_into_limits(self, eps: float = 0.01) -> bool:
        """Move any arm joint that sits on (or just past) its limit 0.01 rad inside it.

        Panthera's home posture is all zeros, exactly the lower limit of joints 2 and 3;
        after a move home the joint controller can settle a hair past it, and CuRobo then
        rejects that start state, so every later plan fails. Best effort; True if changed.
        """
        try:
            entity = self.task.robot.left_entity
            joints = list(entity.get_active_joints())
            qpos = [float(v) for v in entity.get_qpos()]
            arm = {j.get_name() for j in self.task.robot.left_arm_joints}
        except Exception:
            return False
        changed = False
        for i, joint in enumerate(joints):
            if joint.get_name() not in arm or i >= len(qpos):
                continue
            try:
                lo, hi = (float(v) for v in joint.get_limits()[0])
            except Exception:
                continue
            q = min(max(qpos[i], lo + eps), hi - eps)
            if abs(q - qpos[i]) > 1e-6:
                qpos[i] = q
                changed = True
                with contextlib.suppress(Exception):
                    joint.set_drive_target(q)
        if changed:
            with contextlib.suppress(Exception):
                entity.set_qpos(qpos)
                self._settle(5)
        return changed

    def force_home(self, arm=None) -> bool:
        """Last resort when CuRobo refuses every motion (for instance a start state it
        rejects): drive the joints to the home posture directly, without planning. Without
        this, one refused descent can make every later motion fail. Returns True if the
        drive was available."""
        robot = getattr(self.task, "robot", None)
        drive = getattr(robot, "move_to_homestate", None)
        if drive is None:
            return False
        if arm is not None:
            pose = self.ee_pose(arm)
            if pose is not None:
                self.rise_to(arm, pose[2] + ts.RETREAT_DZ)  # clear the objects below first
        try:
            drive()
            self._settle(250)
        except Exception:
            return False
        self.nudge_into_limits()
        self.task.plan_success = True
        print("  NOTE: planner refused to go home; joints driven to the home posture directly")
        return True

    def go_home(self, arm) -> bool:
        """Best effort: back to the start posture (the planner's known-good start state)."""
        for _ in range(2):
            try:
                self.move_with_fallback(lambda: self.task.back_to_origin(arm), "home")
                self.nudge_into_limits()
                return True
            except Exception:
                self.task.plan_success = True
                if not self.nudge_into_limits():  # nothing to fix: no point retrying
                    break
        return self.force_home(arm)

    def away_from_home(self, arm) -> bool:
        pose = self.ee_pose(arm)
        if pose is None or self.home_pose is None:
            return True
        return math.dist(pose[:3], self.home_pose[:3]) > 0.03

    def straight_to(self, arm, target: list[float], guard, label: str) -> str:
        before = self.world_state
        try:
            return self._straight_to(arm, target, guard, label)
        finally:
            self.restore_world(before)

    def _straight_to(self, arm, target: list[float], guard, label: str) -> str:
        """Move the tool to ``target`` (same orientation) along a short straight line.

        Vertical moves use ``linear_move``. A tilted approach is one CuRobo plan that holds
        the orientation (RoboTwin's own grasp approach uses the same constraint), retried
        in the table-only world, else a plain move to the pose. Raises StepError if refused.
        """
        pose = self.ee_pose(arm)
        if pose is None or len(pose) < 7:
            raise StepError(f"no safe trajectory for the {label} motion")
        dx, dy, dz = (target[k] - pose[k] for k in range(3))
        if abs(dx) + abs(dy) < 0.003:
            return self.linear_move(arm, 0.0, 0.0, dz, guard, label)
        from envs.utils import Action

        task = self.task
        # orientation held; fingers (tool y) held too when they are horizontal, which keeps
        # the approach in its vertical grasp plane (no free fallback: a curved approach
        # pushed objects)
        hold = [1, 1, 1, 0, 0, 0]
        if abs(tg.quat_to_mat(target[3:7])[2, 1]) < 0.05:
            hold = [1, 1, 1, 0, 1, 0]
        for attempt in (0, 1):
            if attempt and not self.relax_world(label):
                break
            task.plan_success = True
            task.move((arm, [Action(arm, "move", target_pose=list(target), constraint_pose=hold)]))
            if task.plan_success is not False:
                problem = guard(label)
                if problem:
                    raise StepError(problem)
                return "one straight motion"
        task.plan_success = True
        raise StepError(f"no safe trajectory for the {label} motion")

    def linear_move(self, arm, dx: float, dy: float, dz: float, guard, label: str) -> str:
        before = self.world_state
        try:
            return self._linear_move(arm, dx, dy, dz, guard, label)
        finally:
            self.restore_world(before)

    def _linear_move(self, arm, dx: float, dy: float, dz: float, guard, label: str) -> str:
        """Straight move of the end effector by (dx, dy, dz) in the world frame.

        First one CuRobo plan that holds the gripper orientation and the height (horizontal) or
        the horizontal position (vertical): a single smooth motion. If that plan is refused, the
        move is split into short hops (each one is a short, nearly straight plan). The collision
        guard runs after the motion. Returns how the move was made.
        """
        task = self.task
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-4:
            return "no move needed"
        vertical = abs(dx) + abs(dy) < 1e-6
        pose = self.ee_pose(arm)
        hold = None
        if pose is not None and len(pose) >= 7:
            hold = tg.hold_vector(pose[3:7], "vertical" if vertical else "horizontal")
            if hold is None:  # tilted gripper
                fingers_flat = abs(tg.quat_to_mat(pose[3:7])[2, 1]) < 0.05
                if vertical and fingers_flat:
                    hold = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]  # orientation + finger axis
                elif length <= 0.12:
                    hold = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # short: orientation only
                # long tilted horizontal moves keep the 10 cm hops
        if hold is not None and length > 0.02:
            try:
                from envs.utils import Action

                target = [pose[0] + dx, pose[1] + dy, pose[2] + dz, *pose[3:7]]
                task.plan_success = True
                task.move((arm, [Action(arm, "move", target_pose=target, constraint_pose=hold)]))
                if task.plan_success is False and self.relax_world(label):
                    task.plan_success = True  # one smooth motion beats many short hops
                    task.move(
                        (arm, [Action(arm, "move", target_pose=target, constraint_pose=hold)])
                    )
                if task.plan_success is not False:
                    problem = guard(label)
                    if problem:
                        raise StepError(problem)
                    self.motion_stats["straight"] += 1
                    return "one straight motion"
            except StepError:
                raise
            except Exception:  # no Action/constraint support in this RoboTwin build
                pass
            task.plan_success = True  # a refused plan did not move the arm
        if vertical:
            hops = [(0.0, 0.0, dz)]
        else:
            hops = [(hx, hy, 0.0) for hx, hy in ts.transit_waypoints((0.0, 0.0), (dx, dy))]
        for hx, hy, hz in hops:
            task.plan_success = True
            task.move(task.move_by_displacement(arm_tag=arm, x=hx, y=hy, z=hz))
            if task.plan_success is False and self.relax_world(label):
                task.plan_success = True
                task.move(task.move_by_displacement(arm_tag=arm, x=hx, y=hy, z=hz))
            if task.plan_success is False:
                raise StepError(f"no safe trajectory for the {label} motion")
            problem = guard(label)
            if problem:
                raise StepError(problem)
        self.motion_stats["hops"] += len(hops)
        return f"{len(hops)} short hop(s)"

    def pick_and_place(self, obj_name: str, dest_name: str, echo: Callable[[str], None]) -> Outcome:
        from envs.utils import ArmTag

        task = self.task
        level = self.level
        self.nudge_into_limits()  # a start state on a joint limit makes CuRobo refuse everything
        objects_xyz = self.objects_xyz()
        orientations = self.orientations()
        blocked = precheck(
            level,
            obj_name,
            dest_name,
            objects_xyz,
            self.table_top,
            orientations=orientations,
            finger_axis=self.finger_axis,
            allow_tilt=True,
        )
        if blocked is not None:
            return blocked
        self.relaxed_this_pick = []
        source = ts.location_of(level, objects_xyz[obj_name], self.table_top)
        # Inside a container its walls are checked by the twin's finger-room test; in CuRobo
        # (with its margin) they rejected the vertical grasps of a carton in the deep bin and
        # left only a 50 deg tilt that could not be carried.
        holder = {
            c.name for c in level.containers if c.name == source and c.kind not in ("table", "pad")
        }
        world = self.sync_planner_world(
            {obj_name},
            drop=holder,
            near=tuple(objects_xyz[obj_name][:2]),
            radius=GRASP_CLEAR_RADIUS,
        )
        obj, dest = level.object(obj_name), level.container(dest_name)
        ee_now = self.ee_pose(ArmTag("left"))  # wrist yaw now: grasps turn it at most 90 deg
        poses = object_poses(objects_xyz, orientations=orientations)
        choice = grasp_choice(
            level,
            obj_name,
            objects_xyz,
            poses,
            self.table_top,
            self.finger_axis,
            dest,
            True,  # tilted approaches always offered: see twin_kinematics
            ee_now[3:7] if ee_now and len(ee_now) >= 7 else None,
        )
        started_upright = not choice.lying
        offset_z = choice.grasp_offset_world[2]
        grasp_above_bottom = choice.vertical_half + offset_z
        actor = self.actors[obj_name]
        actor.config["contact_points_pose"] = choice.frames
        others = [tuple(xyz[:2]) for n, xyz in objects_xyz.items() if n != obj_name]
        slot = ts.choose_slot(dest, others)
        clearance = ts.transport_clearance(level, objects_xyz, self.table_top, obj_name)
        # smallest straight lift: clear the holding container's rim and every object within
        # 12 cm by 2 cm; the transit climbs the rest (a carton in the deep bin cannot be
        # lifted the full 15 cm with any orientation that grasps it: twin_kinematics scan)
        min_lift = minimal_lift(level, objects_xyz, obj_name, self.table_top, holder, slot)
        arm = ArmTag("left")
        notes: dict[str, str] = {}
        notes["grasp"] = (
            choice.describe()
            + (
                f", grasp point {offset_z * 100:+.1f} cm from centre"
                if abs(offset_z) > 1e-3
                else ""
            )
            + (" (finger axis unknown)" if self.finger_axis is None else "")
            + f"; planner world: {world}"
        )
        used: dict[str, object] = {}

        def grasp():
            # Candidates come top-down first, then tilted about the (horizontal) finger axis:
            # the grasp plane stays vertical, only the approach leans. Each candidate is first
            # checked with the Panthera kinematic model (twin_kinematics) for the whole pick:
            # hover, pre-grasp, grasp, lift, above the slot, release, all with its orientation.
            # Feasible ones are executed first by the twin itself with the exact poses (no
            # RoboTwin re-rotation); the others are tried after them; RoboTwin's own grasp is
            # the last resort. Nothing is touched until a candidate's final approach runs.
            here = tuple(self.object_xyz(obj_name))
            bad = {
                key
                for key, xyz in self.bad_grasps.get(obj_name, [])
                if math.dist(xyz, here) < BAD_GRASP_RADIUS
            }
            frames = [(i, f) for i, f in enumerate(choice.frames) if grasp_key(f) not in bad][
                :GRASP_SEARCH_LIMIT
            ]
            if len(frames) < min(len(choice.frames), GRASP_SEARCH_LIMIT):
                notes["grasp"] += "; skipped frame(s) that already failed on this object here"
            stats = {"infeasible": 0, "turn": 0, "approach": 0}

            def poses_of(frame):
                p = actor.get_pose()
                scale = actor.config.get("scale", [1.0, 1.0, 1.0])
                return tg.grasp_poses(list(p.p), list(p.q), frame, scale, PRE_GRASP_DIS)

            plans: dict[str, dict] = {}

            checked: dict[str, bool] = {}

            def feasible(frame) -> bool:
                key = grasp_key(frame)
                if key in checked:  # a second search (reduced world, from home) reuses it
                    return checked[key]
                pre, target = poses_of(frame)
                c = self.object_xyz(obj_name)
                plan = plan_pick(
                    pre,
                    target,
                    c,
                    slot,
                    ts.floor_top(dest, self.table_top) + choice.vertical_half + 0.01,
                    ts.lift_dz(obj, c[2], clearance, self.table_top),
                    min_lift,
                    safe_z,
                )
                if plan is not None:
                    plans[key] = plan
                checked[key] = plan is not None
                return checked[key]

            def exact(frame) -> bool:
                """Twin-executed grasp at the exact frame pose. False: refused before the
                fingers reached the object (nothing touched, the arm backs out upwards)."""
                pre, target = poses_of(frame)
                plan = plans.get(grasp_key(frame))
                how = self.turn_above(arm, pre, safe_z, guard, plan["hover_z"] if plan else None)
                if how is None:
                    stats["turn"] += 1
                    return False
                try:
                    down = self.straight_to(arm, target, guard, "final approach")
                except StepError as exc:
                    if str(exc).startswith("collision"):
                        raise
                    stats["approach"] += 1
                    self.rise_to(arm, safe_z)
                    return False
                task.plan_success = True
                task.move(task.close_gripper(arm))
                task.plan_success = True
                problem = guard("grasp")
                if problem:
                    raise StepError(problem)
                tilt = tg.approach_tilt_deg(pre)
                notes["grasp"] += (
                    f"; {how}; approach {tilt:.0f} deg from vertical in a vertical grasp plane, "
                    f"straight {PRE_GRASP_DIS * 100:.0f} cm final approach ({down})"
                )
                if tilt <= VERTICAL_TOL_DEG:
                    self.motion_stats["vertical_grasps"] += 1
                return True

            def robotwin(index, frame) -> bool:
                actor.config["contact_points_pose"] = [frame]
                pre = self.reachable_grasp(actor, arm, frame)
                if pre is None:
                    return False
                actor.config["contact_points_pose"] = [frame]
                how = self.turn_above(arm, pre, safe_z, guard)
                if how is None:
                    return False
                task.plan_success = True
                try:
                    task.move(
                        task.grasp_actor(
                            actor, arm_tag=arm, pre_grasp_dis=PRE_GRASP_DIS, contact_point_id=0
                        )
                    )
                except Exception as exc:  # e.g. UnStableError "No valid grasp pose"
                    problem = guard("grasp")
                    if problem:
                        raise StepError(problem) from exc
                    task.plan_success = True
                    return False
                problem = guard("grasp")
                if problem:
                    raise StepError(problem)
                if task.plan_success is False:
                    task.plan_success = True
                    return False
                notes["grasp"] += f"; {how}; RoboTwin grasp (last resort)"
                return True

            def take(index, frame, how: str) -> None:
                used["frame"] = frame
                notes["grasp"] += f"; using candidate {index + 1}/{len(frames)} ({how})"

            def search() -> bool:
                deferred = []
                first = not checked
                for index, frame in frames:
                    if not feasible(frame):
                        stats["infeasible"] += first
                        deferred.append((index, frame))
                        continue
                    if exact(frame):
                        take(index, frame, "kinematically feasible")
                        return True
                for index, frame in deferred:
                    if exact(frame):
                        take(index, frame, "outside the kinematic model, planner accepted it")
                        return True
                for index, frame in frames[:ROBOTWIN_FALLBACK_LIMIT]:
                    if robotwin(index, frame):
                        take(index, frame, "RoboTwin")
                        return True
                return False

            def report() -> None:
                parts = [
                    f"{stats['infeasible']} outside the arm's reach for the whole pick",
                    f"{stats['turn']} wrist setting refused",
                    f"{stats['approach']} final approach refused",
                ]
                parts = [p for p in parts if not p.startswith("0 ")]
                if parts:
                    notes["grasp"] += "; candidates skipped: " + ", ".join(parts)

            ok = search()
            if not ok and self.relax_world("grasp"):
                ok = search()
                if ok:
                    notes["grasp"] += "; needed the reduced planner world (tall objects kept)"
            if not ok and self.away_from_home(arm) and self.go_home(arm):
                notes["grasp"] += "; nothing reachable from where the arm was: went home"
                ok = search()
            report()
            if ok:
                return None
            raise NoMotionError(
                f"{NO_GRASP} among {len(frames)} candidate(s) (planner world: {world})"
            )

        held_z: dict[str, float] = {}
        start_xyz = list(objects_xyz[obj_name])
        guard = self.guard_moves({obj_name})
        all_clear = ts.transport_clearance(level, objects_xyz, self.table_top, "")
        safe_z = ts.safe_ee_z(self.table_top, all_clear)

        def hops_to(goal_xy, label: str, start_xy=None) -> str:
            """Straight horizontal move at constant height (CuRobo-constrained, else hops)."""
            if start_xy is None:
                start_xy = self.object_xyz(obj_name)[:2]
            dx, dy = goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]
            return self.linear_move(arm, dx, dy, 0.0, guard, label)

        def approach():
            # Best effort: hover above the object at the safe height before RoboTwin's grasp
            # motion, so the arm does not sweep low through the neighbours on its way there.
            pose = self.ee_pose(arm)
            if pose is None:
                notes["approach"] = "skipped (no end-effector pose)"
                return None
            x, y, _ = self.object_xyz(obj_name)
            try:
                up = safe_z - pose[2]
                if up > 0.005:
                    task.plan_success = True
                    task.move(task.move_by_displacement(arm_tag=arm, z=up))
                    if task.plan_success is False:
                        raise StepError("cannot rise to the safe height")
                how = hops_to((x, y), "approach", start_xy=self.ee_pose(arm)[:2])
                notes["approach"] = f"hovered above the object at z={safe_z:.2f} m ({how})"
            except StepError as exc:
                if str(exc).startswith("collision"):
                    raise
                notes["approach"] = f"partial ({exc}); grasping from here"
            task.plan_success = True  # a refused hop did not move the arm
            return None

        def lift():
            measured = self.finger_contact_axis(obj_name)
            if measured is not None and "frame" in used and self.finger_axis:
                expected = tg.finger_world_axis(
                    used["frame"], list(actor.get_pose().q), self.finger_axis
                )
                off = tg.axis_angle_deg(measured, expected[:2])
                notes["lift"] = f"finger contacts {off:.0f} deg from the expected axis; "
                if off > 45 and self.finger_axis_source != "probe":
                    self.finger_axis = "z" if self.finger_axis == "y" else "y"
                    notes["lift"] += f"finger axis corrected to {self.finger_axis}; "
                elif off > 45:
                    notes["lift"] += (
                        "object turned in the grip (finger axis kept: measured at setup); "
                    )
            full = ts.lift_dz(obj, self.object_xyz(obj_name)[2], clearance, self.table_top)
            pose = self.ee_pose(arm)
            dz = full
            if pose is not None and len(pose) >= 7:
                dz = best_lift(pose, lift_options(full, min_lift)) or full
                held_z["transport_ee"] = pose[2] + full
            how = self.linear_move(arm, 0.0, 0.0, dz, guard, "lift")
            notes["lift"] = notes.get("lift", "") + (
                f"up {dz * 100:.1f} cm ({how})"
                + (
                    f"; the transit climbs the remaining {(full - dz) * 100:.1f} cm to clear "
                    f"{clearance * 100:.1f} cm above the table"
                    if full - dz > 0.005
                    else f"; clears {clearance * 100:.1f} cm above the table"
                )
            )
            return None

        def transit():
            if self.obstacles_enabled:
                self.sync_planner_world({obj_name})  # lifted clear: everything counts again
            pose = self.ee_pose(arm)
            turn, quat2 = 0.0, None
            if pose is not None and len(pose) >= 7:
                turn, quat2 = tg.yaw_to_lean(
                    pose[3:7], (slot[0] - BASE_XY[0], slot[1] - BASE_XY[1])
                )
            climb = 0.0
            if pose is not None and len(pose) >= 7 and "transport_ee" in held_z:
                climb = held_z["transport_ee"] - pose[2]
            if abs(turn) >= SLOT_TURN_MIN_DEG or climb > 0.01:
                if abs(turn) < SLOT_TURN_MIN_DEG:
                    turn, quat2 = 0.0, list(pose[3:7])
                # A tilted grasp leans away from the robot where it was picked; above the slot
                # it must lean away from the robot there too, or the arm cannot reach down
                # (CuRobo refuses the descent). Turn about the vertical while carrying: the object
                # stays upright and only spins in place.
                c = self.object_xyz(obj_name)
                off = _rotz(turn) @ np.array([c[k] - pose[k] for k in range(3)])
                goal = [slot[0] - off[0], slot[1] - off[1], pose[2] + max(climb, 0.0), *quat2]
                from envs.utils import Action

                for attempt in (0, 1):
                    if attempt and not self.relax_world("transit"):
                        break
                    task.plan_success = True
                    task.move((arm, [Action(arm, "move", target_pose=goal)]))
                    if task.plan_success is not False:
                        problem = guard("transit")
                        if problem:
                            raise StepError(problem)
                        notes["transit"] = (
                            "at the safe height (one planned move"
                            + (f", climbing {climb * 100:.1f} cm" if climb > 0.01 else "")
                            + (
                                f", wrist turned {turn:+.0f} deg about the vertical so the "
                                "approach leans away from the robot above the slot"
                                if turn
                                else ""
                            )
                            + ")"
                        )
                        return None
                task.plan_success = True
                if climb > 0.01:  # the planned move was refused: climb straight, then hops
                    self.linear_move(arm, 0.0, 0.0, climb, guard, "lift")
            how = hops_to(slot, "transit")
            notes["transit"] = f"at the safe height ({how})"
            return None

        def align():
            # In the air, above the slot: correct the residual offset before going down.
            x, y, _ = self.object_xyz(obj_name)
            error_mm = math.dist(slot, (x, y)) * 1000
            if error_mm <= ts.ALIGN_TOL * 1000:
                notes["align"] = f"error {error_mm:.1f} mm, no correction"
                return None
            hops_to(slot, "align")
            x2, y2, _ = self.object_xyz(obj_name)
            notes["align"] = (
                f"corrected {error_mm:.1f} -> {math.dist(slot, (x2, y2)) * 1000:.1f} mm above the slot"
            )
            return None

        def descend():
            # Straight down over the slot (replaces RoboTwin place_actor, whose joint-space
            # move to its pre-place pose swept the carried object through the neighbours).
            x, y, z = self.object_xyz(obj_name)
            dz = ts.descend_dz(
                obj, z, list(actor.get_pose().q), dest, self.table_top, 0.0, grasp_above_bottom
            )
            if dz < -0.001:
                self.linear_move(arm, 0.0, 0.0, dz, guard, "descend")
            x, y, _ = self.object_xyz(obj_name)
            error_mm = math.dist(slot, (x, y)) * 1000
            if error_mm > ts.ALIGN_MAX * 1000:
                raise StepError(
                    f"object is {error_mm:.0f} mm from the target slot (dropped or slipped); not releasing"
                )
            gap = ts.release_gap(dest, obj, 0.0, grasp_above_bottom)
            notes["descend"] = (
                f"down {-dz * 100:.1f} cm; release {gap * 100:.1f} cm "
                f"above the floor, {error_mm:.1f} mm from the slot"
            )
            return None

        def retreat():
            before = self.object_xyz(obj_name)
            held_z["before"], held_z["xy"] = before[2], before[:2]
            dz = ts.RETREAT_DZ
            pose = self.ee_pose(arm)
            if pose is not None:
                dz = max(dz, safe_z - pose[2])
            # constrained straight up: a free-path retreat can drag the object sideways
            how = self.linear_move(arm, 0.0, 0.0, dz, guard, "retreat")
            notes["retreat"] = f"straight up {dz * 100:.1f} cm ({how})"
            return None

        def check_released():
            x, y, z = self.object_xyz(obj_name)
            rose = z - held_z.get("before", 0.0)
            if rose > 0.01:
                raise StepError(f"object is {STUCK_TEXT} (rose {rose * 100:.1f} cm with it)")
            drag = math.dist((x, y), held_z.get("xy", (x, y)))
            if drag > DRAG_TOL:
                notes["check release"] = (
                    f"the fingers pushed the object {drag * 100:.1f} cm sideways while opening/rising"
                )
            return None

        steps = [
            ("approach", approach),
            ("grasp", grasp),
            ("lift", lift),
            ("transit", transit),
            ("align", align),
            ("descend", descend),
            ("release", self._open_and_settle(arm)),
            ("retreat", retreat),
            ("check release", check_released),
            # the object is already placed: going home never fails the step (planner first,
            # then the direct drive), and the joints are kept off their limits
            ("home", lambda: self.go_home(arm) and None),
        ]
        task.plan_success = True
        try:
            reports = run_steps(
                steps,
                execute=task.move,
                plan_ok=lambda: task.plan_success is not False,
                object_xyz=lambda: self.object_xyz(obj_name),
                echo=echo,
                notes=notes,
                guard=guard,
            )
        finally:
            actor.config["contact_points_pose"] = ts.contact_poses(0.0)
        if not reports or not reports[-1].ok:
            failed = reports[-1] if reports else StepReport("start", False, "no step ran")
            placed_back = ""
            carried = failed.name in ("lift", "transit", "align", "descend")
            if "frame" in used and carried:
                # this frame missed the object or could not carry it: a new attempt from the
                # same spot starts with the other frames
                self.bad_grasps.setdefault(obj_name, []).append(
                    (grasp_key(used["frame"]), tuple(start_xyz))
                )
            if carried:
                placed_back = self._put_back(arm, obj_name, start_xyz, guard)
            if failed.name == "grasp" and failed.detail.startswith(NO_GRASP):
                self._recover(arm, None)
                self._end_pick_world()
                return Outcome(
                    False,
                    f"{obj_name}: {failed.detail}; the object was not touched",
                    reports,
                    moved=False,
                )
            self._recover(arm, obj_name)
            self._end_pick_world()
            return Outcome(
                False, f"motion '{failed.name}' failed: {failed.detail}{placed_back}", reports
            )
        self._end_pick_world()
        self._settle(60)
        late = guard("settle")
        x, y, z = self.object_xyz(obj_name)
        if late:
            return Outcome(False, f"after placing: {late}", reports)
        error_mm = math.dist(slot, (x, y)) * 1000
        if not ts.in_destination(level, dest, (x, y, z), self.table_top):
            return Outcome(
                False,
                f"{obj_name} ended at x={x:+.3f}, y={y:+.3f}, outside the {dest_name}",
                reports,
            )
        lying = ""
        if (
            started_upright
            and obj.shape not in ("sphere", "tetrahedron")
            and ts.toppled(obj, list(actor.get_pose().q))
        ):
            # It is where the operator wanted it; standing it up again would need a
            # re-grasp that leaves it lying anyway, so the step counts as done (reported).
            lying = "; it fell over inside and was left lying there"
        where_text = "on the table" if dest.kind == "table" else f"in the {dest_name}"
        return Outcome(
            True,
            f"{obj_name} is {where_text} (x={x:+.3f}, y={y:+.3f}; {error_mm:.0f} mm from slot centre){lying}",
            reports,
        )

    def _put_back(self, arm, obj_name: str, start_xyz: list[float], guard) -> str:
        """After a failed carry: bring the held object back over where it was picked and set
        it down there instead of opening the gripper high up (a carton dropped from
        15 cm falls over in the bin). Best effort; returns a note."""
        try:
            x, y, z = self.object_xyz(obj_name)
            if z < start_xyz[2] + 0.01:
                return ""  # not lifted (missed grasp): nothing to put back
            if math.dist((x, y), start_xyz[:2]) > 0.015:
                self.linear_move(arm, start_xyz[0] - x, start_xyz[1] - y, 0.0, guard, "put back")
            x, y, z = self.object_xyz(obj_name)
            if math.dist((x, y), start_xyz[:2]) > 0.015:
                return ""
            dz = start_xyz[2] + 0.004 - z
            if dz < -0.005:
                self.linear_move(arm, 0.0, 0.0, dz, guard, "put back")
            return "; the object was set back down where it was picked"
        except Exception:
            self.task.plan_success = True
            return ""

    def _end_pick_world(self) -> None:
        """Reload the full scene after a pick (every pick starts obstacle-aware again)."""
        self.relaxed_streak = self.relaxed_streak + 1 if self.relaxed_this_pick else 0
        self.sync_planner_world(set())

    def _open_and_settle(self, arm) -> Callable[[], object]:
        def release():
            self._settle(25)  # let the carried object stop swinging after the alignment move
            self.task.move(self.task.open_gripper(arm))
            self._settle(80)  # let the object drop and come to rest before moving the arm
            return None

        return release

    def pick_and_place_policy(
        self,
        client,
        obj_name: str,
        dest_name: str,
        echo: Callable[[str], None],
        max_actions: int = 450,
    ) -> Outcome:
        """Learned executor: a LeRobot policy (ACT/SmolVLA) drives the arm from camera images."""
        import numpy as np

        objects_xyz = self.objects_xyz()
        blocked = precheck(self.level, obj_name, dest_name, objects_xyz, self.table_top)
        if blocked is not None:
            return blocked
        if not hasattr(self.task, "take_action"):
            return Outcome(
                False,
                "this RoboTwin build has no take_action(); policy execution unavailable",
                [],
                moved=False,
            )
        obj, dest = self.level.object(obj_name), self.level.container(dest_name)
        text = policy_instruction(obj_name, dest_name, dest.kind)
        echo(f"    policy instruction: '{text}'")
        size = record_size()
        client.reset()
        done_at = None
        for i in range(max_actions):
            state, images = self.record_frame(size)
            action = client.act(state, images, text)
            self.task.take_action(np.asarray(action, dtype=np.float64), action_type="qpos")
            if (i + 1) % 15 == 0:
                xyz = self.object_xyz(obj_name)
                if policy_done(dest, obj, xyz, self.table_top):
                    done_at = i + 1
                    break
                echo(
                    f"    policy t={i + 1}: object at x={xyz[0]:+.3f} y={xyz[1]:+.3f} z={xyz[2]:.3f}"
                )
        from envs.utils import ArmTag

        self._recover(ArmTag("left"))
        self._settle(60)
        x, y, z = self.object_xyz(obj_name)
        report = [
            StepReport(
                "policy", done_at is not None, f"{done_at or max_actions} actions", [x, y, z]
            )
        ]
        if ts.is_inside(dest, (x, y, z), self.table_top):
            return Outcome(
                True,
                f"{obj_name} is in the {dest_name} after {done_at or max_actions} policy actions",
                report,
            )
        return Outcome(
            False,
            f"policy did not finish within {max_actions} actions; {obj_name} at x={x:+.3f}, y={y:+.3f}",
            report,
        )

    def _recover(self, arm: object, held: str | None = None) -> None:
        """Open the gripper, move straight up, then go home (never drag a stuck object home)."""
        task = self.task
        task.plan_success = True
        try:
            task.move(task.open_gripper(arm))
            self._settle(80)
            before = self.object_xyz(held)[2] if held else None
            task.move(task.move_by_displacement(arm_tag=arm, z=ts.RETREAT_DZ))
            if held and self.object_xyz(held)[2] - before > 0.01:
                print(
                    f"WARNING: {held} is still stuck in the gripper; the arm stays up (not going home)."
                )
                return
            task.plan_success = True
            task.move(task.back_to_origin(arm))
            if task.plan_success is False:
                self.force_home(arm)
            else:
                self.nudge_into_limits()
        except Exception:  # best effort; the failure is already reported
            self.force_home(arm)


def policy_instruction(obj: str, dest: str, kind: str) -> str:
    """English instruction in the same style as the recorded training episodes."""
    return (
        f"put the {obj} on the {dest}"
        if kind == "pad"
        else f"pick up the {obj} and put it in the {dest}"
    )


def policy_done(
    dest: ts.ContainerSpec, obj: ts.ObjectSpec, xyz: list[float], table_top: float
) -> bool:
    """Object is inside the destination and resting (not still held in the air)."""
    resting = xyz[2] <= ts.floor_top(dest, table_top) + obj.half_height + 0.01
    return ts.is_inside(dest, xyz, table_top) and resting


def record_size() -> tuple[int, int]:
    from twin_geometry import CAMERA_CONFIG

    cfg = {"width": 320, "height": 240}
    if CAMERA_CONFIG.is_file():
        cfg.update(json.loads(CAMERA_CONFIG.read_text()).get("record", {}))
    return int(cfg["width"]), int(cfg["height"])


class VisionState:
    """Last VLM detections and their mapping to simulator objects (sim-only execution aid)."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.to_truth: dict[str, str | None] = {}
        self.rows: list[str] = []
        self.positions: dict[str, tuple[float, float]] = {}


WRIST_SCAN_MOVES = (
    ("wrist home", None),
    ("wrist left", (-0.08, 0.0, 0.04)),
    ("wrist right", (0.16, 0.0, 0.0)),
)


def scan_workspace(
    twin: Twin, router, log_dir: Path, echo: Callable[[str], None], wrist: bool = True
) -> tuple[dict, VisionState | None]:
    """Scan the workspace with the fixed workspace camera and (optionally) a wrist-camera sweep.

    Every view is sent to the vision model; boxes are turned into viewing rays with that view's
    camera model (fixed extrinsics for the workspace camera, the wrist camera's pose at capture
    time for the wrist views), fused across views (triangulated when seen from several
    viewpoints) and, in simulation, scored against the true object poses.
    """
    import cv2
    import numpy as np
    from twin_models import Image
    from twin_perception import (
        DETECT_SCHEMA,
        DETECT_SYSTEM,
        annotate,
        fuse_views,
        fused_as_detections,
        match_to_truth,
        parse_detections,
        report_lines,
        truth_pixels,
        unique_labels,
        validate_detections,
    )

    views = twin.scan_views(wrist, echo)
    if not views:
        echo("PERCEPTION_UNAVAILABLE: no camera images.")
        return {"event": "scan", "result": "no_camera"}, None
    record = {"event": "scan", "views": [v[0] for v in views], "calls": []}
    all_dets, panels = [], []
    xyz = twin.objects_xyz()
    for name, image, model in views:
        _ok, jpeg = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        result = router.call(
            DETECT_SYSTEM,
            f"Detect the objects on the table in this {name} camera image.",
            DETECT_SCHEMA,
            validate=validate_detections,
            images=(Image(jpeg.tobytes()),),
        )
        record["calls"].append(
            {"view": name, "model": result.model, "seconds": round(result.seconds, 2)}
        )
        if result.payload is None:
            echo(f"  {name}: vision model failed ({result.error})")
            continue
        dets = parse_detections(result.payload, model, twin.table_top, name)
        all_dets += dets
        echo(f"  {name}: {len(dets)} object(s) ({result.model}, {result.seconds:.1f} s)")
        panel = annotate(image, dets, truth_pixels(model, xyz))
        panels.append(cv2.resize(panel, (int(panel.shape[1] * 240 / panel.shape[0]), 240)))
    fused = fuse_views(all_dets, twin.table_top)
    truth = {
        o.name: (tuple(xyz[o.name][:2]), o.color_name, o.shape, o.keywords)
        for o in twin.level.objects
        if ts.on_table(xyz[o.name], twin.table_top)
    }
    report = match_to_truth(fused_as_detections(fused), truth)
    for line in report_lines(report):
        echo(line)
    triangulated = sum(1 for f in fused if f.z is not None)
    echo(
        f"  fused {len(fused)} object(s) from {len(views)} view(s); {triangulated} triangulated from 2+ viewpoints"
    )
    if panels:
        picture = log_dir / f"twin_scan_{utc_stamp()}.jpg"
        cv2.imwrite(str(picture), np.hstack(panels))
        echo(f"  annotated views: {picture}")
        record["image"] = str(picture)
    vision = VisionState()
    vision.names = unique_labels(fused_as_detections(fused))
    matched = {m.detection.label + str(m.detection.xy_est): m.name for m in report.matches}
    rows = ["Objects (from the camera scan; positions are estimates):"]
    for name, f in zip(vision.names, fused, strict=True):
        vision.to_truth[name] = matched.get(f.label + str(f.xy))
        where = ts.location_of(twin.level, (*f.xy, twin.table_top + 0.02), twin.table_top)
        place = "on the table" if where == "table" else f"in the {where}"
        height = (
            f", centre {100 * (f.z - twin.table_top):.0f} cm above the table"
            if f.z is not None
            else ""
        )
        kind = " ".join(v for v in (f.color, f.shape) if v) or "object"
        rows.append(
            f"- {name}: {kind}; at x={f.xy[0]:+.2f}, y={f.xy[1]:+.2f}{height}, {place} (seen by {', '.join(f.views)})"
        )
    vision.rows = rows
    vision.positions = {name: f.xy for name, f in zip(vision.names, fused, strict=True)}
    record.update(
        result="ok",
        summary=report.summary(),
        fused=[
            {
                "name": n,
                "label": f.label,
                "xy": f.xy,
                "z": f.z,
                "views": f.views,
                "truth": vision.to_truth[n],
            }
            for n, f in zip(vision.names, fused, strict=True)
        ],
    )
    return record, vision


def camera_check(
    twin: Twin,
    reference,
    watched: dict[str, tuple[float, float, float]],
    ignore_points: list[tuple[float, float, float]] | None = None,
    threshold: float | None = None,
    unstable=None,
) -> dict | None:
    """Compare the workspace camera with the reference image (arm at home).

    Only the work area (table region, not the robot) is watched; pixels of objects the plan
    moved on purpose (``ignore_points``) and pixels that flicker between two reference frames
    (``unstable``) are masked. ``watched`` objects (where the workspace
    model believes they are) are checked one by one, so a moved or vanished object is named.
    """
    import numpy as np
    from twin_perception import (
        OBJECT_CHANGE,
        change_map,
        change_threshold,
        changed_fraction,
        circle_mask,
        object_circles,
        region_changes,
        workspace_roi,
    )

    current = twin.workspace_rgb()
    if reference is None or current is None:
        return None
    limit = change_threshold(None) if threshold is None else threshold
    changed = change_map(reference, current)
    if changed is None:
        return {"fraction": 1.0, "threshold": limit, "objects": {}, "flagged": [], "changed": True}
    shape = changed.shape
    roi = workspace_roi(twin.camera_spec, shape, twin.table_top)
    mask = None if unstable is None else np.asarray(unstable, bool).copy()
    if ignore_points:
        planned = circle_mask(shape, object_circles(twin.camera_spec, ignore_points, 0.07))
        mask = planned if mask is None else (mask | planned)
    fraction = changed_fraction(None, None, mask, roi=roi, changed=changed)
    circles = {}
    for name, xyz in watched.items():
        found = object_circles(twin.camera_spec, [xyz], 0.035)
        if found:
            circles[name] = found[0]
    per_object = region_changes(None, current, circles, changed=changed, mask=mask)
    flagged = sorted(n for n, v in per_object.items() if v > OBJECT_CHANGE)
    return {
        "fraction": round(fraction, 5),
        "threshold": round(limit, 5),
        "objects": {n: round(v, 3) for n, v in per_object.items()},
        "flagged": flagged,
        "changed": fraction > limit or bool(flagged),
    }


def describe_camera_change(check: dict) -> list[str]:
    out = [
        f"camera: the image around {n} changed ({check['objects'][n] * 100:.0f}% of it): moved or gone"
        for n in check["flagged"]
    ]
    if not out and check["changed"]:
        out.append(
            f"camera: {check['fraction'] * 100:.2f}% of the work area changed outside the planned objects"
        )
    return out


MAX_REPLANS = 6  # rounds of rescan + replan per command
MAX_OBJECT_ATTEMPTS = 3  # failed motions per object before it is given up


@dataclass
class CommandProgress:
    """What has happened so far while working on one operator command (all rounds).

    Recovery policy: a failed motion is never repeated as it was. The workspace is
    rescanned, the planner makes a new plan from the CURRENT scene, and the plan is validated
    again; an object may appear in such a new plan until it has failed MAX_OBJECT_ATTEMPTS
    times. An object stuck in the gripper is never planned again in this command.
    """

    done: list[tuple[str, str]] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # last failure of an unfinished object
    attempts: dict[str, int] = field(default_factory=dict)  # failed motions per object
    hard: dict[str, str] = field(default_factory=dict)  # never planned again (stuck)
    recovered: list[str] = field(default_factory=list)  # failed once, finished later
    displaced: dict[str, str] = field(default_factory=dict)  # placed, later knocked out
    blocked: dict[str, str] = field(default_factory=dict)  # refused before any motion
    changes: list[str] = field(default_factory=list)  # unexpected workspace changes
    pending: list[tuple[str, str]] = field(default_factory=list)  # planned, never reached

    def absorb(self, record: dict) -> set[str]:
        """Add one round's outcomes; return the objects the robot actually handled."""
        handled = set()
        plan = [tuple(p) for p in record.get("plan", [])]
        self.pending = plan[len(record.get("outcomes", [])) :]
        for (obj, dest), out in zip(
            record.get("plan", []), record.get("outcomes", []), strict=False
        ):  # a stopped plan has fewer outcomes than steps
            if out["moved"]:
                handled.add(obj)
            if out["ok"]:
                if not out["skipped"]:
                    self.done.append((obj, dest))
                self.blocked.pop(obj, None)
                self.displaced.pop(obj, None)
                if self.failed.pop(obj, None) is not None:
                    self.recovered.append(obj)
            elif out["moved"]:
                self.failed[obj] = out["summary"]
                self.attempts[obj] = self.attempts.get(obj, 0) + 1
                self.blocked.pop(obj, None)
                if STUCK_TEXT in out["summary"]:
                    self.hard[obj] = out["summary"]
            else:
                self.blocked[obj] = out["summary"]
        return handled

    def excluded(self) -> set[str]:
        """Objects the planner must not use again in this command."""
        spent = {o for o, n in self.attempts.items() if n >= MAX_OBJECT_ATTEMPTS}
        return set(self.hard) | (spent & set(self.failed))

    def check_placed(self, placed: Callable[[str, str], bool] | None) -> list[str]:
        """Objects already placed (final destination, not a temporary table spot) that are no
        longer there; they become work again."""
        if placed is None:
            return []
        latest: dict[str, str] = {}
        for obj, dest in self.done:
            latest[obj] = dest
        lost = []
        for obj, dest in latest.items():
            if dest == "table" or obj in self.displaced:
                continue
            if not placed(obj, dest):
                self.displaced[obj] = dest
                lost.append(f"{obj} was knocked out of the {dest}")
        return lost

    def outstanding(self) -> bool:
        return bool(self.failed or self.blocked or self.displaced)

    def overall(self, last_result: str, rounds: int) -> str:
        if rounds == 1 and last_result in ("refused", "model_error"):
            return last_result
        if not self.outstanding() and last_result in ("done", "nothing_left"):
            return "done"
        return "partial" if self.done else "failed"


def build_replan_prompt(
    command: str,
    names: NameTable,
    listing: list[str],
    progress: CommandProgress,
    display: dict[str, str] | None = None,
) -> str:
    """Planner prompt for the remaining work, from the rescanned (current) scene."""
    show = (display or {}).get
    excluded = progress.excluded()
    lines = [
        build_prompt(command, names, listing),
        "",
        "REPLANNING: the robot already worked on this command and the scene above was rebuilt "
        "by a fresh scan afterwards, so it is the CURRENT state. What happened:",
    ]
    lines += [f"- done: {show(o, o)} -> {d}" for o, d in progress.done]
    for o, why in progress.failed.items():
        if o in excluded:
            lines.append(f"- GIVE UP, do not plan {show(o, o)} again: {why}")
        else:
            n = progress.attempts.get(o, 1)
            lines.append(
                f"- failed attempt {n}/{MAX_OBJECT_ATTEMPTS} for {show(o, o)}: {why}. It may be "
                "planned again from its CURRENT position (shown above) if the command still needs it"
            )
    lines += [
        f"- knocked out of place: {show(o, o)} had been put in the {d} but is no longer there; "
        "put it back"
        for o, d in progress.displaced.items()
    ]
    lines += [f"- not started ({show(o, o)}): {why}" for o, why in progress.blocked.items()]
    lines += [
        f"- not executed yet: {show(o, o)} -> {d} (the plan stopped before this step)"
        for o, d in progress.pending
    ]
    lines += [f"- unexpected change: {c}" for c in progress.changes]
    lines.append(
        "Plan only the steps still needed to fulfil the operator command from the CURRENT "
        "scene, including objects knocked over or pushed out of place by a collision (an object "
        "lying on its side can still be picked unless it is marked NOT GRASPABLE NOW). An "
        "object marked done that is still in its destination counts as done even if it now "
        "leans or lies: do not take it out again unless that is the only way to finish. Never "
        "include an object marked GIVE UP. An object marked NOT GRASPABLE NOW because of a "
        "neighbour can be picked after its neighbour is moved away (to the table) if that "
        "serves the command. Only if nothing useful is left or the rest is impossible, set "
        "decision to 'refuse' and give the reason."
    )
    return "\n".join(lines)


def adaptive_command(
    data: dict,
    echo: Callable[[str], None],
    *,
    objects_xyz: Callable[[], dict[str, list[float]]],
    run_round: Callable[[dict, set[str], int], dict],
    observe: Callable[[dict[str, list[float]], set[str]], list[str]],
    rescan: Callable[[str], tuple],
    planner: Planner | None,
    log: Callable[[dict], None],
    max_replans: int = MAX_REPLANS,
    placed: Callable[[str, str], bool] | None = None,
) -> dict:
    """Execute a command and keep working on it until it is done.

    Each round: run the plan -> look for changes the plan did not cause (simulator check +
    camera) and for placed objects that were knocked out again -> if anything is unfinished,
    rebuild the workspace model (rescan) and ask the planner for the remaining work. Failed
    objects come back only through such a new plan (see CommandProgress). The loop ends when
    the command is done, the planner says the rest is impossible, every unfinished object is
    excluded, two rounds in a row change nothing, or after ``max_replans`` replans.
    """
    command = data["input"]
    progress = CommandProgress()
    rounds: list[dict] = []
    replans = 0
    idle_rounds = 0
    while True:
        before = objects_xyz()
        record = run_round(data, progress.excluded(), replans)
        handled = progress.absorb(record)
        changes = observe(before, handled) if record.get("plan") else []
        changes += progress.check_placed(placed)
        if changes:
            echo("UNEXPECTED WORKSPACE CHANGE: " + "; ".join(changes))
        record["unexpected_changes"] = changes
        progress.changes += changes
        log(record)
        rounds.append(record)
        idle_rounds = 0 if handled or changes else idle_rounds + 1
        if record["result"] in ("refused", "model_error", "nothing_left"):
            break
        open_work = (set(progress.failed) | set(progress.blocked) | set(progress.displaced)) - (
            progress.excluded()
        )
        if record["result"] == "done" and not changes and not open_work:
            break
        if planner is None:
            echo("No planner available: cannot replan.")
            break
        if idle_rounds >= 2:
            echo("STOPPED: two rounds in a row changed nothing (every remaining step was refused).")
            break
        if replans >= max_replans:
            echo(f"STOPPED: replanning limit reached ({max_replans} rounds).")
            break
        replans += 1
        reason = "; ".join(changes) if changes else "a step did not finish"
        echo(f"RESCAN ({reason}): rebuilding the workspace model")
        names, listing, scene, display = rescan(reason)
        echo("\n".join(listing))
        echo(f"REPLAN {replans}/{max_replans}: asking the planner for the remaining work...")
        result = planner.plan(
            build_replan_prompt(command, names, listing, progress, display),
            names,
            scene,
            escalate=False,
        )
        log(
            {
                "event": "replan",
                "round": replans,
                "reason": reason,
                "model": result.model,
                "payload": result.payload,
            }
        )
        data = {"input": command, "result": result, "names": names}
    overall = progress.overall(rounds[-1]["result"], len(rounds))
    if replans:
        echo(
            f"COMMAND RESULT after {replans} replan(s): {overall.upper()} | done: "
            + (", ".join(f"{o} -> {d}" for o, d in progress.done) or "nothing")
            + (f" | recovered: {', '.join(progress.recovered)}" if progress.recovered else "")
            + (f" | failed: {', '.join(progress.failed)}" if progress.failed else "")
            + (f" | knocked out: {', '.join(progress.displaced)}" if progress.displaced else "")
            + (f" | blocked: {', '.join(progress.blocked)}" if progress.blocked else "")
        )
    summary = {
        "event": "command_summary",
        "input": command,
        "rounds": len(rounds),
        "replans": replans,
        "result": overall,
        "done": [list(p) for p in progress.done],
        "failed": progress.failed,
        "attempts": progress.attempts,
        "recovered": progress.recovered,
        "displaced": progress.displaced,
        "blocked": progress.blocked,
        "changes": progress.changes,
    }
    log(summary)
    return summary


def submit(inbox: queue.Queue, twin_ref: dict, kind: str, data: dict | None) -> None:
    """Hand one job to the render/physics thread and wait until it has finished."""
    twin_ref["busy"].clear()
    inbox.put((kind, data))
    twin_ref["busy"].wait()


def parse_level(text: str) -> int | None:
    parts = text.split()
    if len(parts) == 2 and parts[0] == "level" and parts[1].isdigit():
        number = int(parts[1])
        return number if number in ts.LEVELS else None
    return None


def input_worker(
    inbox: queue.Queue, planner: Planner | None, twin_ref: dict, stop: threading.Event
) -> None:
    while not stop.is_set():
        try:
            line = input("\ncommand> ").strip()
        except (EOFError, KeyboardInterrupt):
            inbox.put(("quit", None))
            return
        low = line.casefold()
        if not line:
            continue
        if low in {"quit", "exit", "q"}:
            inbox.put(("quit", None))
            return
        if low in {"help", "?"}:
            print(help_text(twin_ref["level"]))
            continue
        if low in {"scene", "reset", "models", "perceive", "scan"}:
            submit(inbox, twin_ref, low, None)
            continue
        if low in {"source sim", "source vlm"}:
            submit(inbox, twin_ref, "source", {"source": low.split()[1]})
            continue
        if low.startswith("level"):
            number = parse_level(low)
            if number is None:
                print(f"Unknown level. Use one of: {', '.join(map(str, ts.LEVELS))}.")
            else:
                submit(inbox, twin_ref, "level", {"level": number})
            continue
        if planner is None:
            print("REFUSED: GEMINI_API_KEY is not set, free-text commands are unavailable.")
            continue
        names: NameTable = twin_ref["names"]
        prompt = build_prompt(line, names, twin_ref["listing"])
        print("  asking the planner...", flush=True)
        result = planner.plan(prompt, names, twin_ref.get("scene"))
        submit(inbox, twin_ref, "command", {"input": line, "result": result, "names": names})


def handle_command(
    twin: Twin,
    data: dict,
    echo: Callable[[str], None],
    vision: VisionState | None = None,
    do_step=None,
    exclude: set[str] | None = None,
    round_index: int = 0,
) -> dict:
    """Run one planning round. ``exclude``: objects whose motion already failed in this
    command; steps that touch them are dropped (given up: stuck in the gripper or
    MAX_OBJECT_ATTEMPTS failed attempts)."""
    result = data["result"]
    names: NameTable = data.get("names") or NameTable.from_level(twin.level)
    record = {
        "event": "command",
        "input": data["input"],
        "model": result.model,
        "payload": result.payload,
        "attempts": [a.__dict__ for a in result.attempts],
        "level": twin.level.number,
        "source": "vlm" if vision is not None else "sim",
        "round": round_index,
    }
    echo(f"  planner: {result.model or 'none'} ({result.seconds:.1f} s)")
    notable = [a for a in result.attempts if a.outcome != "ok" or a.detail]
    if notable and result.payload is not None:
        echo(
            "  router: "
            + "; ".join(
                f"{a.model} {a.outcome}{' (' + a.detail + ')' if a.detail else ''}" for a in notable
            )
        )
    record["seconds"] = round(result.seconds, 2)
    record["escalated"] = result.escalated
    if result.payload is None:
        echo(f"FAILED: no model produced a usable answer: {result.error}")
        record["result"] = "model_error"
        return record
    resolved = resolve_plan(result.payload, names)
    if isinstance(resolved, str):
        if round_index:
            echo(f"NO FURTHER STEPS ({result.model}): {resolved}")
            record["result"] = "nothing_left"
        else:
            echo(f"REFUSED ({result.model}): {resolved} The robot does not move.")
            record["result"] = "refused"
        return record
    if vision is not None:
        mapped = []
        for obj, dest in resolved:
            truth = vision.to_truth.get(obj)
            if truth is None:
                echo(
                    f"NOT DONE: '{obj}' was detected but could not be located in the scene; the robot does not move."
                )
                record["result"] = "failed"
                return record
            mapped.append((truth, dest))
        echo(
            "  vision -> scene: "
            + "; ".join(f"{o} = {t}" for (o, _), (t, _) in zip(resolved, mapped, strict=True))
        )
        resolved = mapped
    if exclude:
        dropped = [o for o, _ in resolved if o in exclude]
        resolved = [(o, d) for o, d in resolved if o not in exclude]
        if dropped:
            echo(
                "  dropped from the plan (given up in this command: stuck in the gripper or "
                f"{MAX_OBJECT_ATTEMPTS} failed attempts): " + ", ".join(dict.fromkeys(dropped))
            )
            record["dropped"] = dropped
        if not resolved:
            echo("NO FURTHER STEPS: the new plan only touched objects that were given up.")
            record["result"] = "nothing_left"
            return record
    steps = "; ".join(f"{i}. {o} -> {d}" for i, (o, d) in enumerate(resolved, start=1))
    echo(f"PLAN ({result.model}, {len(resolved)} step(s)): {steps}")
    note = str(result.payload.get("reason") or "").strip()
    if note:
        echo(f"  planner note: {note}")
    outcomes = run_plan(resolved, do_step or twin.pick_and_place, echo)
    done = len(outcomes) == len(resolved) and all(o.ok for o in outcomes)
    if done:
        echo(f"DONE: all {len(resolved)} step(s) completed.")
    else:
        bad = [o for o in outcomes if not o.ok]
        echo(
            f"NOT DONE ({sum(o.ok for o in outcomes)}/{len(resolved)} step(s) ok; rescan and replan follow): "
            + " | ".join(o.summary for o in bad)
        )
    record.update(
        result="done" if done else "failed",
        plan=[list(p) for p in resolved],
        outcomes=[
            {
                "ok": o.ok,
                "skipped": o.skipped,
                "moved": o.moved,
                "summary": o.summary,
                "motions": [s.__dict__ for s in o.steps],
            }
            for o in outcomes
        ],
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path.home() / "worktrees/Panthera-HT_RoboTwin/RoboTwin",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("VRS_LOG_DIR", str(Path.home() / "twin_logs"))),
        help="directory for twin_session_*.jsonl logs",
    )
    parser.add_argument("--level", type=int, choices=sorted(ts.LEVELS), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-freq", type=int, default=10)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--camera-hz", type=float, default=5.0, help="camera window refresh rate (0 = off)"
    )
    parser.add_argument("--place-constrain", choices=PLACE_CONSTRAINS, default="free")
    parser.add_argument("--executor", choices=("scripted", "policy"), default="scripted")
    parser.add_argument(
        "--policy-server", default="127.0.0.1:5555", help="policy_bridge.py serve address"
    )
    args = parser.parse_args(argv)
    if not os.environ.get("DISPLAY"):
        print("NO_DISPLAY: run inside a graphical session.")
        return 3
    from twin_models import router_from_env

    try:
        router = router_from_env()
        planner: Planner | None = Planner(router)
    except RuntimeError as exc:
        print(f"WARNING: {exc} The twin opens, but free-text commands are disabled.")
        router, planner = None, None
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    session = log_dir / f"twin_session_{utc_stamp()}.jsonl"

    def log(record: dict) -> None:
        record["utc"] = utc_stamp()
        with session.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    from twin_cameras import CameraStream

    twin = Twin(
        args.robotwin_root.resolve(), args.seed, args.render_freq, args.place_constrain, args.level
    )
    do_step = twin.pick_and_place
    if args.executor == "policy":
        from policy_bridge import PolicyClient

        try:
            client = PolicyClient(args.policy_server)
        except OSError as exc:
            print(f"POLICY_SERVER_UNREACHABLE at {args.policy_server}: {exc}")
            return 6

        def do_step(obj, dest, echo):
            return twin.pick_and_place_policy(client, obj, dest, echo)

    cameras = CameraStream(None, hz=args.camera_hz)
    vision: VisionState | None = None

    def build(event: str) -> None:
        monitor["warm"] = False
        print(f"Building level {twin.level_number} (seed {twin.seed}, about 15 s)...", flush=True)
        twin.setup()
        cameras.attach(twin.task)
        cameras.extra_sources = (
            {"workspace (config) camera": twin.workspace_rgb}  # fixed camera; head camera not shown
            if twin.workspace_camera is not None
            else {}
        )
        log(
            {
                "event": event,
                "task": TASK,
                "level": twin.level_number,
                "seed": twin.seed,
                "twin_version": TWIN_VERSION,
                "planner_world": twin.planner_status,
                "planner_world_sample": twin.world_debug,
                "finger_axis": twin.finger_axis,
                "closed_loop_place": twin.closed_loop,
                "place_constrain": args.place_constrain,
                "objects": twin.objects_xyz(),
                "containers": [c.__dict__ for c in twin.level.containers],
                "model_chain": list(router.chain) if router else [],
                "model_escalation": list(router.escalation) if router else [],
            }
        )
        print(
            f"Level {twin.level.number}: {twin.level.title} | executor: {args.executor} | "
            + "place: safe-height transit, alignment in the air, straight descent"
        )
        print(
            f"Motion planner world: {twin.planner_status} | finger axis: "
            + (twin.finger_axis or "unknown (both finger directions must be free)")
        )
        print("\n".join(twin.listing()))

    monitor: dict = {
        "reference": None,
        "threshold": change_threshold(None),
        "warm": False,
        "unstable": None,
    }

    def capture_reference() -> None:
        """Reference image of the work area (arm at home) + its frame-to-frame noise level."""
        if not monitor["warm"]:
            for _ in range(3):  # first frames after a build can still be loading textures
                twin.workspace_rgb()
            monitor["warm"] = True
        first = twin.workspace_rgb()
        second = twin.workspace_rgb()
        monitor["reference"] = first
        if first is None or second is None:
            return
        roi = workspace_roi(twin.camera_spec, first.shape[:2], twin.table_top)
        flicker = change_map(first, second)
        monitor["unstable"] = None if flicker is None else dilate(flicker, 2)
        noise = changed_fraction(first, second, monitor["unstable"], roi=roi)
        monitor["threshold"] = change_threshold(noise)
        monitor["noise"] = noise

    def believed_positions() -> dict[str, tuple[float, float, float]]:
        """Where the current workspace model says the objects are (truth or camera scan)."""
        if vision is None:
            return {n: tuple(xyz) for n, xyz in twin.objects_xyz().items()}
        z = twin.table_top + 0.03
        return {
            truth: (xy[0], xy[1], z)
            for name, xy in vision.positions.items()
            if (truth := vision.to_truth.get(name))
        }

    def refresh(ref: dict) -> None:
        ref["level"] = twin.level
        capture_reference()  # the workspace model and its reference image move together
        if vision is None:
            ref["names"] = NameTable.from_level(twin.level)
            ref["listing"] = twin.listing()
            ref["scene"] = (twin.level, twin.objects_xyz(), twin.table_top)
        else:
            ref["names"] = NameTable.from_level(twin.level, vision.names)
            ref["listing"] = vision.rows + destination_rows(
                twin.level, twin.objects_xyz(), twin.table_top
            )
            ref["scene"] = None

    def rescan(reason: str) -> tuple:
        """Rebuild the workspace model: camera scan (source vlm) or simulator list (source sim)."""
        nonlocal vision
        if vision is not None and router is not None:
            record, new_vision = scan_workspace(twin, router, log_dir, print, wrist=True)
            log(record)
            vision = new_vision or vision
        refresh(twin_ref)
        log(
            {
                "event": "rescan",
                "reason": reason,
                "source": "vlm" if vision is not None else "sim",
                "listing": twin_ref["listing"],
                "change_threshold": monitor["threshold"],
            }
        )
        display = {t: n for n, t in vision.to_truth.items() if t} if vision is not None else None
        return twin_ref["names"], twin_ref["listing"], twin_ref.get("scene"), display

    build("setup")
    print(f"Twin {TWIN_VERSION} ready | log: {session}")
    if router:
        print(
            "Model chain: "
            + " -> ".join(router.chain)
            + " | escalation: "
            + (", ".join(router.escalation) or "none")
        )
    print(help_text(twin.level))

    inbox: queue.Queue = queue.Queue()
    busy = threading.Event()
    busy.set()
    twin_ref: dict = {"busy": busy}
    refresh(twin_ref)
    stop = threading.Event()
    threading.Thread(
        target=input_worker, args=(inbox, planner, twin_ref, stop), daemon=True
    ).start()
    period = 1.0 / max(args.fps, 1.0)
    try:
        while not getattr(twin.task.viewer, "closed", False):
            try:
                kind, data = inbox.get_nowait()
            except queue.Empty:
                twin.task.viewer.render()
                cameras.tick()
                time.sleep(period)
                continue
            try:
                if kind == "quit":
                    break
                if kind == "scene":
                    print("\n".join(twin_ref["listing"]))
                elif kind == "models":
                    print(
                        "\n".join(router.summary())
                        if router
                        else "No model router (GEMINI_API_KEY missing)."
                    )
                elif kind in ("reset", "level"):
                    if kind == "reset":
                        twin.seed += 1
                    else:
                        twin.level_number = data["level"]
                    vision = None
                    build(kind)
                elif kind in ("perceive", "scan"):
                    if router is None:
                        print("PERCEPTION_UNAVAILABLE: GEMINI_API_KEY missing.")
                    else:
                        record, new_vision = scan_workspace(
                            twin, router, log_dir, print, wrist=kind == "scan"
                        )
                        log(record)
                        if vision is not None and new_vision is not None:
                            vision = new_vision
                elif kind == "source":
                    if data["source"] == "sim":
                        vision = None
                        print("Planning from the simulator object list.")
                    elif router is None:
                        print("PERCEPTION_UNAVAILABLE: GEMINI_API_KEY missing.")
                    else:
                        record, new_vision = scan_workspace(
                            twin, router, log_dir, print, wrist=True
                        )
                        log(record)
                        if new_vision is not None:
                            vision = new_vision
                            print("Planning from the camera scan (scene list below).")
                            print("\n".join(vision.rows))
                elif kind == "command":
                    # 1) Pre-action check: has the workspace changed since the last scan?
                    check = camera_check(
                        twin,
                        monitor["reference"],
                        believed_positions(),
                        None,
                        monitor["threshold"],
                        monitor["unstable"],
                    )
                    if check is not None and check["changed"] and planner is not None:
                        what = "; ".join(describe_camera_change(check))
                        print(
                            f"WORKSPACE CHANGED since the last scan ({what}); rescanning and re-planning"
                        )
                        names, listing, scene, _display = rescan("changed before the command")
                        data["names"] = names
                        data["result"] = planner.plan(
                            build_prompt(data["input"], names, listing), names, scene
                        )
                        log({"event": "rescan_before_command", "check": check})

                    # 2) Execute with adaptive replanning after failures / unexpected changes.
                    def run_round(round_data, exclude, round_index):
                        monitor["q_before"] = twin.orientations()
                        record = handle_command(
                            # vision is read at call time on purpose: a rescan may replace it
                            twin,
                            round_data,
                            print,
                            vision,  # noqa: B023
                            do_step,
                            exclude,
                            round_index,
                        )
                        record["executor"] = args.executor
                        record["objects_after"] = twin.objects_xyz()
                        return record

                    def observe(before, handled):
                        after = twin.objects_xyz()
                        changes = [
                            f"{n} moved {d * 100:.1f} cm (now x={after[n][0]:+.2f}, y={after[n][1]:+.2f})"
                            for n, d in ts.unexpected_moves(before, after, handled).items()
                        ]
                        tipped = tg.unexpected_tilts(
                            monitor.get("q_before", {}), twin.orientations(), handled
                        )
                        changes += [f"{n} tipped {a:.0f} deg" for n, a in tipped.items()]
                        ignore = [tuple(before[o]) for o in handled if o in before] + [
                            tuple(after[o]) for o in handled if o in after
                        ]
                        watched = {
                            n: xyz
                            for n, xyz in believed_positions().items()
                            if n not in handled and n in before
                        }
                        check = camera_check(
                            twin,
                            monitor["reference"],
                            watched,
                            ignore,
                            monitor["threshold"],
                            monitor["unstable"],
                        )
                        if check is not None:
                            log({"event": "camera_check", "check": check})
                            if check["changed"] and not changes:
                                changes += describe_camera_change(check)
                            elif check["flagged"]:
                                print("  camera agrees: " + ", ".join(check["flagged"]))
                        return changes

                    adaptive_command(
                        data,
                        print,
                        objects_xyz=twin.objects_xyz,
                        run_round=run_round,
                        observe=observe,
                        rescan=rescan,
                        planner=planner,
                        log=log,
                        placed=lambda o, d: ts.in_destination(
                            twin.level, twin.level.container(d), twin.object_xyz(o), twin.table_top
                        ),
                    )
                refresh(twin_ref)
            finally:
                busy.set()
    except KeyboardInterrupt:
        pass
    stop.set()
    cameras.close()
    if router:
        log({"event": "end", "models": router.summary()})
    print(f"Session ended. Log: {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
