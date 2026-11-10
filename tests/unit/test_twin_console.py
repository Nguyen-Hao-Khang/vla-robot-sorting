# ruff: noqa: E501
"""Digital twin console: plan resolution, aliases, executor prechecks, step/plan runners (no GPU)."""

from __future__ import annotations

import importlib.util
import queue
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/twin"))
spec = importlib.util.spec_from_file_location(
    "twin_console", ROOT / "tools/twin/twin_console.py"
)
twin = importlib.util.module_from_spec(spec)
sys.modules["twin_console"] = twin
spec.loader.exec_module(twin)
ts = twin.ts
models = importlib.import_module("twin_models")

TABLE = 0.74


def _xyz(level):
    return {
        name: [x, y, TABLE + level.object(name).half_height]
        for name, (x, y) in level.spawn_xy().items()
    }


def test_listing_describes_objects_locations_room_and_aliases() -> None:
    level = ts.LEVELS[5]
    xyz = _xyz(level)
    xyz["small red cylinder"] = [ts.DEEP_BIN.xy[0] - 0.04, ts.DEEP_BIN.xy[1], TABLE + 0.034]
    xyz["large red cylinder"] = [ts.DEEP_BIN.xy[0] + 0.04, ts.DEEP_BIN.xy[1], TABLE + 0.034]
    rows = twin.scene_listing(level, xyz, TABLE)
    small = next(r for r in rows if r.startswith("- small red cylinder"))
    assert "3.0 cm diameter" in small and "in the deep bin" in small and "CANNOT" not in small
    deep = next(r for r in rows if r.startswith("- deep bin"))
    assert "holds small red cylinder, large red cylinder" in deep and "free room 0" in deep
    assert "khay sâu" in deep
    tray = next(r for r in rows if r.startswith("- wide tray"))
    assert "empty" in tray and "free room 2" in tray


def test_listing_flags_objects_that_cannot_leave_the_bin() -> None:
    level = ts.LEVELS[7]
    xyz = _xyz(level)
    xyz["tennis ball"] = [*ts.DEEP_BIN.xy, TABLE + 0.028]
    row = next(r for r in twin.scene_listing(level, xyz, TABLE) if r.startswith("- tennis ball"))
    assert "CANNOT be picked" in row and "tennis ball with a white seam" in row


def test_prompt_lists_exact_names_and_ends_with_command() -> None:
    level = ts.LEVELS[3]
    names = twin.NameTable.from_level(level)
    prompt = twin.build_prompt(
        "bỏ vật tròn vào khay sâu", names, twin.scene_listing(level, _xyz(level), TABLE)
    )
    assert "Exact object names: cube, cylinder, triangular prism" in prompt
    assert "Exact destination names: wide tray, deep bin, table" in prompt
    assert prompt.endswith("bỏ vật tròn vào khay sâu")


def test_system_prompt_rules() -> None:
    for word in ("English", str(twin.MAX_STEPS), "staging pad", "also called", "free room"):
        assert word in twin.SYSTEM


def test_resolve_plan_accepts_aliases_and_normalises() -> None:
    level = ts.LEVELS[4]
    plan = twin.resolve_plan(
        {
            "decision": "execute",
            "steps": [
                {"object": "Small Cylinder", "destination": "khay sâu"},
                {"object": "large cylinder", "destination": "TALL BIN"},
                {"object": "medium cylinder", "destination": "mặt bàn"},
            ],
        },
        level,
    )
    assert plan == [
        ("small cylinder", "deep bin"),
        ("large cylinder", "deep bin"),
        ("medium cylinder", "table"),
    ]
    staging = twin.resolve_plan(
        {"decision": "execute", "steps": [{"object": "red cube", "destination": "staging area"}]},
        ts.LEVELS[2],
    )
    assert staging == [("red cube", "staging pad")]


def test_resolve_refusals_never_execute() -> None:
    level = ts.LEVELS[2]
    assert (
        twin.resolve_plan({"decision": "clarify", "reason": "Which cube?"}, level) == "Which cube?"
    )
    assert "empty plan" in twin.resolve_plan({"decision": "execute", "steps": []}, level)
    assert "Step 1: object" in twin.resolve_plan(
        {"decision": "execute", "steps": [{"object": "red cup", "destination": "white tray"}]},
        level,
    )
    assert "Step 2: destination" in twin.resolve_plan(
        {
            "decision": "execute",
            "steps": [
                {"object": "red cube", "destination": "white tray"},
                {"object": "blue cube", "destination": "deep bin"},
            ],
        },
        level,
    )
    too_long = {
        "decision": "execute",
        "steps": [{"object": "red cube", "destination": "white tray"}] * 7,
    }
    assert "limit" in twin.resolve_plan(too_long, level)
    assert "malformed" in twin.resolve_plan({"decision": "execute", "steps": ["x"]}, level)


def test_plan_problem_is_router_validator() -> None:
    names = twin.NameTable.from_level(ts.LEVELS[2])
    assert twin.plan_problem({"decision": "refuse", "reason": "no"}, names) is None
    assert (
        twin.plan_problem(
            {"decision": "execute", "steps": [{"object": "red cube", "destination": "white tray"}]},
            names,
        )
        is None
    )
    assert "not in the scene" in twin.plan_problem(
        {"decision": "execute", "steps": [{"object": "x", "destination": "white tray"}]}, names
    )
    assert "unknown decision" in twin.plan_problem({"decision": "maybe"}, names)


def test_name_table_from_detections() -> None:
    names = twin.NameTable.from_level(ts.LEVELS[3], ["red cube", "red cube #2"])
    assert names.object_names == ["red cube", "red cube #2"]
    assert names.places["thùng cao"] == "deep bin"


def test_precheck_blocks_full_destination_and_unreachable_objects() -> None:
    level = ts.LEVELS[3]
    xyz = _xyz(level)
    xyz["cube"] = [ts.DEEP_BIN.xy[0] - 0.04, ts.DEEP_BIN.xy[1], TABLE + 0.029]
    xyz["triangular prism"] = [ts.DEEP_BIN.xy[0] + 0.04, ts.DEEP_BIN.xy[1], TABLE + 0.034]
    full = twin.precheck(level, "cylinder", "deep bin", xyz, TABLE)
    assert full is not None and not full.ok and not full.moved and "full" in full.summary
    same = twin.precheck(level, "cube", "deep bin", xyz, TABLE)
    assert same.ok and same.skipped
    assert twin.precheck(level, "cube", "wide tray", xyz, TABLE) is None  # 5 cm cube can leave
    l7 = ts.LEVELS[7]
    xyz7 = _xyz(l7)
    xyz7["volleyball"] = [*ts.DEEP_BIN.xy, TABLE + 0.028]
    stuck = twin.precheck(l7, "volleyball", "wide tray", xyz7, TABLE)
    assert stuck is not None and "cannot be picked" in stuck.summary


def test_help_lists_all_levels() -> None:
    text = twin.help_text(ts.LEVELS[1])
    assert all(f"level {n}:" in text for n in ts.LEVELS) and "perceive" in text


def test_parse_level_command() -> None:
    assert twin.parse_level("level 6") == 6
    assert twin.parse_level("level 9") is None and twin.parse_level("levels") is None


def _steps(names):
    return [(n, lambda n=n: n) for n in names]


def test_run_steps_all_ok_and_none_actions_skip_execution() -> None:
    z = iter([0.765, 0.765, 0.875, 0.88, 0.88, 0.78, 0.78, 0.78])
    executed, out = [], []
    steps = [
        *_steps(["grasp", "lift", "place"]),
        ("align", lambda: None),
        *_steps(["release", "retreat", "home"]),
    ]
    reports = twin.run_steps(
        steps,
        executed.append,
        lambda: True,
        lambda: [0.0, 0.0, next(z)],
        out.append,
        {"align": "error 1.0 mm"},
    )
    assert [r.ok for r in reports] == [True] * 7 and "align" not in executed
    assert reports[3].detail == "error 1.0 mm" and "error 1.0 mm" in out[3]


def test_run_steps_detects_missed_grasp_and_stops() -> None:
    z = iter([0.765, 0.765, 0.766])
    executed = []
    reports = twin.run_steps(
        _steps(["grasp", "lift", "place"]),
        executed.append,
        lambda: True,
        lambda: [0, 0, next(z)],
        print,
    )
    assert executed == ["grasp", "lift"] and "grasp missed" in reports[-1].detail


def test_run_steps_stops_on_planner_failure_and_exception() -> None:
    reports = twin.run_steps(
        _steps(["grasp", "lift"]), lambda _: None, lambda: False, lambda: [0, 0, 0.8], print
    )
    assert len(reports) == 1 and "no safe trajectory" in reports[0].detail

    def boom(_):
        raise RuntimeError("UnStable")

    reports = twin.run_steps(_steps(["grasp"]), boom, lambda: True, lambda: [0, 0, 0.8], print)
    assert not reports[0].ok and "RuntimeError" in reports[0].detail


def test_run_plan_stops_at_first_failed_step() -> None:
    calls = []

    def do_step(obj, dest, _echo):
        calls.append(obj)
        return twin.Outcome(obj != "b", f"{obj} -> {dest}", [])

    outcomes = twin.run_plan([("a", "x"), ("b", "y"), ("c", "z")], do_step, lambda _: None)
    assert calls == ["a", "b"] and [o.ok for o in outcomes] == [True, False]


class FakeTwin:
    level = ts.LEVELS[2]

    def __init__(self):
        self.done = []

    def pick_and_place(self, obj, dest, _echo):
        self.done.append((obj, dest))
        return twin.Outcome(True, "ok", [])


def _result(payload, model="gemma-4-31b-it", attempts=None):
    return models.RouterResult(
        model if payload else None, payload, attempts or [models.Attempt(model, "ok")]
    )


def test_handle_command_paths() -> None:
    fake, lines = FakeTwin(), []
    refused = twin.handle_command(
        fake,
        {"input": "x", "result": _result({"decision": "refuse", "reason": "no"})},
        lines.append,
    )
    assert refused["result"] == "refused" and fake.done == []
    payload = {
        "decision": "execute",
        "steps": [
            {"object": "red cube", "destination": "staging pad"},
            {"object": "blue cube", "destination": "khay vàng"},
        ],
        "reason": "",
    }
    attempts = [
        models.Attempt("gemma-4-31b-it", "invalid", "x"),
        models.Attempt("gemini-robotics-er-2-preview", "ok"),
    ]
    record = twin.handle_command(
        fake,
        {"input": "y", "result": _result(payload, "gemini-robotics-er-2-preview", attempts)},
        lines.append,
    )
    assert record["result"] == "done" and fake.done == [
        ("red cube", "staging pad"),
        ("blue cube", "yellow tray"),
    ]
    assert any(line.startswith("  router: gemma-4-31b-it invalid") for line in lines)
    failed = twin.handle_command(
        fake,
        {"input": "z", "result": models.RouterResult(None, None, [models.Attempt("m", "quota")])},
        lines.append,
    )
    assert failed["result"] == "model_error"


def test_handle_command_with_vision_maps_detections_to_scene_objects() -> None:
    fake, lines = FakeTwin(), []
    vision = twin.VisionState()
    vision.names = ["red block", "blue block"]
    vision.to_truth = {"red block": "red cube", "blue block": None}
    names = twin.NameTable.from_level(fake.level, vision.names)
    ok_payload = {
        "decision": "execute",
        "steps": [{"object": "red block", "destination": "white tray"}],
    }
    record = twin.handle_command(
        fake, {"input": "a", "result": _result(ok_payload), "names": names}, lines.append, vision
    )
    assert record["result"] == "done" and fake.done == [("red cube", "white tray")]
    lost = {"decision": "execute", "steps": [{"object": "blue block", "destination": "white tray"}]}
    record = twin.handle_command(
        fake, {"input": "b", "result": _result(lost), "names": names}, lines.append, vision
    )
    assert record["result"] == "failed" and len(fake.done) == 1


def test_policy_helpers() -> None:
    assert (
        twin.policy_instruction("ball", "deep bin", "bin")
        == "pick up the ball and put it in the deep bin"
    )
    assert (
        twin.policy_instruction("ball", "staging pad", "pad") == "put the ball on the staging pad"
    )
    cube = ts.LEVELS[2].objects[0]
    tray = ts.LEVELS[2].containers[0]
    resting = [tray.xy[0], tray.xy[1], TABLE + ts.FLOOR_T + cube.half_height]
    held = [tray.xy[0], tray.xy[1], TABLE + 0.1]
    assert twin.policy_done(tray, cube, resting, TABLE) and not twin.policy_done(
        tray, cube, held, TABLE
    )
    assert twin.record_size() == (320, 240)


def test_submit_blocks_until_render_thread_finishes() -> None:
    inbox: queue.Queue = queue.Queue()
    busy = threading.Event()
    busy.set()
    ref = {"busy": busy}
    done = []
    worker = threading.Thread(
        target=lambda: (twin.submit(inbox, ref, "scene", None), done.append(1))
    )
    worker.start()
    kind, _ = inbox.get(timeout=2)
    assert kind == "scene" and not done
    busy.set()
    worker.join(timeout=2)
    assert done == [1]


def test_simulate_plan_checks_capacity_step_by_step() -> None:
    level = ts.LEVELS[6]
    xyz = _xyz(level)
    ok = [
        ("fresh milk carton", "wide tray"),
        ("box 1", "wide tray"),
        ("soy milk carton", "deep bin"),
    ]
    assert twin.simulate_plan(level, xyz, TABLE, ok) is None
    over = [*ok, ("box 2", "wide tray")]
    assert "step 4: the wide tray would be full" in twin.simulate_plan(level, xyz, TABLE, over)
    in_trays = {**xyz}
    for name, dx in (("fresh milk carton", -0.06), ("soy milk carton", 0.06)):
        in_trays[name] = [ts.WIDE_TRAY.xy[0] + dx, ts.WIDE_TRAY.xy[1], TABLE + 0.054]
    for name, dx in (("box 1", -0.04), ("box 2", 0.04)):
        in_trays[name] = [ts.DEEP_BIN.xy[0] + dx, ts.DEEP_BIN.xy[1], TABLE + 0.029]
    swap = [("box 1", "table"), ("fresh milk carton", "deep bin"), ("box 1", "wide tray")]
    assert twin.simulate_plan(level, in_trays, TABLE, swap) is None  # swap through the table
    direct = [("fresh milk carton", "deep bin")]
    assert "deep bin would be full" in twin.simulate_plan(level, in_trays, TABLE, direct)
    names = twin.NameTable.from_level(level)
    payload = {"decision": "execute", "steps": [{"object": o, "destination": d} for o, d in over]}
    assert "would be full" in twin.plan_problem(payload, names, (level, xyz, TABLE))
    assert twin.plan_problem(payload, names) is None  # without scene state: names only
    xyz["box 1"] = [1.2, -0.09, 0.024]
    assert "off the table" in twin.simulate_plan(level, xyz, TABLE, [("box 1", "wide tray")])


def test_precheck_refuses_off_table_and_leaning_but_allows_lying_objects() -> None:
    level = ts.LEVELS[6]
    xyz = _xyz(level)
    xyz["box 2"] = [1.9, -0.09, 0.024]
    off = twin.precheck(level, "box 2", "wide tray", xyz, TABLE)
    assert off is not None and "off the table" in off.summary and not off.moved
    xyz["soy milk carton"][2] = TABLE + ts.CARTON_HALF[1]
    lying = twin.precheck(
        level, "soy milk carton", "wide tray", xyz, TABLE, {"soy milk carton": 90.0}, None, "y"
    )
    assert lying is None  # lying flat on its side: grasped across its 4.5 cm width
    leaning = twin.precheck(
        level, "soy milk carton", "wide tray", xyz, TABLE, {"soy milk carton": 40.0}, None, "y"
    )
    assert leaning is not None and "resting on an edge" in leaning.summary and not leaning.moved
    assert twin.precheck(level, "box 1", "table", xyz, TABLE).skipped


def test_run_steps_guard_and_step_errors_stop_the_sequence() -> None:
    executed = []
    calls = {"n": 0}

    def guard(step):
        calls["n"] += 1
        return "collision: volleyball moved 120.0 cm" if step == "lift" else None

    z = iter([0.76, 0.76, 0.90, 0.90])
    reports = twin.run_steps(
        _steps(["grasp", "lift", "place"]),
        executed.append,
        lambda: True,
        lambda: [0, 0, next(z)],
        print,
        guard=guard,
    )
    assert executed == ["grasp", "lift"] and "collision" in reports[-1].detail

    def boom():
        raise twin.StepError("object is stuck in the gripper")

    reports = twin.run_steps(
        [("check release", boom)], executed.append, lambda: True, lambda: [0, 0, 0.8], print
    )
    assert reports[0].detail == "object is stuck in the gripper"


def test_listing_marks_off_table_and_fallen_objects() -> None:
    level = ts.LEVELS[6]
    xyz = _xyz(level)
    xyz["box 2"] = [1.9, -0.09, 0.024]
    rows = twin.scene_listing(level, xyz, TABLE, {"soy milk carton": 90.0}, finger_axis="y")
    assert any("box 2" in r and "OFF THE TABLE" in r for r in rows)
    assert any("soy milk carton" in r and "lying on its side" in r for r in rows)
    rows = twin.scene_listing(level, xyz, TABLE, {"soy milk carton": 40.0}, finger_axis="y")
    assert any("soy milk carton" in r and "NOT GRASPABLE NOW" in r for r in rows)
    assert any(r.startswith("- table:") for r in rows)


def test_run_plan_continues_after_a_blocked_step_but_stops_after_a_motion_failure() -> None:
    calls = []

    def do_step(obj, dest, _echo):
        calls.append(obj)
        if obj == "a":
            return twin.Outcome(False, "a is too close", [], moved=False)
        return twin.Outcome(obj != "b", f"{obj} -> {dest}", [])

    lines = []
    outcomes = twin.run_plan([("a", "x"), ("b", "y"), ("c", "z")], do_step, lines.append)
    assert calls == ["a", "b"] and [o.ok for o in outcomes] == [False, False]
    assert any("BLOCKED" in line for line in lines)


def test_grasp_direction_avoids_a_close_neighbour() -> None:
    level = ts.LEVELS[6]
    xyz = _xyz(level)
    xyz["soy milk carton"] = [
        xyz["box 1"][0] + 0.0384,
        xyz["box 1"][1] - 0.0042,
        TABLE + 0.02,
    ]  # v5 log: fallen next to box 1
    tilts = {"soy milk carton": 90.0}
    # finger axis known: the fingers close along y, away from the carton lying on the left
    assert twin.precheck(level, "box 1", "wide tray", xyz, TABLE, tilts, None, "y") is None
    poses = twin.object_poses(xyz, tilts)
    choice = twin.grasp_choice(level, "box 1", xyz, poses, TABLE, "y")
    assert choice.ok and abs(choice.finger_world[1]) > 0.99
    # finger axis unknown: both directions must have room, so the grasp is refused before motion
    blocked = twin.precheck(level, "box 1", "wide tray", xyz, TABLE, tilts)
    assert blocked is not None and not blocked.moved and "soy milk carton" in blocked.summary
    rows = twin.scene_listing(level, xyz, TABLE, tilts)
    assert any(r.startswith("- box 1") and "NOT GRASPABLE NOW" in r for r in rows)


def test_handle_command_drops_objects_that_already_failed() -> None:
    fake, lines = FakeTwin(), []
    payload = {
        "decision": "execute",
        "steps": [
            {"object": "red cube", "destination": "white tray"},
            {"object": "blue cube", "destination": "yellow tray"},
        ],
    }
    record = twin.handle_command(
        fake,
        {"input": "x", "result": _result(payload)},
        lines.append,
        exclude={"red cube"},
        round_index=1,
    )
    assert fake.done == [("blue cube", "yellow tray")] and record["dropped"] == ["red cube"]
    only_failed = twin.handle_command(
        fake,
        {"input": "x", "result": _result(payload)},
        lines.append,
        exclude={"red cube", "blue cube"},
        round_index=1,
    )
    assert only_failed["result"] == "nothing_left"
    refused = twin.handle_command(
        fake,
        {"input": "x", "result": _result({"decision": "refuse", "reason": "all done"})},
        lines.append,
        round_index=1,
    )
    assert refused["result"] == "nothing_left" and any("NO FURTHER STEPS" in line for line in lines)


class ReplanWorld:
    """Level 6 replay of the v5 log: carrying the fresh milk carton knocks the soy carton over."""

    def __init__(self, failures: int = 1):
        self.level = ts.LEVELS[6]
        self.xyz = _xyz(self.level)
        self.tilts: dict[str, float] = {}
        self.done = []
        self.failures = failures  # how many times carrying the fresh carton hits the soy one

    def pick_and_place(self, obj, dest, _echo):
        blocked = twin.precheck(self.level, obj, dest, self.xyz, TABLE, self.tilts)
        if blocked is not None:
            return blocked
        c = self.level.container(dest)
        if obj == "fresh milk carton" and self.failures > 0:  # soy carton falls over
            self.failures -= 1
            soy = self.xyz["soy milk carton"]  # knocked over where it stood
            self.xyz["soy milk carton"] = [soy[0], soy[1] - 0.0042, TABLE + 0.02]
            self.tilts["soy milk carton"] = 90.0
            self.xyz[obj] = [0.025, -0.14, TABLE + 0.05]  # dropped between the trays
            return twin.Outcome(
                False, "motion 'place' failed: collision: soy milk carton moved 3.9 cm", []
            )
        self.done.append((obj, dest))
        dx = {"box 2": 0.06, "box 1": -0.06, "fresh milk carton": 0.04}.get(obj, -0.04)
        self.xyz[obj] = [c.xy[0] + dx, c.xy[1], TABLE + 0.03]
        self.tilts.pop(obj, None)
        return twin.Outcome(True, f"{obj} is in the {dest}", [])


class ScriptedPlanner:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def plan(self, prompt, names, scene=None, escalate=True):
        assert escalate is False  # replans never spend the escalation budget
        self.prompts.append(prompt)
        return _result(self.payloads.pop(0))


def _plan(*pairs):
    return {
        "decision": "execute",
        "steps": [{"object": o, "destination": d} for o, d in pairs],
        "reason": "",
    }


def _run_adaptive(world, first, planner, lines, logs, max_replans=2, placed=None):
    names = twin.NameTable.from_level(world.level)

    def run_round(data, exclude, k):
        return twin.handle_command(
            world, data, lines.append, None, world.pick_and_place, exclude, k
        )

    def observe(before, handled):
        moved = ts.unexpected_moves(before, world.xyz, handled)
        return [f"{n} moved {d * 100:.1f} cm" for n, d in moved.items()]

    def rescan(reason):
        listing = twin.scene_listing(world.level, world.xyz, TABLE, world.tilts)
        return names, listing, (world.level, dict(world.xyz), TABLE), None

    return twin.adaptive_command(
        {
            "input": "put both milk cartons in the deep bin and both boxes in the wide tray",
            "result": _result(first),
            "names": names,
        },
        lines.append,
        objects_xyz=lambda: {k: list(v) for k, v in world.xyz.items()},
        run_round=run_round,
        observe=observe,
        rescan=rescan,
        planner=planner,
        log=logs.append,
        max_replans=max_replans,
        placed=placed,
    )


def test_adaptive_command_recovers_after_a_collision_until_done() -> None:
    world, lines, logs = ReplanWorld(failures=1), [], []
    first = _plan(
        ("fresh milk carton", "deep bin"),
        ("soy milk carton", "deep bin"),
        ("box 1", "wide tray"),
        ("box 2", "wide tray"),
    )
    planner = ScriptedPlanner(
        [
            # round 1: a NEW plan from the rescanned scene: the failed carton again (attempt
            # 2 of 3) and the soy carton lying on its side
            _plan(
                ("fresh milk carton", "deep bin"),
                ("soy milk carton", "deep bin"),
                ("box 1", "wide tray"),
                ("box 2", "wide tray"),
            ),
        ]
    )
    placed = lambda o, d: ts.in_destination(  # noqa: E731
        world.level, world.level.container(d), world.xyz[o], TABLE
    )
    summary = _run_adaptive(world, first, planner, lines, logs, max_replans=6, placed=placed)
    assert world.done == [
        ("fresh milk carton", "deep bin"),
        ("soy milk carton", "deep bin"),
        ("box 1", "wide tray"),
        ("box 2", "wide tray"),
    ]
    assert summary["result"] == "done" and summary["replans"] == 1
    assert summary["recovered"] == ["fresh milk carton"] and summary["failed"] == {}
    assert summary["attempts"] == {"fresh milk carton": 1}
    assert any(c.startswith("soy milk carton moved") for c in summary["changes"])
    prompt = planner.prompts[0]
    assert "REPLANNING" in prompt and "failed attempt 1/3 for fresh milk carton" in prompt
    assert "unexpected change: soy milk carton moved" in prompt and "NOT GRASPABLE NOW" in prompt
    assert "knocked over" in prompt and "not executed yet: soy milk carton -> deep bin" in prompt
    assert [r.get("round") for r in logs if r["event"] == "command"] == [0, 1]
    assert any(line.startswith("RESCAN (soy milk carton moved") for line in lines)


def test_adaptive_command_gives_up_an_object_after_three_failed_attempts() -> None:
    world, lines, logs = ReplanWorld(failures=99), [], []
    again = _plan(("fresh milk carton", "deep bin"))
    planner = ScriptedPlanner([again, again, again])
    summary = _run_adaptive(world, again, planner, lines, logs, max_replans=6)
    assert summary["attempts"] == {"fresh milk carton": 3} and summary["result"] == "failed"
    assert summary["replans"] == 3 and any("given up in this command" in x for x in lines)
    assert "GIVE UP, do not plan fresh milk carton again" in planner.prompts[-1]


def test_adaptive_command_puts_back_an_object_knocked_out_of_its_destination() -> None:
    progress = twin.CommandProgress(done=[("box 1", "table"), ("box 2", "wide tray")])
    lost = progress.check_placed(lambda o, d: o != "box 2")
    assert lost == ["box 2 was knocked out of the wide tray"]
    assert progress.displaced == {"box 2": "wide tray"} and progress.outstanding()
    assert progress.check_placed(lambda o, d: False) == []  # temporary table spots ignored
    names = twin.NameTable.from_level(ts.LEVELS[6])
    prompt = twin.build_replan_prompt("cmd", names, ["Objects:"], progress)
    assert "knocked out of place: box 2 had been put in the wide tray" in prompt


def test_stuck_object_is_never_planned_again() -> None:
    progress = twin.CommandProgress()
    record = {
        "plan": [["box 1", "wide tray"]],
        "outcomes": [
            {
                "ok": False,
                "moved": True,
                "skipped": False,
                "summary": f"motion 'check release' failed: object is {twin.STUCK_TEXT}",
            }
        ],
    }
    progress.absorb(record)
    assert progress.excluded() == {"box 1"} and progress.attempts == {"box 1": 1}


def test_adaptive_command_stops_without_replanning_when_all_went_well() -> None:
    world, lines, logs = ReplanWorld(), [], []
    planner = ScriptedPlanner([])
    summary = _run_adaptive(world, _plan(("box 2", "wide tray")), planner, lines, logs)
    assert summary == {**summary, "replans": 0, "result": "done", "rounds": 1}
    assert planner.prompts == [] and not any("COMMAND RESULT" in line for line in lines)


def test_adaptive_command_respects_the_replan_limit() -> None:
    world, lines, logs = ReplanWorld(), [], []
    first = _plan(("fresh milk carton", "deep bin"))
    planner = ScriptedPlanner([_plan(("box 1", "wide tray"))])
    summary = _run_adaptive(world, first, planner, lines, logs, max_replans=1)
    assert summary["replans"] == 1 and any("replanning limit reached" in line for line in lines)


def test_camera_check_names_a_moved_object_and_ignores_planned_ones() -> None:
    import numpy as np
    from twin_geometry import load_camera_spec, project

    spec = load_camera_spec()

    class CamTwin:
        camera_spec = spec
        table_top = TABLE
        image = None

        def workspace_rgb(self):
            return self.image

    cam = CamTwin()
    reference = np.full((spec.height, spec.width, 3), 120, np.uint8)
    watched = {"box 1": (0.05, -0.05, TABLE + 0.025), "box 2": (0.15, -0.05, TABLE + 0.025)}
    cam.image = reference.copy()
    quiet = twin.camera_check(cam, reference, watched)
    assert quiet["changed"] is False and quiet["flagged"] == []
    u, v = (round(c) for c in project(spec, watched["box 1"]))
    cam.image[v - 30 : v + 30, u - 30 : u + 30] = 240  # box 1 moved / vanished
    moved = twin.camera_check(cam, reference, watched)
    assert moved["changed"] and moved["flagged"] == ["box 1"]
    assert "around box 1" in twin.describe_camera_change(moved)[0]
    planned = twin.camera_check(cam, reference, {"box 2": watched["box 2"]}, [watched["box 1"]])
    assert planned["changed"] is False  # the plan moved box 1 on purpose
    assert twin.camera_check(cam, None, watched) is None


def test_replan_prompt_uses_the_scanned_names() -> None:
    progress = twin.CommandProgress(
        done=[("box 2", "wide tray")], failed={"box 1": "grasp missed"}, changes=["x moved"]
    )
    names = twin.NameTable.from_level(ts.LEVELS[6], ["brown box", "box two"])
    prompt = twin.build_replan_prompt(
        "cmd", names, ["Objects:"], progress, {"box 1": "brown box", "box 2": "box two"}
    )
    assert "done: box two -> wide tray" in prompt and "failed attempt 1/3 for brown box" in prompt
    assert prompt.index("Operator command:") < prompt.index("REPLANNING")
    progress.attempts["box 1"] = 3
    prompt = twin.build_replan_prompt(
        "cmd", names, ["Objects:"], progress, {"box 1": "brown box", "box 2": "box two"}
    )
    assert "GIVE UP, do not plan brown box again" in prompt


def test_system_prompt_handles_categories_partial_plans_and_typos() -> None:
    for word in ("chai nước", "any bottle", "possible part", "typos", "fallen ones"):
        assert word in twin.SYSTEM
    water = ts.LEVELS[7].object("water bottle")
    assert "chai nước" not in water.aliases  # generic word, not the water bottle's name


def test_grasp_key_is_stable_for_equal_frames() -> None:
    frame = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.24], [0, 0, 0, 1]]
    nudged = [[v - 1e-5 for v in row] for row in frame]  # also turns 0.0 into -0.0
    assert twin.grasp_key(frame) == twin.grasp_key(nudged)
    assert twin.grasp_key(frame) != twin.grasp_key([[0.0, 1.0, 0.0, 0.0], *frame[1:]])


def test_relaxed_world_keeps_tall_objects_and_drops_walls() -> None:
    import twin_obstacles as tob

    level = ts.LEVELS[6]
    xyz = _xyz(level)
    boxes = tob.scene_boxes(level, xyz, {n: [1.0, 0, 0, 0] for n in xyz}, TABLE)
    loaded = []

    class Fake(twin.Twin):
        def __init__(self):
            self.world_state, self.world_boxes, self.relaxed_this_pick = "scene", boxes, []
            self.task = None

    original = tob.load_world
    tob.load_world = lambda task, kept: loaded.append([b.name for b in kept]) or "ok"
    try:
        world = Fake()
        assert world.relax_world("grasp") and world.world_state == "relaxed"
        assert sorted(loaded[-1]) == ["fresh milk carton", "soy milk carton"]
        assert not world.relax_world("again")  # already relaxed
    finally:
        tob.load_world = original


def test_nudge_into_limits_moves_joints_off_their_limits() -> None:
    class Joint:
        def __init__(self, name, lo, hi):
            self.name, self.lim, self.target = name, (lo, hi), None

        def get_name(self):
            return self.name

        def get_limits(self):
            return [self.lim]

        def set_drive_target(self, q):
            self.target = q

    joints = [Joint("joint1", -2.4, 2.4), Joint("joint2", 0.0, 3.2), Joint("gripper", 0.0, 0.04)]

    class Entity:
        def __init__(self):
            self.qpos = [0.5, -0.0003, 0.0]

        def get_active_joints(self):
            return joints

        def get_qpos(self):
            return list(self.qpos)

        def set_qpos(self, q):
            self.qpos = list(q)

    class Robot:
        left_entity = Entity()
        left_arm_joints = joints[:2]

    class World(twin.Twin):
        def __init__(self):
            self.task = type("T", (), {"robot": Robot()})()

        def _settle(self, steps=0):
            pass

    world = World()
    assert world.nudge_into_limits()
    q = Robot.left_entity.qpos
    assert q[0] == 0.5 and abs(q[1] - 0.01) < 1e-12 and q[2] == 0.0  # gripper untouched
    assert joints[1].target == q[1] and not world.nudge_into_limits()


def test_table_only_world_lasts_one_straight_motion() -> None:
    import twin_obstacles as tob

    level = ts.LEVELS[6]
    xyz = _xyz(level)
    boxes = tob.scene_boxes(level, xyz, {n: [1.0, 0, 0, 0] for n in xyz}, TABLE)
    loaded = []

    class Fake(twin.Twin):
        def __init__(self):
            self.world_state, self.world_boxes, self.relaxed_this_pick = "scene", boxes, []
            self.task = None

        def _linear_move(self, arm, dx, dy, dz, guard, label):
            assert self.relax_world(label) and self.world_state == "table"
            raise twin.StepError("no safe trajectory")

    original = tob.load_world
    tob.load_world = lambda task, kept: loaded.append(len(kept)) or "ok"
    try:
        world = Fake()
        try:
            world.linear_move(None, 0, 0, 0.1, None, "lift")
            raise AssertionError("expected StepError")
        except twin.StepError:
            pass
        assert world.world_state == "scene" and loaded == [0, len(boxes)]
    finally:
        tob.load_world = original


def test_twenty_degree_tilt_is_not_held_as_vertical() -> None:
    import math

    import twin_grasp as tg

    def q_about_y(deg):  # gripper x down, tilted
        r = math.radians(90 + deg) / 2
        return [math.cos(r), 0.0, math.sin(r), 0.0]

    assert tg.hold_vector(q_about_y(0), "vertical") is not None
    assert tg.hold_vector(q_about_y(20), "vertical") is None
