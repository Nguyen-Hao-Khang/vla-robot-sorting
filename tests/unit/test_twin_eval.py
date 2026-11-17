"""Simulated evaluation: judging, metrics and seeded episode plans (tools/twin/twin_eval.py)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))

import eval_metrics as em  # noqa: E402
import sorting_task as st  # noqa: E402
import twin_eval as te  # noqa: E402

BANK = st.load_template_bank(ROOT / "configs" / "command_templates.json")


def test_wilson_matches_reference_values() -> None:
    assert em.wilson(0, 30) == pytest.approx((0.0, 0.1135), abs=1e-4)
    assert em.wilson(15, 30) == pytest.approx((0.3315, 0.6685), abs=1e-4)
    assert em.wilson(30, 30) == pytest.approx((0.8865, 1.0), abs=1e-4)
    assert em.wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        em.wilson(5, 3)


def test_judge_success_and_failures() -> None:
    before = {"a": "table", "b": "table", "c": "tray"}
    ok = em.judge_episode(
        before, {"a": "bin", "b": "table", "c": "tray"}, ["a"], "bin", ["a"], False
    )
    assert ok["success"] and ok["target_selected"] and ok["placed"] == ["a"]
    # a class command: either member counts
    member = em.judge_episode(
        before, {"a": "table", "b": "bin", "c": "tray"}, ["a", "b"], "bin", ["b"], False
    )
    assert member["success"]
    # wrong object in the container
    wrong = em.judge_episode(
        before, {"a": "table", "b": "bin", "c": "tray"}, ["a"], "bin", ["b"], False
    )
    assert not wrong["success"] and not wrong["target_selected"] and wrong["grasped"]
    # right object, but another object also moved into a container
    extra = em.judge_episode(
        before, {"a": "bin", "b": "tray", "c": "tray"}, ["a"], "bin", ["a", "b"], False
    )
    assert not extra["success"] and extra["intruders"] == ["b"]
    # safety stop
    stop = em.judge_episode(
        before, {"a": "bin", "b": "table", "c": "tray"}, ["a"], "bin", ["a"], True
    )
    assert not stop["success"]
    # an object already in a container that stays there is not an intruder
    assert ok["intruders"] == []


def test_failure_phases() -> None:
    assert em.failure_phase("final approach") == "approach"
    assert em.failure_phase("grasp") == "grasp"
    assert em.failure_phase("lift") == "lift"
    assert em.failure_phase("transit to the bin") == "transport"
    assert em.failure_phase("descend") == "place"
    assert em.failure_phase("calibrate") is None
    assert em.policy_failure_phase([], ["a"], False) == "grasp"
    assert em.policy_failure_phase(["b"], ["a"], False) == "grasp"
    assert em.policy_failure_phase(["a"], ["a"], False) == "transport"
    assert em.policy_failure_phase(["a"], ["a"], True) == "place"


def test_summarise_groups_and_table() -> None:
    records = [
        {
            "language": "en",
            "success": True,
            "target_selected": True,
            "grasped": True,
            "episode_time_s": 10.0,
        },
        {
            "language": "en",
            "success": False,
            "target_selected": True,
            "grasped": True,
            "failure_phase": "place",
            "episode_time_s": 20.0,
        },
        {
            "language": "vi",
            "success": False,
            "target_selected": False,
            "grasped": False,
            "failure_phase": "grasp",
            "safety_interventions": 2,
        },
    ]
    rows = em.summarise(records, ["language"])
    assert [r["language"] for r in rows] == ["en", "vi"]
    en, vi = rows
    assert en["n"] == 2 and en["success"]["k"] == 1 and en["episode_time_s"]["median"] == 15.0
    assert en["failure_phases"] == {"place": 1}
    assert vi["safety_interventions_mean"] == 2.0 and vi["episode_time_s"]["median"] is None
    table = em.format_table(rows, ["language"])
    assert "| en | 2 | 1/2 = 0.50" in table


def test_episode_plans_are_seeded_and_paired_across_languages() -> None:
    for index in range(20):
        seed = te.episode_seed(3, index)
        en = te.plan_episode(seed, "en", 0.3, BANK, 1.0)
        vi = te.plan_episode(seed, "vi", 0.3, BANK, 1.0)
        again = te.plan_episode(seed, "en", 0.3, BANK, 1.0)
        assert en == again
        assert en.scene == vi.scene and en.expert_target == vi.expert_target
        for field in ("kind", "targets", "destination", "object_reference", "container_reference"):
            assert getattr(en.command, field) == getattr(vi.command, field)
        assert en.expert_target in en.command.targets
        assert en.command.language == "en" and vi.command.language == "vi"


def test_command_sources(tmp_path) -> None:
    assert te.load_bank("builtin", tmp_path / "x.json") == ({}, 0.0)
    bank, share = te.load_bank("generated", ROOT / "configs" / "command_templates.json")
    assert share == 1.0 and len(bank["vi"]["class"]) >= 30
    with pytest.raises(SystemExit):
        te.load_bank("generated", tmp_path / "x.json")
    with pytest.raises(SystemExit):
        te.load_human_plans(tmp_path / "x.csv", 0.3)


def _draft_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "draft_test_commands", ROOT / "tools/data/draft_test_commands.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_human_command_file_round_trip(tmp_path) -> None:
    draft = _draft_module()
    items = draft.draft_items(9, 6, 0.3)
    assert [i["language"] for i in items] == ["vi", "en"] * 3
    path = tmp_path / "commands.csv"
    draft.write_csv(path, items, {i["id"]: f"text {i['id']}" for i in items})
    with pytest.raises(SystemExit):  # nothing reviewed yet
        te.load_human_plans(path, 0.3)
    plans = te.load_human_plans(path, 0.3, reviewed_only=False)
    assert [p.command.text for p in plans] == [f"text {i}" for i in range(6)]
    for plan, item in zip(plans, items, strict=True):
        assert plan.seed == item["seed"] and plan.command.destination == item["destination"]
        assert plan.command.language == item["language"]
    # the same seeds with template commands give the same scenes (paired comparison)
    template = te.plan_episode(items[0]["seed"], "vi", 0.3, {}, 0.0)
    assert template.scene == plans[0].scene
    # a wrong class share changes the commands and is caught
    with pytest.raises(SystemExit):
        te.load_human_plans(path, 1.0, reviewed_only=False)


def test_draft_check_rejects_training_templates() -> None:
    draft = _draft_module()
    item = draft.draft_items(9, 2, 0.0)[1]  # English object command
    copy = st.EN_OBJECT_TEMPLATES[0].format(
        obj=item["object_phrase"].removeprefix("the "), dest=item["container_phrase"]
    )
    assert draft.check_draft(copy, item, {}) == "same as a training template"
    assert (
        draft.check_draft(
            "would you be so kind as to tidy " + item["object_phrase"] + " away", item, {}
        )
        is None
    )
    assert draft.check_draft("đặt nó vào khay", item, {}) == "not english"


def test_episode_record_and_summarise_cli(tmp_path, capsys) -> None:
    plan = te.plan_episode(te.episode_seed(1, 0), "en", 0.0, {}, 0.0)
    target, dest = plan.command.targets[0], plan.command.destination
    before = {o.name: "table" for o in plan.scene.objects}
    after = dict(before, **{target: dest})
    result = {
        "lifted": [target],
        "episode_time_s": 12.5,
        "safety_interventions": 0,
        "safety_stop": False,
        "executor_phase": None,
        "summary": "ok",
    }
    run = {"executor": "expert", "command_source": "builtin"}
    record = te.episode_record(run, 0, plan, before, after, result)
    assert record["success"] and record["failure_phase"] is None and record["language"] == "en"
    slow = te.episode_record(run, 1, plan, before, after, dict(result, episode_time_s=61.0))
    assert not slow["success"] and slow["failure_phase"] == "timeout"
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (record, slow)) + "\n", encoding="utf-8")
    out = tmp_path / "rows.json"
    assert te.main(["summarise", str(path), "--by", "executor", "--json", str(out)]) == 0
    assert "| expert | 2 | 1/2" in capsys.readouterr().out
    assert json.loads(out.read_text())[0]["n"] == 2


def test_reason_key_drops_numbers() -> None:
    assert te.reason_key("joint 2 target -0.123 rad outside [0.00, 3.20]") == (
        "joint # target # rad outside [#, #]"
    )
    assert te.reason_key("gripper clamped to its range") == "gripper clamped to its range"


class _FakeRobot:
    def __init__(self, state):
        self.state = list(state)

    def get_left_arm_jointState(self):
        return list(self.state)


class _FakeTwin:
    """Enough of twin_console.Twin for run_actions: the arm jumps to every commanded target."""

    table_top = 0.74

    def __init__(self, plan, state):
        self.level = st.to_level(plan.scene)
        self.task = types.SimpleNamespace(robot=_FakeRobot(state))
        self.sent = []

    def objects_xyz(self):
        return {o.name: [0.0, 0.5, self.table_top + 0.02] for o in self.level.objects}

    def record_frame(self, size):
        return self.task.robot.state, {"wrist": None}

    def drive_left_arm(self, command):
        self.sent.append([float(v) for v in command])
        self.task.robot.state = [float(v) for v in command]

    def _recover(self, arm):
        pass

    def _settle(self, steps):
        pass


def test_run_actions_filters_counts_reasons_and_stops_on_none(monkeypatch) -> None:
    envs = types.ModuleType("envs")
    utils = types.ModuleType("envs.utils")
    utils.ArmTag = lambda name: name
    monkeypatch.setitem(sys.modules, "envs", envs)
    monkeypatch.setitem(sys.modules, "envs.utils", utils)
    plan = te.plan_episode(te.episode_seed(1, 0), "en", 0.0, {}, 0.0)
    q0 = [0.0, 1.4, 0.4, 0.6, 0.0, 0.0, 1.0]  # tool 39 cm out, 10 cm above the table
    targets = [
        [*q0[:6], 1.04],  # gripper slightly past open: clamped, not counted
        [q0[0] + 0.8, *q0[1:6], 1.0],  # jump: rejected, the arm holds
        [q0[0] + 0.01, *q0[1:6], 1.0],
    ]
    for safety_on in (True, False):
        twin = _FakeTwin(plan, q0)
        queue = iter(targets)
        result = te.run_actions(
            twin, plan, lambda s, i, q=queue: next(q, None), 15.0, (320, 240), 100, safety_on
        )
        assert result["summary"] == "3 actions" and not result["safety_stop"]
        # without the filter the arm really jumps, so the step back is a second jump
        assert result["safety_interventions"] == (1 if safety_on else 2)
        assert list(result["safety_reasons"]) == ["jump of # rad from the measured state"]
        assert twin.sent[0][6] == (1.0 if safety_on else 1.04)
        # with the filter the rejected jump becomes "hold"; without it the raw target is sent
        assert twin.sent[1][0] == pytest.approx(0.0 if safety_on else 0.8)


def test_record_meta_matches_the_recorder_format() -> None:
    plan = te.plan_episode(te.episode_seed(9, 0), "vi", 0.3, {}, 0.0)
    result = {"executor_phase": None, "summary": "ok"}
    meta = te.record_meta(plan, result, 15.0, {"a": (0.1, 0.2, 0.3)}, (320, 240))
    assert meta["task"] == plan.command.text and meta["language"] == "vi"
    assert meta["success"] and meta["seed"] == plan.seed and meta["fps"] == 15.0
    assert meta["object"] in meta["targets"] and meta["spawns"] == {"a": [0.1, 0.2, 0.3]}
    assert {o["name"] for o in meta["scene"]["objects"]} == {o.name for o in plan.scene.objects}
    assert not te.record_meta(plan, {"executor_phase": "grasp", "summary": "x"}, 15, {}, (1, 1))[
        "success"
    ]


def test_record_dir_needs_the_expert(capsys) -> None:
    with pytest.raises(SystemExit):
        te.main(["run", "--executor", "replay", "--record-dir", "x", "--out", "y.jsonl"])
    assert "--record-dir needs --executor expert or dagger" in capsys.readouterr().err


def test_dagger_needs_a_record_dir_and_a_checkpoint(capsys) -> None:
    for argv, message in (
        (["--checkpoint", "c"], "--executor dagger needs --record-dir"),
        (["--record-dir", "x"], "--checkpoint is required with --executor dagger"),
        (
            ["--checkpoint", "c", "--record-dir", "x", "--dagger-min-actions", "50",
             "--dagger-max-actions", "10"],
            "need 1 <= --dagger-min-actions",
        ),
    ):
        with pytest.raises(SystemExit):
            te.main(["run", "--executor", "dagger", "--out", "y.jsonl", *argv])
        assert message in capsys.readouterr().err


def test_dagger_prefix_is_seeded_and_in_range() -> None:
    plans = [te.plan_episode(te.episode_seed(41, i), "en", 0.3, {}, 0.0) for i in range(20)]
    first = [te.dagger_prefix_actions(p, 15, 150) for p in plans]
    assert first == [te.dagger_prefix_actions(p, 15, 150) for p in plans]
    assert all(15 <= n <= 150 for n in first) and len(set(first)) > 5
    vi = te.plan_episode(te.episode_seed(41, 3), "vi", 0.3, {}, 0.0)
    assert te.dagger_prefix_actions(vi, 15, 150) == first[3]  # paired across languages


def test_run_actions_without_recover_leaves_the_arm(monkeypatch) -> None:
    envs = types.ModuleType("envs")
    utils = types.ModuleType("envs.utils")
    utils.ArmTag = lambda name: name
    monkeypatch.setitem(sys.modules, "envs", envs)
    monkeypatch.setitem(sys.modules, "envs.utils", utils)
    plan = te.plan_episode(te.episode_seed(1, 0), "en", 0.0, {}, 0.0)
    q0 = [0.0, 1.4, 0.4, 0.6, 0.0, 0.0, 1.0]
    calls = []
    for recover in (True, False):
        twin = _FakeTwin(plan, q0)
        twin._recover = lambda arm, c=calls, r=recover: c.append(r)
        queue = iter([[q0[0] + 0.01, *q0[1:]]])
        te.run_actions(twin, plan, lambda s, i, q=queue: next(q, None), 15.0, (1, 1), 5,
                       recover=recover)
    assert calls == [True]


def test_record_meta_keeps_the_dagger_prefix() -> None:
    plan = te.plan_episode(te.episode_seed(41, 0), "en", 0.3, {}, 0.0)
    info = {"policy": "p", "actions": 30, "lifted": [], "safety_stop": False}
    meta = te.record_meta(plan, {"executor_phase": None, "summary": "ok"}, 15.0, {}, (1, 1), info)
    assert meta["dagger_prefix"] == info
    assert "dagger_prefix" not in te.record_meta(
        plan, {"executor_phase": None, "summary": "ok"}, 15.0, {}, (1, 1)
    )
