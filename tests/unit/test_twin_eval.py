"""Simulated evaluation: judging, metrics and seeded episode plans (tools/twin/twin_eval.py)."""

from __future__ import annotations

import json
import sys
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
    assert te.load_bank("builtin", tmp_path / "x.json", None) == ({}, 0.0)
    bank, share = te.load_bank("generated", ROOT / "configs" / "command_templates.json", None)
    assert share == 1.0 and len(bank["vi"]["class"]) >= 30
    with pytest.raises(SystemExit):
        te.load_bank("human", tmp_path / "x.json", None)


def test_safety_filter() -> None:
    f = te.SafetyFilter([(-1.0, 1.0), (0.0, 2.0)], max_step=0.5)
    assert f.check([0.2, 0.1], [0.0, 0.0]) == [0.2, 0.1] and f.interventions == 0
    assert f.check([1.2, 0.1], [0.9, 0.0]) == [1.0, 0.1] and f.interventions == 1
    assert not f.stopped
    f.check([0.0, 1.5], [0.0, 0.0])
    assert f.stopped


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
