"""Evidence boundaries and replay tests for the interactive S1-I fixture demo."""

from __future__ import annotations

import json
import sys
from copy import deepcopy

import pytest
from scripts.demo_semantic_to_grasp_fixture import main, run_fixture

from vla_robot_sorting.visualization.semantic_report import render_fixture_dashboard


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return run_fixture()


def test_synthetic_geometry_report_has_real_scene_and_all_6d_candidates(result) -> None:
    assert result["schema"] == "semantic-to-grasp-fixture-demo-v2"
    assert result["source"] == "synthetic_fixture"
    assert result["network_calls"] == 0
    assert result["native_simulation_calls"] == 0
    assert result["physics_steps"] == 0
    assert len(result["scene"]["objects"]) == 2
    assert len(result["scene"]["destinations"]) == 1
    en, vi = result["cases"]
    assert (en["target"], en["destination"]) == (vi["target"], vi["destination"])
    assert en["candidate_ids"] == vi["candidate_ids"]
    for case in (en, vi):
        assert case["candidate_count"] == len(case["candidates"]) == 12
        assert case["feasibility_checked"] is False
        assert case["executed"] is False
        for candidate in case["candidates"]:
            assert len(candidate["grasp_pose"]["position_xyz"]) == 3
            assert len(candidate["grasp_pose"]["orientation_xyzw"]) == 4
            assert len(candidate["pregrasp_pose"]["position_xyz"]) == 3
            assert len(candidate["approach_direction"]) == 3
            assert candidate["scene_revision"] == case["scene_revision"]
            assert candidate["transform_revision"] == case["transform_revision"]


def test_adversarial_scenarios_reject_without_any_grasp_or_execution(result) -> None:
    failures = {item["failure_code"] for item in result["rejections"]}
    assert failures == {"observation_stale", "target_unavailable", "grounding_failed"}
    assert all(item["candidate_count"] == 0 for item in result["rejections"])
    assert all(item["executed"] is False for item in result["rejections"])


def test_html_is_offline_and_contains_geometry_inspection(result) -> None:
    html = render_fixture_dashboard(result)
    assert 'id="topView"' in html and 'id="sideView"' in html
    assert 'id="candidateList"' in html
    assert "fixtureData" in html
    assert "IK / COLLISION: NOT TESTED" in html
    assert "NO PHYSICS" in html
    assert "raw_quality" in html
    assert "addEventListener" in html
    assert "https://cdn" not in html
    payload = html.split('<script id="fixtureData" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]
    assert json.loads(payload)["cases"][0]["candidate_count"] == 12


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "recorded_native"),
        ("network_calls", 1),
        ("native_simulation_calls", 1),
        ("physics_steps", 1),
        ("schema", "unknown"),
    ],
)
def test_renderer_refuses_to_relabel_other_data_as_fixture(result, field, value) -> None:
    data = deepcopy(result)
    data[field] = value
    with pytest.raises(ValueError, match="synthetic"):
        render_fixture_dashboard(data)


@pytest.mark.parametrize(
    ("key", "value"),
    [("executed", True), ("feasibility_checked", True), ("candidate_count", 0)],
)
def test_renderer_rejects_false_physical_proofs(result, key, value) -> None:
    data = deepcopy(result)
    data["cases"][0][key] = value
    with pytest.raises(ValueError, match="geometry-only"):
        render_fixture_dashboard(data)


def test_dashboard_escapes_embedded_script_in_data(result) -> None:
    data = deepcopy(result)
    data["cases"][0]["command"] = '</script><script>alert("bad")</script>'
    html = render_fixture_dashboard(data)
    assert 'alert("bad")</script>' not in html
    assert "\\u003c/script\\u003e" in html


def test_cli_emits_html_and_json_in_same_run(result, tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "replay.json"
    monkeypatch.setattr(sys, "argv", ["demo_semantic_to_grasp_fixture.py", "--output", str(output)])
    main()
    json_record = json.loads(output.read_text(encoding="utf-8"))
    html = output.with_suffix(".html").read_text(encoding="utf-8")
    assert json_record == json.loads(json.dumps(result, ensure_ascii=False))
    assert "SYNTHETIC FIXTURE" in html
    assert "safe_rejections=3" in capsys.readouterr().out
