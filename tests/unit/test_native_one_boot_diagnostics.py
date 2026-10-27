from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_native_manipulation_campaign import analyze, as_text

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "scripts/linux/run_sprint1_one_boot.sh"
CAMPAIGN = ROOT / "scripts/linux/run_sprint1_manipulation_campaign.sh"


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_digest_fail_closed_on_missing_campaign_and_seeds(tmp_path: Path) -> None:
    summary = analyze(tmp_path)
    assert summary["campaign_result_present"] is False
    assert summary["campaign_reported"]["accepted"] is None
    assert summary["seeds"]["manipulation-seed-0"]["present"] is False
    assert summary["seeds"]["manipulation-seed-1"]["passed"] is None
    assert "not acceptance" in as_text(summary)


def test_digest_extracts_bounded_ik_and_retreat_link_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "CAMPAIGN_RESULT.txt").write_text(
        "MANIP_PASS_COUNT=1\nMANIP_AUDIT_PASS_COUNT=1\nCAMPAIGN_ACCEPTED=0\n",
        encoding="utf-8",
    )
    _write(
        tmp_path / "manipulation-seed-1-expanded.json",
        {
            "passed": False,
            "task_success_after_release": True,
            "B5": {"selected": {"candidate_id": "pillbottle:cyl:side:0:s-1:up75"}},
            "stages": [
                {"stage": "place_above_support", "passed": True},
                {
                    "stage": "retreat",
                    "passed": False,
                    "contacts": {
                        "physics_steps": 1200,
                        "robot_target_contact_frames": 350,
                        "robot_target_finger_contact_frames": 340,
                        "robot_target_other_link_contact_frames": 10,
                        "first_robot_target_step": 31,
                        "last_robot_target_step": 390,
                    },
                    "post_release_separation": {
                        "passed": True,
                        "finger_contacts_after": [],
                    },
                },
            ],
        },
    )
    _write(
        tmp_path / "manipulation-seed-2.json",
        {
            "passed": False,
            "stages": [
                {
                    "stage": "place_above_support",
                    "passed": False,
                    "failure": {
                        "code": "optimization_failed",
                        "details": {"native_status": "IK Fail"},
                    },
                    "placement_recovery": {
                        "attempts": [{"yaw_offset_rad": 0, "plan_success": False}],
                        "vertical_reachability_diagnostic": {
                            "planning_only": True,
                            "executed": False,
                            "acceptance_override": False,
                            "attempts": [
                                {
                                    "extra_height_m": 0.01,
                                    "plan_success": True,
                                    "native_status": None,
                                },
                            ],
                        },
                    },
                },
            ],
        },
    )
    result = analyze(tmp_path)
    assert result["campaign_reported"]["accepted"] == "0"
    expanded = result["seeds"]["manipulation-seed-1-expanded"]
    assert expanded["retreat"]["contacts"]["robot_target_contact_frames"] == 350
    assert expanded["retreat"]["contacts"]["robot_target_finger_contact_frames"] == 340
    assert expanded["retreat"]["contacts"]["robot_target_other_link_contact_frames"] == 10
    assert expanded["retreat"]["contacts"]["first_robot_target_step"] == 31
    assert expanded["task_success_after_release"] is True
    assert expanded["passed"] is False
    place = result["seeds"]["manipulation-seed-2"]["place"]
    assert place["failure"]["native_status"] == "IK Fail"
    assert place["height_diagnostic"]["planning_only"] is True
    assert place["height_diagnostic"]["executed"] is False
    assert place["height_diagnostic"]["attempts"][0]["plan_success"] is True
    assert result["seeds"]["manipulation-seed-2"]["passed"] is False
    assert "Z+0.01: plan=True" in as_text(result)


def test_digest_ignores_invalid_seed_json_and_nonlist_stages(tmp_path: Path) -> None:
    (tmp_path / "manipulation-seed-0.json").write_text("{bad", encoding="utf-8")
    _write(tmp_path / "manipulation-seed-1.json", {"stages": "not a list"})
    report = analyze(tmp_path)
    assert report["seeds"]["manipulation-seed-0"]["present"] is False
    assert report["seeds"]["manipulation-seed-1"]["stages"] == []


def test_one_boot_runner_never_promotes_accepted_or_reboots() -> None:
    body = BATCH.read_text(encoding="utf-8")
    assert "RUN_SPRINT1_MANIPULATION_CAMPAIGN.sh" in body
    assert "sha256sum -c SHA256SUMS.txt" in body
    assert "--preflight-only" in body
    assert "--verify-only" in body
    assert "--rerun" in body
    assert "Evidence export exit: 0" in body
    assert "CAMPAIGN_RC=" in body
    assert "exit \"$CAMPAIGN_RC\"" in body
    assert "LATEST.txt" in body
    assert "git rev-parse HEAD" in body
    assert "EXPECTED_SOURCE_SHA=" in body
    assert "git reset --hard" not in body
    assert "git clean -fd" not in body
    assert "reboot" not in "\n".join(
        line for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "echo"))
    )
    assert "experiments/accepted/" not in body


def test_native_campaign_digest_is_additive_and_preexport() -> None:
    body = CAMPAIGN.read_text(encoding="utf-8")
    digest = "scripts/analyze_native_manipulation_campaign.py"
    assert digest in body
    assert body.index(digest) < body.index('cp -a "$RESULT_DIR/." "$EVIDENCE/"')
    assert "DIAGNOSTIC_DIGEST_RC" in body
    assert '--json-output "$RESULT_DIR/DIAGNOSTIC_DIGEST.json"' in body
