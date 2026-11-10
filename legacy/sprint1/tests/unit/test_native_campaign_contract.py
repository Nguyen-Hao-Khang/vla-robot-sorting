from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "scripts/linux/run_sprint1_native_campaign.sh"
MANIP_CAMPAIGN = ROOT / "scripts/linux/run_sprint1_manipulation_campaign.sh"
MANIP_PROBE = ROOT / "scripts/linux/probe_panthera_project_manipulation.py"


def test_native_campaign_is_non_fail_fast_and_preserves_stage_order() -> None:
    text = CAMPAIGN.read_text(encoding="utf-8")
    stripped_lines = {line.strip() for line in text.splitlines()}

    assert "set -e" not in stripped_lines
    assert "set -euo pipefail" not in stripped_lines

    expected = [
        "### Stage A - MPlib scaled-target preflight (seeds 0,1)",
        "### Stage B - 3-seed B5/B7 diagnostic matrix",
        "### Stage C - integrated pregrasp diagnostics (seeds 0,1)",
        "### Stage D - project-owned manipulation diagnostic seed 0",
        "### Stage E - full B5/B7 20-run acceptance",
        "### Stage F - manipulation acceptance seeds 0,1,2",
        "### Stage G - fixed Sprint-1 matrix assembly",
        "=== Exporting complete native campaign evidence ===",
    ]
    positions = [text.index(item) for item in expected]
    assert positions == sorted(positions)

    assert 'SMOKE3_B5_COUNT" -eq 3' in text
    assert 'SMOKE3_B7_COUNT" -eq 3' in text
    assert "MANIP_ACCEPT_PASS_COUNT" in text
    assert "native-working-tree.patch" in text
    assert "native-staged.patch" in text
    assert "collect_to_bridge.sh" in text


def test_manipulation_campaign_is_non_fail_fast_and_runs_all_three_seeds() -> None:
    text = MANIP_CAMPAIGN.read_text(encoding="utf-8")
    stripped_lines = {line.strip() for line in text.splitlines()}

    assert "set -e" not in stripped_lines
    assert "set -euo pipefail" not in stripped_lines

    expected = [
        "=== Accepted prerequisite manifests ===",
        "### Stage A - target-aware MPlib smoke",
        "### Stage B - project-owned manipulation seeds 0,1,2",
        "### Stage C - provisional native closure candidate",
        "### Stage D - current readiness snapshot",
        "### Stage E - Sprint-1 native closure audit",
        "=== Exporting manipulation campaign evidence ===",
    ]
    positions = [text.index(item) for item in expected]
    assert positions == sorted(positions)

    assert "for seed in 0 1 2; do" in text
    assert "MANIP_PASS_COUNT" in text
    assert "MANIP_AUDIT_PASS_COUNT" in text
    assert "failure_phase=" in text
    assert "contact_policy" in text
    assert "phase-summary.txt" in text


def test_manipulation_probe_uses_target_aware_acquisition_and_compact_stdout() -> None:
    text = MANIP_PROBE.read_text(encoding="utf-8")

    assert "PantheraRoboTwinMPlibAcquisitionPlanner(task.robot)" in text
    assert 'payload["mplib_world_before_approach"]' in text
    assert "helpers.sync_scaled_target_into_mplib_world(task)" in text
    assert '"MANIPULATION_SUMMARY "' in text
    assert '"MANIPULATION_STAGE "' in text
    assert "if args.print_json:" in text

    campaign = MANIP_CAMPAIGN.read_text(encoding="utf-8")
    assert "--print-json" not in campaign
    assert "planner_strategy=" in campaign
    assert "selected_planner=" in campaign
    assert "primary_vendor=" in campaign
    assert "fallback_attempted=" in campaign
    assert "fallback_success=" in campaign
    assert "native_status=" in campaign
    assert "native_valid_query=" in campaign
    assert "native_start_valid=" in campaign
    assert "native_start_status=" in campaign
    assert "support_contact_geometry=" in campaign
    assert "lift_table_disabled=" in campaign


def test_manipulation_campaign_has_seed1_recovery_diagnostic_without_acceptance_bypass() -> None:
    text = MANIP_CAMPAIGN.read_text(encoding="utf-8")

    assert "### Stage B2 - seed1 deterministic acquisition recovery" in text
    assert text.index("### Stage B - project-owned manipulation seeds 0,1,2") < text.index(
        "### Stage B2 - seed1 deterministic acquisition recovery"
    ) < text.index("### Stage C - provisional native closure candidate")
    assert 'policy.get("strategy") == "mplib_target_aware_acquisition"' in text
    assert 'policy.get("primary_vendor_status") == "screw plan failed"' in text
    assert 'policy.get("fallback_attempted") is True' in text
    assert "fallback_planning_failed" in text
    assert "target_collision_rejected" in text
    assert "execution_target_contact_rejected" in text
    assert 'contacts.get("robot_target_contact_frames", 0)' in text
    assert 'contact_policy.get("allow_robot_target_contact") is False' in text
    assert 'approach.get("execution_status") == "completed"' in text
    assert 'policy.get("fallback_success") is False' in text
    assert 'policy.get("fallback_success") is True' in text
    assert '"target" in (row.get("matched_environment_roles") or [])' in text
    assert 'failure.get("code") == "optimization_failed"' in text
    assert "--b5-evaluate-all-height-groups" in text
    assert "--candidate-id" in text
    assert "eligible[:2]" in text
    assert "RECOVERY_DIAG_COUNT" in text
    assert "RECOVERY_PASS_COUNT" in text

    acceptance_block = text[text.index('FIRST_BLOCKER=""') : text.index(
        'cat > "$RESULT_DIR/CAMPAIGN_RESULT.txt"'
    )]
    assert "MANIP_PASS_COUNT" in acceptance_block
    assert "MANIP_AUDIT_PASS_COUNT" in acceptance_block
    assert "RECOVERY_PASS_COUNT" not in acceptance_block


def test_manipulation_probe_exposes_diagnostic_candidate_controls() -> None:
    text = MANIP_PROBE.read_text(encoding="utf-8")

    assert '"--b5-evaluate-all-height-groups"' in text
    assert '"--candidate-id"' in text
    assert "stop_after_feasible_height_group=(" in text
    assert "selected_candidate_id=args.candidate_id" in text
    assert '"candidate_selection_diagnostic"' in text
    assert '"carried_asset_paths"' in text
    assert '"carried_start_state_diagnostic"' in text
    assert '"lift_table_disabled_diagnostic"' in text
    assert "diagnose_plan_with_obstacle_disabled" in text
    assert 'obstacle_name="table"' in text
    assert '"collision_panthera.yml"' in text
    assert "asset_path_overrides={" in text


def test_manipulation_campaign_runs_followup_classifier_after_acceptance_decision() -> None:
    text = MANIP_CAMPAIGN.read_text(encoding="utf-8")

    result_marker = 'cat > "$RESULT_DIR/CAMPAIGN_RESULT.txt"'
    classifier_marker = "scripts/classify_native_manipulation_followup.py"
    export_marker = 'cp -a "$RESULT_DIR/." "$EVIDENCE/"'

    assert classifier_marker in text
    assert '"$RESULT_DIR/FOLLOWUP_DECISION.json"' in text
    assert '"$RESULT_DIR/FOLLOWUP_DECISION.txt"' in text
    assert "FOLLOWUP_PRIMARY_DECISION=" in text
    assert text.index(result_marker) < text.index(classifier_marker) < text.index(
        export_marker
    )

    acceptance_block = text[
        text.index('FIRST_BLOCKER=""') : text.index(result_marker)
    ]
    assert "FOLLOWUP" not in acceptance_block


def test_manipulation_campaign_builds_provisional_closure_without_acceptance_bypass() -> None:
    text = MANIP_CAMPAIGN.read_text(encoding="utf-8")

    stage_b = "### Stage B - project-owned manipulation seeds 0,1,2"
    stage_b2 = "### Stage B2 - seed1 deterministic acquisition recovery"
    stage_c = "### Stage C - provisional native closure candidate"
    stage_d = "### Stage D - current readiness snapshot"
    stage_e = "### Stage E - Sprint-1 native closure audit"

    assert text.index(stage_b) < text.index(stage_b2) < text.index(stage_c)
    assert text.index(stage_c) < text.index(stage_d) < text.index(stage_e)
    assert "scripts/build_manipulation_closure_candidate.py" in text
    assert '"$MANIP_PASS_COUNT" -eq 3' in text
    assert '"$MANIP_AUDIT_PASS_COUNT" -eq 3' in text
    assert "--manipulation-candidate" in text
    assert "scripts/audit_sprint1_native_closure.py" in text
    assert "manipulation-closure-candidate.json" in text
    assert "sprint1-native-closure.json" in text
    assert "CLOSURE_CANDIDATE_RC=" in text
    assert "NATIVE_CLOSURE_RC=" in text

    acceptance_block = text[
        text.index('FIRST_BLOCKER=""') : text.index(
            'cat > "$RESULT_DIR/CAMPAIGN_RESULT.txt"'
        )
    ]
    assert "CLOSURE_CANDIDATE_RC" not in acceptance_block
    assert "NATIVE_CLOSURE_RC" not in acceptance_block

    result_block = text[
        text.index('cat > "$RESULT_DIR/CAMPAIGN_RESULT.txt"') : text.index(
            "FOLLOWUP_RC=0"
        )
    ]
    assert "CLOSURE_CANDIDATE_RC=$CLOSURE_CANDIDATE_RC" in result_block
    assert "NATIVE_CLOSURE_RC=$NATIVE_CLOSURE_RC" in result_block


def test_seed1_deterministic_recovery_can_become_effective_seed_only_after_audit() -> None:
    text = MANIP_CAMPAIGN.read_text(encoding="utf-8")

    assert 'SEED1_RECOVERY_ACCEPTED=0' in text
    assert 'SEED1_EFFECTIVE_LABEL="manipulation-seed-1"' in text
    assert 'SEED1_EFFECTIVE_CANDIDATE=""' in text
    assert 'if [[ "$audit_rc" -eq 0 ]]; then' in text
    assert 'SEED1_RECOVERY_ACCEPTED=1' in text
    assert 'SEED1_EFFECTIVE_LABEL="$label"' in text
    assert 'SEED1_EFFECTIVE_CANDIDATE="$candidate_id"' in text
    assert 'MANIP_PASS_COUNT=$((MANIP_PASS_COUNT + 1))' in text
    assert 'MANIP_AUDIT_PASS_COUNT=$((MANIP_AUDIT_PASS_COUNT + 1))' in text
    assert 'if [[ "$SEED1_EFFECTIVE_LABEL" == "$label" ]]; then' in text

    resolution_marker = '"$RESULT_DIR/manipulation-seed-resolution.json"'
    closure_marker = '### Stage C - provisional native closure candidate'
    assert resolution_marker in text
    assert '"schema": "manipulation-seed-resolution-v1"' in text
    assert '"policy": "deterministic-ranked-recovery-v1"' in text
    assert '--seed-resolution "$RESULT_DIR/manipulation-seed-resolution.json"' in text
    assert text.index(resolution_marker) < text.index(closure_marker)

    assert 'SEED1_RECOVERY_ACCEPTED=$SEED1_RECOVERY_ACCEPTED' in text
    assert 'SEED1_EFFECTIVE_LABEL=$SEED1_EFFECTIVE_LABEL' in text
    assert 'SEED1_EFFECTIVE_CANDIDATE=$SEED1_EFFECTIVE_CANDIDATE' in text


def test_seed1_collision_recovery_trigger_ignores_dynamic_safe_false(
    tmp_path: Path,
) -> None:
    import json
    import subprocess
    import sys

    shell = MANIP_CAMPAIGN.read_text(encoding="utf-8")
    marker = 'python3 - "$SEED1_JSON" <<\'PY\''
    predicate = shell.split(marker, 1)[1].split("\nPY", 1)[0]
    assert "target_collision_rejected" in predicate

    approach = {
        "stage": "approach",
        "passed": False,
        "planner_id": "robotwin-mplib-rrt",
        "plan_success": True,
        "dynamic_safe": False,
        "collision_safe": False,
        "execution_status": None,
        "planner_policy": {
            "strategy": "mplib_target_aware_acquisition",
            "primary_success": False,
            "primary_failure_code": "optimization_failed",
            "primary_vendor_status": "screw plan failed",
            "fallback_allowed": True,
            "fallback_attempted": True,
            "fallback_planner_id": "robotwin-mplib-rrt",
            "fallback_success": True,
        },
        "collision_validation": {
            "collisions": [{"matched_environment_roles": ["target"]}],
        },
    }
    path = tmp_path / "seed1.json"

    def evaluate() -> int:
        path.write_text(json.dumps({"stages": [approach]}), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-c", predicate, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert not result.stderr
        return result.returncode

    assert evaluate() == 0
    # Triggering recovery is NOT acceptance of the colliding original.
    assert approach["passed"] is False

    approach["collision_validation"]["collisions"] = []
    assert evaluate() == 1
    approach["collision_validation"]["collisions"] = [
        {"matched_environment_roles": ["target"]}
    ]
    approach["passed"] = True
    assert evaluate() == 1



def test_seed1_recovery_trigger_missing_approach_fails_closed_without_traceback(
    tmp_path: Path,
) -> None:
    import json
    import subprocess
    import sys

    script = MANIP_CAMPAIGN.read_text(encoding="utf-8")
    predicate = script.split('python3 - "$SEED1_JSON" <<\'PY\'', 1)[1].split(
        "\nPY", 1
    )[0]
    path = tmp_path / "incomplete-seed1.json"

    for payload in (
        {"stages": []},
        {"stages": [{"stage": "pregrasp", "passed": True}]},
        {"stages": [None]},
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-c", predicate, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert result.stderr == ""


def test_seed1_expanded_candidate_requires_independent_audit_and_provenance() -> None:
    text = MANIP_CAMPAIGN.read_text(encoding="utf-8")
    start = text.index("expanded_audit_rc=99")
    finish = text.index('### Stage C - provisional native closure candidate')
    block = text[start:finish]
    assert '"$expanded_rc" -eq 0' in block
    assert (
        'python scripts/audit_project_manipulation.py '
        '"$RESULT_DIR/${expanded_label}.json"'
    ) in block
    assert 'if [[ "$expanded_audit_rc" -eq 0 ]]; then' in block
    assert 'expanded.get("passed") is True' in block
    assert 'row.get("eligible") is True' in block
    assert 'selected != original_id' in block
    assert 'if [[ -n "$expanded_candidate" ]]; then' in block
    assert 'SEED1_EFFECTIVE_LABEL="$expanded_label"' in block
    assert 'MANIP_AUDIT_PASS_COUNT=$((MANIP_AUDIT_PASS_COUNT + 1))' in block
    assert '"$SEED1_RECOVERY_ACCEPTED" -eq 0' in block
    assert block.index('if [[ "$expanded_audit_rc" -eq 0 ]]; then') < block.index(
        'SEED1_EFFECTIVE_LABEL="$expanded_label"'
    )
    assert block.index('SEED1_EFFECTIVE_LABEL="$expanded_label"') < block.index(
        'if [[ -f "$RESULT_DIR/${expanded_label}.json" &&'
    )
