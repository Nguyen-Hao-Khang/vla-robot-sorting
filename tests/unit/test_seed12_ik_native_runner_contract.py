"""Offline contract tests: combined seed1/seed2 native run never grants acceptance."""
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "scripts/linux/run_seed12_ik_blocker_diagnostic.sh"
)


def test_one_bounded_session_runs_only_seed1_selected_up75_and_seed2_canonical():
    script = SOURCE.read_text(encoding="utf-8")
    assert "for case_id in seed1-up75 seed2-canonical; do" in script
    assert "--seed 1 --b5-evaluate-all-height-groups" in script
    assert "--candidate-id 'pillbottle:cyl:side:0:s-1:up75'" in script
    assert "args+=(--seed 2)" in script
    assert "--diagnose-native-ik-blockers" in script
    assert "3600s" in script


def test_immutable_manifest_check_before_sim_and_both_audits_preserved():
    script = SOURCE.read_text(encoding="utf-8")
    assert script.index("sha256sum -c SHA256SUMS.txt") < script.index("for case_id in")
    assert '[[ "$ACTUAL_HEAD" == "$EXPECTED_HEAD" ]]' in script
    assert "git status --porcelain --untracked-files=all" in script
    assert "scripts/audit_project_manipulation.py" in script
    assert "DIAGNOSTIC_ONLY_NOT_ACCEPTED" in script
    assert "checksum-verification.txt" in script
    assert "sha256sum -c SHA256SUMS.txt > checksum-verification.txt" in script
    assert '[[ ! -e "$RESULT" && ! -e "$EXPORT" ]]' in script


def test_runner_retains_native_failures_without_promoting_audit():
    script = SOURCE.read_text(encoding="utf-8")
    assert 'printf \'%s\\n\' "$PROBE_RC"' in script
    assert 'printf \'%s\\n\' "$AUDIT_RC"' in script
    assert 'if [[ -s "$RESULT/${case_id}.json" ]]' in script
    assert "S1M_SEED12_DIAGNOSTIC_EXPORTED" in script
    assert "classification=ACCEPTED" not in script
    assert "PASS_AUDITED" not in script


def test_one_boot_batch_is_pinned_and_failed_json_still_exports_logs():
    script = SOURCE.read_text(encoding="utf-8")
    assert 'ROBOTWIN_PIN="2aee31ad75cb7b0cd779540441c6025853a5b60d"' in script
    assert 'git -C "$ROBOTWIN" rev-parse HEAD' in script
    assert "native-depth-adapter-sha256.txt" in script
    assert "native-depth-auditor-sha256.txt" in script
    assert "observation-probe-sha256.txt" in script
    assert "SUMMARY_PARSE_FAILED" in script
    assert script.index("SUMMARY_PARSE_FAILED") < script.index('cp -a "$RESULT/." "$EXPORT/"')
    assert "DIAGNOSTIC_ONLY_NOT_ACCEPTED" in script


def test_upstream_git_worktree_and_null_candidate_diagnostics_preserve_export():
    script = SOURCE.read_text(encoding="utf-8")
    assert '[[ -d "$PROJECT/.git" && -d "$ROBOTWIN" ]]' in script
    assert '-d "$ROBOTWIN/.git"' not in script
    assert (
        'candidate_id = selected.get("candidate_id") if isinstance(selected, dict) else None'
        in script
    )
    assert '"error_type": "NotObject"' in script
    assert "SUMMARY_PARSE_FAILED" in script
