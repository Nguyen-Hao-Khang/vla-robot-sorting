"""Read-only contract for evidence-driven Ubuntu follow-up, never acceptance."""

import ast
import re
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/linux/run_s1v4_seed2_alternative_followup.sh"
)


def test_followup_uses_previous_verified_native_evidence_not_repeat_original():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "s1m_seed12_ik_blockers_20261012T054448Z" in text
    assert '[[ "$(tr -d' in text and 'BASELINE/git-head.txt' in text
    assert 'sha256sum -c SHA256SUMS.txt' in text
    assert 'selected == "pillbottle:cyl:side:2:s-1"' in text
    assert 'alternative != selected and alternative in eligible' in text
    assert 'diag["start_state"]["status"] == "Start state is out of joint limits"' in text
    assert "1.6 <= joint4 <= 1.601" in text
    assert text.index("BASELINE_VERIFIED") < text.index('echo "=== FOLLOW-UP depth')


def test_followup_does_not_replay_failed_seed1_or_original_seed2():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "ALT='pillbottle:cyl:side:3:s-1'" in text
    assert '--seed 2 --b5-evaluate-all-height-groups' in text
    assert '--candidate-id "$ALT" --diagnose-native-ik-blockers' in text
    assert "--seed 1" not in text
    assert 'for case_id in seed1-up75 seed2-canonical' not in text
    assert "Original canonical target, joint limits and collision policy unchanged" in text


def test_followup_source_locked_and_fail_closed_provenance():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "2aee31ad75cb7b0cd779540441c6025853a5b60d" in text
    assert 'git -C "$ROBOTWIN" rev-parse HEAD' in text
    assert '[[ "$(git rev-parse HEAD)" == "$EXPECTED_HEAD" ]]' in text
    assert "git status --porcelain --untracked-files=all" in text
    assert 'b5["selected"]["candidate_id"] == alt' in text
    assert 'row.get("eligible") is True' in text
    assert 'printf \'%s\\n\' "$PROVENANCE_RC"' in text
    assert "DIAGNOSTIC_ONLY_NOT_ACCEPTED" in text
    assert "S1V4_SEED2_ALTERNATIVE_DIAGNOSTIC_EXPORTED" in text


def test_followup_retains_both_native_logs_exit_codes_and_checksums():
    text = SCRIPT.read_text(encoding="utf-8")
    for item in (
        "native-depth-probe-exit.txt",
        "native-depth-audit-exit.txt",
        "seed2-alternative-probe-exit.txt",
        "seed2-alternative-audit-exit.txt",
        "seed2-provenance-exit.txt",
        "native-depth-independent-audit.log",
        "seed2-alternative-independent-audit.log",
        "checksum-verification.txt",
    ):
        assert item in text
    assert 'sha256sum -c SHA256SUMS.txt > checksum-verification.txt' in text
    assert '[[ ! -e "$RESULT" && ! -e "$EXPORT" ]]' in text


def test_all_embedded_python_preflights_parse_with_python310_grammar():
    script = SCRIPT.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", script, flags=re.DOTALL)
    assert len(blocks) == 3
    for block in blocks:
        ast.parse(block, feature_version=(3, 10))
