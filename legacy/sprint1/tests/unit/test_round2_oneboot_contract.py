"""Source-level safety contract for the source-locked one-boot native coordinator."""

from pathlib import Path


def test_oneboot_diagnostic_to_strict_requires_two_independent_readiness_gates():
    script = (
        Path(__file__).resolve().parents[2] / "scripts/linux/run_sprint1_round2_oneboot.sh"
    ).read_text(encoding="utf-8")
    assert "if (( DEPENDENCY_RC != 0 )); then" in script
    assert "if (( PLAN_RC != 0 )); then" in script
    assert "if (( GATE_RC != 0 )); then" in script
    assert "--additional-recovery-diagnostics" in script
    assert "--accept-after-diagnostics" not in script
    assert "if (( PHYS_RC == 0 && PHYSICAL_GATE_RC == 0 )); then" in script
    assert "--mode strict-acceptance" in script
    assert '--readiness-dir "$PHYSICAL_DIR"' in script
    assert (
        "--additional-recovery-diagnostics"
        not in script.split("--mode strict-acceptance")[1].split("--output-dir")[0]
    )
    assert "--mode physical-diagnostic" in script
    assert (
        script.index("--mode physical-diagnostic")
        < script.index("PHYSICAL_GATE_RC=99")
        < script.index("--mode strict-acceptance")
    )
    assert "STRICT_RC=99" in script
    assert "if (( PHYS_RC != 0 || STRICT_RC != 0 )); then" in script
    assert "exit 40" in script
    assert "exit 0  #" in script


def test_oneboot_never_rewrites_source_or_autoruns_after_failed_preflight():
    script = (
        Path(__file__).resolve().parents[2] / "scripts/linux/run_sprint1_round2_oneboot.sh"
    ).read_text(encoding="utf-8")
    assert "sha256sum -c SHA256SUMS.txt" in script
    assert "git -C" in script
    assert "dependency-diagnostic.stdout.log" in script
    assert "DEPENDENCY_DIAGNOSTIC_BLOCKED_NO_NATIVE_MOTION" in script
    assert "no automatic rerun" in script.lower()
    assert "STAY IN UBUNTU" in script
    assert "reboot Windows" not in script
    for forbidden in ("git reset --hard", "git clean -fd", "fix_joint_limits=True"):
        assert forbidden not in script
