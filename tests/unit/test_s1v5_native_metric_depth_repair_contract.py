"""Contract: S1-V5 follow-up only captures depth and audits physical scale."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/linux/run_s1v5_native_metric_depth_repair.sh"
)


def test_s1v5_runner_is_source_locked_to_older_invalid_evidence():
    data = SCRIPT.read_text(encoding="utf-8")
    assert "s1v4_seed2_alt_20261012T133414Z" in data
    assert "sha256sum -c SHA256SUMS.txt" in data
    assert "PREVIOUS_FAILED_METRIC_SOURCE_VERIFIED" in data
    assert '[[ "$(git rev-parse HEAD)" == "$HEAD" ]]' in data
    assert 'git -C "$ROBOTWIN" rev-parse HEAD' in data
    assert "git status --porcelain --untracked-files=all" in data
    assert "2aee31ad75cb7b0cd779540441c6025853a5b60d" in data
    assert "--capture-native-depth-dir" in data


def test_s1v5_runner_never_executes_manipulation_or_relaxes_collision():
    data = SCRIPT.read_text(encoding="utf-8")
    assert "probe_panthera_project_manipulation.py" not in data
    assert "run_sprint1_manipulation_campaign" not in data
    assert "Gemini" not in data.replace("Gemini API, ", "")
    assert "allow_robot_target_contact=" not in data
    assert "--seed 0" in data
    assert "--seed 1" not in data
    assert "--seed 2" not in data


def test_s1v5_runner_records_audit_outputs_and_export_checksums():
    data = SCRIPT.read_text(encoding="utf-8")
    for expected in (
        'PYTHONPATH="$PROJECT/src"',
        "scripts/audit_native_depth_capture.py",
        "vla_robot_sorting.perception.native_metric_quality",
        "probe-exit.txt",
        "artifact-audit-exit.txt",
        "metric-quality-exit.txt",
        "metric-quality-audit.log",
        "source-sha256.txt",
        "SHA256SUMS.txt",
        "checksum-verification.txt",
        "DIAGNOSTIC_ONLY_NOT_TASK_ACCEPTANCE",
    ):
        assert expected in data
    assert '[[ ! -e "$RESULT" && ! -e "$EXPORT" ]]' in data


def test_s1v5_embedded_python_guards_parse_on_python310():
    text = SCRIPT.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", text, flags=re.DOTALL)
    assert len(blocks) == 2
    for block in blocks:
        ast.parse(block, feature_version=(3, 10))
