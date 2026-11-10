"""Operator oneboot must reuse the accepted S1-M strict B5 candidate policy per seed."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/linux/run_sprint1_operator_native_oneboot.sh"
STRICT = ROOT / "scripts/run_final_manipulation_session.py"
UP75 = "pillbottle:cyl:side:0:s-1:up75"


def _runner() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_strict_campaign_seed1_policy_is_the_reference() -> None:
    text = STRICT.read_text(encoding="utf-8")
    anchor = text.index('"--preferred-candidate-id"')
    block = text[text.rindex("if seed == 1:", 0, anchor) :][:400]
    assert '"--b5-evaluate-all-height-groups"' in block
    assert '"--preferred-candidate-id"' in block
    assert f'"{UP75}"' in block


def test_operator_seed1_passes_same_b5_policy() -> None:
    text = _runner()
    match = re.search(r'if \[\[ "\$seed" == "1" \]\]; then\n\s*b5_policy=\((.*?)\)\n', text)
    assert match is not None
    assert match.group(1) == f'--b5-evaluate-all-height-groups --preferred-candidate-id "{UP75}"'
    assert '--operator-rgbd-dir "$rgbd" "${b5_policy[@]}"' in text


def test_policy_is_preference_only_not_override() -> None:
    text = _runner()
    assert "--candidate-id " not in text.replace("--preferred-candidate-id", "")
    assert "local -a b5_policy=()" in text


def test_seed_to_command_mapping_unchanged() -> None:
    text = _runner()
    assert 'run_one en 0 "Pick up the pill bottle and place it on the pad"' in text
    assert 'run_one vi 1 "Gắp chai thuốc và đặt lên tấm đệm"' in text
    assert text.count("run_one en ") == 1
    assert text.count("run_one vi ") == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_policy_array_expands_under_nounset() -> None:
    snippet = (
        "set -Eeuo pipefail\n"
        "f(){ local seed=$1; local -a b5_policy=();"
        ' if [[ "$seed" == "1" ]]; then b5_policy=(--x "a b"); fi;'
        ' printf "<%s>" probe "${b5_policy[@]}"; echo; }\n'
        "f 0; f 1\n"
    )
    out = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, check=True)
    assert out.stdout.splitlines() == ["<probe>", "<probe><--x><a b>"]
