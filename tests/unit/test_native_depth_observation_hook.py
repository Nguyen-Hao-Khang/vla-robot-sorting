"""Windows-only contracts for a native RGB-D probe; not physics evidence."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.linux.probe_panthera_observation import (
    _select_depth_camera_handles,
    parse_args,
)


def _cameras():
    return SimpleNamespace(
        single_arm_mode=True,
        collect_head_camera=True,
        collect_wrist_camera=True,
        static_camera_name=["head_camera", "front_camera"],
        static_camera_list=[object(), object()],
        wrist_camera=object(),
    )


def test_source_audited_camera_selection_is_single_arm_head_and_wrist():
    cameras = _cameras()
    selected = _select_depth_camera_handles(cameras)
    assert selected == {
        "head_camera": cameras.static_camera_list[0],
        "wrist_camera": cameras.wrist_camera,
    }


@pytest.mark.parametrize("change", [
    {"single_arm_mode": False},
    {"collect_head_camera": False},
    {"collect_wrist_camera": False},
    {"static_camera_name": ["head_camera", "head_camera"]},
    {"static_camera_list": [object()]},
    {"wrist_camera": None},
])
def test_camera_selection_fails_closed_if_source_handles_not_unique(change):
    cameras = _cameras()
    for key, value in change.items():
        setattr(cameras, key, value)
    with pytest.raises(RuntimeError):
        _select_depth_camera_handles(cameras)


def test_probe_depth_is_explicit_opt_in_and_does_not_default_to_capture(monkeypatch):
    args = [
        "observation.py",
        "--robotwin-root", "/source",
        "--project-root", "/project",
        "--output", "/result.json",
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert parse_args().capture_native_depth_dir is None
    monkeypatch.setattr(sys, "argv", [
        *args, "--capture-native-depth-dir", "/probe-depth",
    ])
    assert parse_args().capture_native_depth_dir == Path("/probe-depth")


def test_native_batch_keeps_checksums_diagnostics_and_independent_failure_codes():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/linux/run_seed12_ik_blocker_diagnostic.sh").read_text(
        encoding="utf-8"
    )
    probe = (root / "scripts/linux/probe_panthera_observation.py").read_text(
        encoding="utf-8"
    )
    assert runner.index("sha256sum -c SHA256SUMS.txt") < runner.index("native-depth-probe.log")
    assert runner.index("native-depth-probe.log") < runner.index("for case_id in")
    assert "1200s" in runner
    assert 'printf \'%s\\n\' "$DEPTH_PROBE_RC"' in runner
    assert "3600s" in runner
    assert "sha256sum -c SHA256SUMS.txt > checksum-verification.txt" in runner
    assert "DIAGNOSTIC_ONLY_NOT_ACCEPTED" in runner
    assert "refreshed_camera_ids.count(camera_id) == 1" in probe
    assert "scene_revision=None" in probe
    assert '"controller_eligible": False' in probe
    assert 'return 6' in probe
