"""Static/unit checks for optional native operator execution entrypoint."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "scripts" / "linux" / "probe_panthera_project_manipulation.py"


def module():
    spec = importlib.util.spec_from_file_location("native_operator_contract", NATIVE)
    assert spec is not None and spec.loader is not None
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_original_native_invocation_remains_optional(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native", "--robotwin-root", "/robotwin",
            "--project-root", "/project", "--output", "/out",
        ],
    )
    parsed = module().parse_args()
    assert parsed.operator_command is None
    assert parsed.operator_language is None
    assert parsed.operator_rgbd_dir is None
    assert parsed.seed == 0


def test_new_command_requires_explicit_language_and_rgbd_output(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native", "--robotwin-root", "/robotwin",
            "--project-root", "/project", "--output", "/out",
            "--operator-command", "Gắp chai thuốc và đặt lên tấm đệm",
            "--operator-language", "vi",
            "--operator-rgbd-dir", "/unique_native_capture",
        ],
    )
    parsed = module().parse_args()
    assert parsed.operator_language == "vi"
    assert parsed.operator_rgbd_dir.name == "unique_native_capture"
    assert module().validate_operator_entrypoint_args(parsed) == (True, False)


def test_capture_only_with_fresh_rgbd_dir_is_valid_without_operator_command(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native", "--robotwin-root", "/robotwin",
            "--project-root", "/project", "--output", "/out",
            "--operator-capture-only", "--operator-rgbd-dir", "/fresh_capture",
        ],
    )
    parsed = module().parse_args()
    assert module().validate_operator_entrypoint_args(parsed) == (False, True)


def test_capture_only_requires_rgbd_output_dir(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["native", "--robotwin-root", "/r", "--project-root", "/p",
         "--output", "/o", "--operator-capture-only"],
    )
    parsed = module().parse_args()
    with pytest.raises(ValueError, match="capture-only requires a new RGBD output directory"):
        module().validate_operator_entrypoint_args(parsed)


def test_capture_only_and_operator_mode_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["native", "--robotwin-root", "/r", "--project-root", "/p", "--output", "/o",
         "--operator-capture-only", "--operator-rgbd-dir", "/fresh",
         "--operator-language", "en", "--operator-command", "Pick up the pill bottle"],
    )
    parsed = module().parse_args()
    with pytest.raises(
        ValueError, match="capture-only and physical operator mode are mutually exclusive"
    ):
        module().validate_operator_entrypoint_args(parsed)


def test_rgbd_dir_requires_explicit_operator_or_capture_mode(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["native", "--robotwin-root", "/r", "--project-root", "/p", "--output", "/o",
         "--operator-rgbd-dir", "/fresh"],
    )
    parsed = module().parse_args()
    with pytest.raises(
        ValueError,
        match="operator command/language and fresh RGBD output required together",
    ):
        module().validate_operator_entrypoint_args(parsed)


def test_operator_mode_requires_both_command_and_language(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["native", "--robotwin-root", "/r", "--project-root", "/p", "--output", "/o",
         "--operator-command", "command only", "--operator-rgbd-dir", "/fresh"],
    )
    parsed = module().parse_args()
    with pytest.raises(ValueError, match="operator command and language must both be set"):
        module().validate_operator_entrypoint_args(parsed)


def test_invalid_operator_language_rejected(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        [
            "native", "--robotwin-root", "/r", "--project-root", "/p",
            "--output", "/o", "--operator-language", "unknown",
        ],
    )
    with pytest.raises(SystemExit):
        module().parse_args()


def test_full_native_source_python310_parses():
    ast.parse(NATIVE.read_text(encoding="utf-8"), feature_version=(3, 10))


def test_b5_selected_geometry_must_belong_to_grounded_candidates():
    source = NATIVE.read_text(encoding="utf-8")
    assert source.index('payload["operator_command_preflight"]') < source.index(
        "b5 = helpers.evaluate_candidates"
    )
    assert source.index(
        "capture_task_native_pair("
    ) < source.index("b5 = helpers.evaluate_candidates")
    assert "selected.candidate_id not in allowed_ids" in source
    assert "candidate_geometry_binding_passed" in source
    assert "raise RuntimeSafetyStop" in source


def test_source_discloses_rgbd_actor_registry_boundary():
    source = NATIVE.read_text(encoding="utf-8")
    assert "simulator_actor_registry_NOT_RGBD_detection" in source
    assert "explicit_fixed_task_compiler_NOT_GEMINI" in source
    assert "controller_eligible_from_semantic" in source
