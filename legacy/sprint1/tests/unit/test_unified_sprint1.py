"""Regression for a unified command-to-native fail-closed integration boundary."""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from vla_robot_sorting.contracts.grasping import ParallelJawGripper
from vla_robot_sorting.contracts.perception import GeometrySource
from vla_robot_sorting.contracts.semantic import SourceLanguage
from vla_robot_sorting.orchestration.unified_sprint1 import (
    NativeBinding,
    NativeOutcome,
    PipelineStatus,
    UnifiedSprint1Pipeline,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "s1i_fixture", ROOT / "scripts" / "demo_semantic_to_grasp_fixture.py"
)
assert SPEC is not None and SPEC.loader is not None
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
GRIPPER = ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08)
COMMAND = "Put the red block in the left tray"


class Backend:
    def __init__(self, passed: bool = True) -> None:
        self.calls = 0
        self.passed = passed

    def execute_checked(self, proposal, binding):
        self.calls += 1
        assert proposal.target_id == binding.target_id
        return NativeOutcome(self.passed, "complete" if self.passed else "retreat", 1, self.passed)


def context():
    scene = fixture.fixture_scene()
    semantic = fixture.fixture_semantic(COMMAND, SourceLanguage.EN, "red block")
    return dict(
        command=COMMAND,
        language=SourceLanguage.EN,
        scene=scene,
        semantic=semantic,
        semantic_command_sha256=sha256(COMMAND.encode("utf-8")).hexdigest(),
        gripper=GRIPPER,
    )


def native_context():
    args = context()
    scene = args["scene"]
    objects = tuple(
        replace(obj, geometry_source=GeometrySource.SIM_GROUND_TRUTH)
        for obj in scene.objects
    )
    scene = replace(
        scene,
        objects=objects,
        metadata={
            "observation_revision": "fixture-obs-1",
            "observation_source": "sim_ground_truth",
            "identity_provenance": "simulator_native_actor_registry",
            "destination_provenance": "simulator_native_actor_registry",
            "evidence_session": "same-native-session",
        },
    )
    args["scene"] = scene
    binding = NativeBinding(
        "move_pillbottle_pad",
        0,
        scene.scene_revision,
        scene.transform_revision,
        scene.metadata["observation_revision"],
        "same-native-session",
        "red-block",
        "left-tray",
    )
    return args, binding


def test_fixture_does_not_execute_even_with_binding():
    args = context()
    _, binding = native_context()
    backend = Backend()
    report = UnifiedSprint1Pipeline().run(**args, binding=binding, native_backend=backend)
    assert report.status is PipelineStatus.NON_CONTROL_EVIDENCE
    assert report.candidate_count == 12
    assert backend.calls == 0


def test_native_rgbd_without_scene_cannot_authorize():
    args = context()
    args["scene"] = None
    backend = Backend()
    report = UnifiedSprint1Pipeline().run(**args, native_backend=backend)
    assert report.status is PipelineStatus.NON_CONTROL_EVIDENCE
    assert report.stage == "observation"
    assert backend.calls == 0


def test_vietnamese_command_must_match_language():
    args = context()
    args["language"] = SourceLanguage.VI
    report = UnifiedSprint1Pipeline().run(**args)
    assert report.status is PipelineStatus.REJECTED
    assert report.stage == "semantic"


def test_command_modified_after_semantic_rejects():
    args = context()
    args["command"] = "Pick a different object"
    report = UnifiedSprint1Pipeline().run(**args)
    assert report.reason == "command-to-semantic attestation mismatch"


def test_stale_observation_rejects():
    args = context()
    args["scene"] = replace(
        args["scene"],
        metadata={"observation_source": "fixture", "observation_revision": "changed"},
    )
    assert UnifiedSprint1Pipeline().run(**args).status is PipelineStatus.REJECTED


def test_unknown_object_rejects():
    args = context()
    args["scene"] = replace(args["scene"], objects=())
    assert UnifiedSprint1Pipeline().run(**args).status is PipelineStatus.REJECTED


def test_real_native_contract_requires_backend_and_binding():
    args, _ = native_context()
    report = UnifiedSprint1Pipeline().run(**args)
    assert report.status is PipelineStatus.NATIVE_PENDING
    assert report.executed is False


def test_native_binding_session_mismatch_prevents_execution():
    args, binding = native_context()
    backend = Backend()
    report = UnifiedSprint1Pipeline().run(
        **args, binding=replace(binding, evidence_session="other-session"),
        native_backend=backend,
    )
    assert report.status is PipelineStatus.REJECTED
    assert report.stage == "native_binding"
    assert backend.calls == 0


def test_native_binding_target_mismatch_prevents_execution():
    args, binding = native_context()
    backend = Backend()
    report = UnifiedSprint1Pipeline().run(
        **args, binding=replace(binding, target_id="other-target"), native_backend=backend
    )
    assert report.status is PipelineStatus.REJECTED
    assert backend.calls == 0


def test_native_backend_failure_is_not_success():
    args, binding = native_context()
    backend = Backend(False)
    report = UnifiedSprint1Pipeline().run(**args, binding=binding, native_backend=backend)
    assert report.status is PipelineStatus.NATIVE_FAILED
    assert not report.task_success_final
    assert backend.calls == 1


def test_native_backend_success_only_with_verified_binding():
    args, binding = native_context()
    backend = Backend()
    report = UnifiedSprint1Pipeline().run(**args, binding=binding, native_backend=backend)
    assert report.status is PipelineStatus.NATIVE_PASSED
    assert report.task_success_final is True
    assert backend.calls == 1


def test_native_outcome_cannot_fabricate_terminal_success():
    import pytest

    with pytest.raises(ValueError, match="terminal task success"):
        NativeOutcome(True, "release", 1, True)
