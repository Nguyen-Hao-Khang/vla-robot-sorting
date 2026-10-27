"""A single fail-closed VI/EN -> observation -> grounding -> grasp -> native gate.

Native execution requires a same-session binding and explicit independent physical
safety backend; offline geometry, fixture, and RGBD foreground are never authorized.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Protocol

from vla_robot_sorting.contracts.grasping import ParallelJawGripper
from vla_robot_sorting.contracts.perception import GeometrySource, SceneSnapshot
from vla_robot_sorting.contracts.semantic import SemanticResult, SourceLanguage
from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator
from vla_robot_sorting.orchestration.semantic_to_grasp import (
    GraspProposal,
    SemanticToGraspBridge,
)


class PipelineStatus(str, Enum):
    REJECTED = "rejected"
    NON_CONTROL_EVIDENCE = "non_control_evidence"
    NATIVE_PENDING = "native_pending"
    NATIVE_FAILED = "native_failed"
    NATIVE_PASSED = "native_passed"


@dataclass(frozen=True, slots=True)
class NativeBinding:
    """Native engine-issued same-scene binding, not permission derived from VLM."""

    task: str
    seed: int
    scene_revision: str
    transform_revision: str
    observation_revision: str
    evidence_session: str
    target_id: str
    destination_id: str

    def __post_init__(self) -> None:
        for value in (
            self.task, self.scene_revision, self.transform_revision,
            self.observation_revision, self.evidence_session,
            self.target_id, self.destination_id,
        ):
            if not value.strip():
                raise ValueError("native binding field absent")
        if self.seed not in (0, 1, 2):
            raise ValueError("native binding seed outside fixed accepted scope")


@dataclass(frozen=True, slots=True)
class NativeOutcome:
    passed: bool
    phase: str
    physics_steps: int
    task_success_final: bool

    def __post_init__(self) -> None:
        if not self.phase.strip() or self.physics_steps < 0:
            raise ValueError("invalid native outcome")
        if self.passed and (
            self.phase != "complete" or not self.task_success_final or self.physics_steps == 0
        ):
            raise ValueError("native PASS must have terminal task success")


class NativeSafetyBackend(Protocol):
    """An adapter must run actual IK, FCL/PhysX and task oracle in same native scene."""

    def execute_checked(
        self, proposal: GraspProposal, binding: NativeBinding
    ) -> NativeOutcome: ...


@dataclass(frozen=True, slots=True)
class PipelineReport:
    status: PipelineStatus
    stage: str
    reason: str
    target_id: str | None = None
    destination_id: str | None = None
    candidate_count: int = 0
    candidate_ids: tuple[str, ...] = ()
    executed: bool = False
    task_success_final: bool = False


class UnifiedSprint1Pipeline:
    """Only deterministic orchestration. No fixture-to-native authority transfer."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.5,
        grasp_generator: GeometryFirstGraspGenerator | None = None,
        minimum_table_clearance_m: float = 0.0,
    ) -> None:
        self._bridge = SemanticToGraspBridge(
            minimum_confidence=minimum_confidence,
            grasp_generator=grasp_generator,
            minimum_table_clearance_m=minimum_table_clearance_m,
        )

    def run(
        self,
        *,
        command: str,
        language: SourceLanguage,
        scene: SceneSnapshot | None,
        semantic: SemanticResult | None,
        semantic_command_sha256: str | None,
        gripper: ParallelJawGripper,
        binding: NativeBinding | None = None,
        native_backend: NativeSafetyBackend | None = None,
    ) -> PipelineReport:
        if not command.strip() or language not in (SourceLanguage.VI, SourceLanguage.EN):
            return PipelineReport(
                PipelineStatus.REJECTED, "command", "unsupported command/language"
            )
        if scene is None:
            return PipelineReport(
                PipelineStatus.NON_CONTROL_EVIDENCE, "observation",
                "RGB-D proposals do not establish a versioned identified SceneSnapshot",
            )
        if semantic is None:
            return PipelineReport(
                PipelineStatus.REJECTED, "semantic",
                "no validated semantic result for this observation",
            )
        if semantic.source_language is not language:
            return PipelineReport(
                PipelineStatus.REJECTED, "semantic", "language provenance mismatch",
            )
        if semantic_command_sha256 != sha256(command.encode("utf-8")).hexdigest():
            return PipelineReport(
                PipelineStatus.REJECTED, "semantic", "command-to-semantic attestation mismatch",
            )
        result = self._bridge.propose(semantic, scene, gripper)
        if not result.succeeded:
            assert result.failure_code is not None
            return PipelineReport(
                PipelineStatus.REJECTED, "grounding_or_grasp",
                f"{result.failure_code.value}: {result.reason}",
            )
        assert result.proposal is not None
        proposal = result.proposal
        base = dict(
            target_id=proposal.target_id,
            destination_id=proposal.destination_id,
            candidate_count=len(proposal.candidates),
            candidate_ids=proposal.candidate_ids,
        )
        if scene.metadata.get("observation_source") != "sim_ground_truth":
            return PipelineReport(
                PipelineStatus.NON_CONTROL_EVIDENCE, "vision_identity",
                "fixture/native RGB-D proposal not approved for physical control",
                **base,
            )
        obj = scene.object(proposal.target_id)
        if (
            obj.geometry_source is not GeometrySource.SIM_GROUND_TRUTH
            or scene.metadata.get("identity_provenance") != "simulator_native_actor_registry"
            or scene.metadata.get("destination_provenance") != "simulator_native_actor_registry"
        ):
            return PipelineReport(
                PipelineStatus.NON_CONTROL_EVIDENCE, "vision_identity",
                "missing explicit simulator actor identity and geometry provenance",
                **base,
            )
        if binding is None:
            return PipelineReport(
                PipelineStatus.NATIVE_PENDING, "native_binding",
                "native same-scene binding required",
                **base,
            )
        expected = (
            binding.task == "move_pillbottle_pad"
            and binding.scene_revision == proposal.scene_revision
            and binding.transform_revision == proposal.transform_revision
            and binding.observation_revision == proposal.observation_revision
            and binding.evidence_session == scene.metadata.get("evidence_session")
            and binding.target_id == proposal.target_id
            and binding.destination_id == proposal.destination_id
        )
        if not expected:
            return PipelineReport(
                PipelineStatus.REJECTED, "native_binding",
                "native task/source/target/destination mismatch",
                **base,
            )
        if native_backend is None:
            return PipelineReport(
                PipelineStatus.NATIVE_PENDING, "native_execution",
                "same-scene binding verified; native safety backend not invoked",
                **base,
            )
        outcome = native_backend.execute_checked(proposal, binding)
        if not outcome.passed:
            return PipelineReport(
                PipelineStatus.NATIVE_FAILED, "native_execution",
                f"native execution stopped at {outcome.phase}",
                executed=outcome.physics_steps > 0,
                **base,
            )
        return PipelineReport(
            PipelineStatus.NATIVE_PASSED, "task_verification",
            "native task oracle confirmed terminal success",
            executed=True,
            task_success_final=True,
            **base,
        )
