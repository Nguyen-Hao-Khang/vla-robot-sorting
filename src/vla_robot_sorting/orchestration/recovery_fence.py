"""Fail-closed *veto* barrier across scene revisions and recovery steps.

The fence never issues a motor command, certifies IK/collision, or promotes
fixture/replayed data into a live observation. Passing this preflight means
only that downstream independent native safety gates may be considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from vla_robot_sorting.contracts.recovery import (
    RecoveryAction,
    RecoveryBudget,
    RecoveryDecision,
    RecoveryEvent,
    RecoveryState,
)
from vla_robot_sorting.orchestration.recovery import ClosedLoopRecoveryPolicy


class FenceCode(str, Enum):
    DOWNSTREAM_CHECKS_REQUIRED = "downstream_checks_required"
    INVALID_PROVENANCE = "invalid_provenance"
    STALE_OBSERVATION = "stale_observation"
    REVISION_MISMATCH = "revision_mismatch"
    RECOVERY_PENDING = "recovery_pending"
    ABORTED = "aborted"
    BAD_RECOVERY_PROOF = "bad_recovery_proof"
    MISSING_PLAN = "missing_plan"


@dataclass(frozen=True, slots=True)
class FenceVerdict:
    code: FenceCode
    reason: str

    @property
    def downstream_checks_permitted(self) -> bool:
        return self.code is FenceCode.DOWNSTREAM_CHECKS_REQUIRED

    @property
    def motor_command_authorized(self) -> bool:
        # Even a successful revision preflight is NOT native motion acceptance.
        return False


@dataclass(frozen=True, slots=True)
class ObservationVersion:
    """Caller-supplied frame/revision metadata, not pose/identity evidence."""

    scene_revision: str
    transform_revision: str
    observation_revision: str
    observed_at_s: float
    source_kind: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(field, str) or not field.strip()
            for field in (
                self.scene_revision,
                self.transform_revision,
                self.observation_revision,
                self.source_kind,
            )
        ):
            raise ValueError("all observation revision/source fields must be non-empty strings")
        if not isfinite(self.observed_at_s) or self.observed_at_s < 0:
            raise ValueError("observed_at_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RecoveryCompletion:
    """External recovery result; never independently accepted as native proof."""

    action: RecoveryAction
    observation: ObservationVersion
    plan_id: str | None = None


class RecoveryFence:
    """Track pending recovery and veto stale/provenance-invalid execution paths."""

    # Only native-observed, externally measured data may reach later gates.
    # RGB-only/manual and fixture observations cannot authorize motor commands.
    NATIVE_SOURCE = "native_observed"

    def __init__(
        self,
        initial: ObservationVersion,
        *,
        initial_plan_id: str,
        max_observation_age_s: float = 1.0,
        budget: RecoveryBudget | None = None,
    ) -> None:
        if not isfinite(max_observation_age_s) or max_observation_age_s <= 0:
            raise ValueError("max_observation_age_s must be finite and positive")
        if not isinstance(initial_plan_id, str) or not initial_plan_id.strip():
            raise ValueError("initial_plan_id is required and must be non-empty")
        self._observation = initial
        self._max_age_s = max_observation_age_s
        self._policy = ClosedLoopRecoveryPolicy(budget)
        self._state = RecoveryState()
        self._pending: RecoveryAction | None = None
        self._aborted = False
        self._plan_id = initial_plan_id

    @property
    def recovery_state(self) -> RecoveryState:
        return self._state

    @property
    def pending_action(self) -> RecoveryAction | None:
        return self._pending

    @property
    def aborted(self) -> bool:
        return self._aborted

    def preflight(
        self,
        observation: ObservationVersion,
        *,
        now_s: float,
        plan_id: str | None = None,
    ) -> FenceVerdict:
        """Pre-native veto only: a non-veto is not a dispatch authorization."""
        if not isfinite(now_s) or now_s < 0:
            raise ValueError("now_s must be finite and non-negative")
        if self._aborted:
            return FenceVerdict(FenceCode.ABORTED, "sticky aborted session")
        if self._pending is not None:
            return FenceVerdict(FenceCode.RECOVERY_PENDING, "recovery must complete first")
        if observation.source_kind != self.NATIVE_SOURCE:
            return FenceVerdict(
                FenceCode.INVALID_PROVENANCE,
                "fixture/recorded/manual observation is not a live native measurement",
            )
        if (
            observation.scene_revision != self._observation.scene_revision
            or observation.transform_revision != self._observation.transform_revision
            or observation.observation_revision != self._observation.observation_revision
        ):
            return FenceVerdict(
                FenceCode.REVISION_MISMATCH, "version-locked scene/transform mismatch"
            )
        if (
            observation.observed_at_s != self._observation.observed_at_s
            or now_s < observation.observed_at_s
            or now_s - observation.observed_at_s > self._max_age_s
        ):
            return FenceVerdict(
                FenceCode.STALE_OBSERVATION, "native observation is stale or future"
            )
        if not isinstance(plan_id, str) or not plan_id.strip():
            return FenceVerdict(FenceCode.MISSING_PLAN, "plan ID is required for preflight")
        if self._plan_id is not None and plan_id != self._plan_id:
            return FenceVerdict(FenceCode.REVISION_MISMATCH, "new recovery plan ID required")
        return FenceVerdict(
            FenceCode.DOWNSTREAM_CHECKS_REQUIRED,
            "revision preflight only; verify native IK, collision, limits and executor gates",
        )

    def observe_failure(self, event: RecoveryEvent) -> RecoveryDecision:
        """Close the barrier immediately; any later event while pending aborts."""
        if self._aborted or self._pending is not None:
            self._aborted = True
            self._pending = None
            return RecoveryDecision(
                RecoveryAction.ABORT, "new failure while blocked/aborted", self._state
            )
        decision = self._policy.decide(event, self._state)
        self._state = decision.next_state
        if decision.action is RecoveryAction.ABORT:
            self._aborted = True
        elif decision.action is not RecoveryAction.CONTINUE:
            self._pending = decision.action
        return decision

    def complete_recovery(self, proof: RecoveryCompletion) -> FenceVerdict:
        """Clear a barrier only for a matching, fresh, native-classified outcome.

        This is a source/version checkpoint, not an independently audited
        motion or object-retention result.
        """
        if self._aborted:
            return FenceVerdict(FenceCode.ABORTED, "sticky aborted session")
        if self._pending is None or proof.action is not self._pending:
            return FenceVerdict(FenceCode.BAD_RECOVERY_PROOF, "no matching pending recovery")
        updated = proof.observation
        if updated.source_kind != self.NATIVE_SOURCE:
            return FenceVerdict(FenceCode.INVALID_PROVENANCE, "recovery is not native-observed")
        if (
            updated.observed_at_s <= self._observation.observed_at_s
            or updated.observation_revision == self._observation.observation_revision
        ):
            return FenceVerdict(
                FenceCode.BAD_RECOVERY_PROOF, "recovery requires a newly observed revision",
            )
        if not isinstance(proof.plan_id, str) or not proof.plan_id.strip():
            return FenceVerdict(
                FenceCode.BAD_RECOVERY_PROOF, "every recovery needs a newly verified plan ID",
            )
        if proof.plan_id == self._plan_id:
            return FenceVerdict(FenceCode.BAD_RECOVERY_PROOF, "plan was not revised")
        self._observation = updated
        self._plan_id = proof.plan_id
        self._pending = None
        return FenceVerdict(
            FenceCode.DOWNSTREAM_CHECKS_REQUIRED,
            "recovery version accepted for preflight; native safety still unverified",
        )
