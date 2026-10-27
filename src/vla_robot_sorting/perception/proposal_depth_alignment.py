"""Prevent unsupported fusion of HDF5 RGB proposals with independent native depth.

Source alignment is STRICTLY more than matching camera ID/pixel dimensions.
The 2026-10-11 HDF5 episode and 2026-10-12 native RGB-D export have
different source episodes and no synchronized frame evidence, so cannot fuse.
Even an aligned ROI remains class-agnostic, not an object pose or robot goal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from vla_robot_sorting.perception.native_depth_audit import DepthEvidenceError
from vla_robot_sorting.perception.native_depth_capture import NATIVE_METRIC_DEPTH_CONVENTION
from vla_robot_sorting.perception.native_metric_quality import (
    NativeMetricQualityError,
    audit_native_metric_quality,
)
from vla_robot_sorting.perception.region_proposals import (
    RegionProposalSet,
    validate_bbox_xyxy,
)


class ProposalDepthAlignmentError(ValueError):
    """No authenticated same-frame 2D/metric-depth association is available."""


@dataclass(frozen=True, slots=True)
class ProposalDepthAlignment:
    camera_id: str
    frame_index: int
    episode_sha256: str
    rgb_sha256: str
    calibration_sha256: str
    native_manifest_sha256: str
    proposal_count: int
    source_alignment_verified: bool = True
    object_identity_verified: bool = False
    metric_object_pose_verified: bool = False
    grasp_candidate_validated: bool = False
    controller_eligible: bool = False


def verify_proposal_depth_alignment(
    proposals: RegionProposalSet,
    native_depth_dir: str | Path,
) -> ProposalDepthAlignment:
    """Verify exact episode/frame/byte+calibration association; fail closed.

    Only verifies permitted geometry-source alignment. Does not project any
    object, infer labels, trust a semantic model or command a robot. Native
    plausibility audit independently verifies BOTH capture camera artifacts.
    """
    if (
        not isinstance(proposals, RegionProposalSet)
        or proposals.source_classification != "recorded_sim_observation"
        or proposals.camera_id not in ("head_camera", "wrist_camera")
        or type(proposals.frame_index) is not int
        or proposals.frame_index != 0
        or type(proposals.frame_count) is not int
        or proposals.frame_count < 1
        or type(proposals.image_width_px) is not int
        or type(proposals.image_height_px) is not int
        or proposals.image_width_px <= 0
        or proposals.image_height_px <= 0
        or not proposals.proposals
        or proposals.automatic_object_detection
        or proposals.object_identity_inferred
        or proposals.metric_pose_available
        or proposals.controller_eligible
    ):
        raise ProposalDepthAlignmentError("proposal source or claim boundary invalid")
    for key in ("episode_sha256", "rgb_sha256", "calibration_sha256"):
        value = getattr(proposals, key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ProposalDepthAlignmentError(f"proposal {key} is not source-authenticated")
    for proposal in proposals.proposals:
        if (
            proposal.camera_id != proposals.camera_id
            or proposal.frame_index != proposals.frame_index
            or proposal.object_id is not None
            or proposal.label is not None
            or proposal.confidence is not None
            or proposal.proposal_score is not None
        ):
            raise ProposalDepthAlignmentError("proposal carries unverified identity or frame")
        validate_bbox_xyxy(
            proposal.bbox_xyxy_px,
            width=proposals.image_width_px,
            height=proposals.image_height_px,
        )

    directory = Path(native_depth_dir).resolve()
    try:
        report = audit_native_metric_quality(directory)
    except (DepthEvidenceError, NativeMetricQualityError, OSError, ValueError) as exc:
        raise ProposalDepthAlignmentError(f"native metric audit failed: {exc}") from exc
    if (
        report.get("status")
        != "NATIVE_RGBD_PLAUSIBILITY_PASS_NOT_METRIC_ACCURACY_OR_TASK_ACCEPTANCE"
    ):
        raise ProposalDepthAlignmentError("independent native metric audit not accepted")

    path = directory / f"{proposals.camera_id}_frame_000000_depth-manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    expected = {
        "camera_id": proposals.camera_id,
        "frame_index": proposals.frame_index,
        "image_size_px": [proposals.image_width_px, proposals.image_height_px],
        "depth_convention": NATIVE_METRIC_DEPTH_CONVENTION,
        "source_episode_sha256": proposals.episode_sha256,
        "source_calibration_sha256": proposals.calibration_sha256,
        "rgb_sha256": proposals.rgb_sha256,
        "source_frame_association": "same_verified_observation_frame",
    }
    for key, value in expected.items():
        if type(manifest.get(key)) is not type(value) or manifest.get(key) != value:
            raise ProposalDepthAlignmentError(
                f"native {key} does not establish the exact source/frame association"
            )
    return ProposalDepthAlignment(
        camera_id=proposals.camera_id,
        frame_index=proposals.frame_index,
        episode_sha256=proposals.episode_sha256,
        rgb_sha256=proposals.rgb_sha256,
        calibration_sha256=proposals.calibration_sha256,
        native_manifest_sha256=sha256(raw).hexdigest(),
        proposal_count=len(proposals.proposals),
    )
