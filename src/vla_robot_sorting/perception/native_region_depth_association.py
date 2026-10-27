"""Read-only observed depth association for SAME native RGB-D proposal source.

A rectangle is not an object. The measured 3D pixel is not an object center,
orientation, semantic identity, grasp, or motor-permitted target.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from vla_robot_sorting.perception.metric_pixel import MetricPixelError
from vla_robot_sorting.perception.metric_region import (
    MetricRegionEvidence,
    project_metric_region,
)
from vla_robot_sorting.perception.native_rgbd_proposals import NativeRGBDProposalSet
from vla_robot_sorting.perception.region_proposals import RegionProposalError, validate_bbox_xyxy


@dataclass(frozen=True, slots=True)
class NativeProposalDepthPoint:
    proposal_id: str
    camera_id: str
    frame_index: int
    source_manifest_sha256: str
    source_rgb_sha256: str
    source_depth_sha256: str
    measured_region: MetricRegionEvidence
    geometry_claim: str = "one_observed_pixel_in_unsegmented_RGB_rectangle"
    object_id: None = None
    object_pose_world: None = None
    object_dimensions_m: None = None
    object_orientation: None = None
    grasp_validated: bool = False
    controller_eligible: bool = False


def associate_native_regions_with_observed_depth(
    proposals: NativeRGBDProposalSet,
    native_dir: str | Path,
    *,
    max_regions: int = 16,
) -> tuple[NativeProposalDepthPoint, ...]:
    """Return audited, observed pixel geometry only for bounded candidate boxes.

    No false matches to historical HDF5 images. Rejected/multilayer proposals
    have NO geometry; callers must not substitute nearest/median fake depths.
    """
    if type(max_regions) is not int or not 1 <= max_regions <= 32:
        raise RegionProposalError("max_regions must be in [1, 32]")
    if (
        not isinstance(proposals, NativeRGBDProposalSet)
        or proposals.source_kind != "native_rgbd_single_refresh_region_proposals"
        or proposals.source_classification != "recorded_sim_observation"
        or proposals.source_frame_association != "probe.get_obs single-frame camera refresh"
        or proposals.camera_id not in ("head_camera", "wrist_camera")
        or type(proposals.frame_index) is not int
        or proposals.frame_index != 0
        or proposals.native_depth_metric_quality_audited is not True
        or proposals.verified_hdf5_episode
        or proposals.verified_object_identity
        or proposals.object_pose_world is not None
        or proposals.controller_eligible
    ):
        raise RegionProposalError("native source/claim boundary missing or invalid")

    root = Path(native_dir).resolve()
    prefix = f"{proposals.camera_id}_frame_000000"
    manifest_path = root / f"{prefix}_depth-manifest.json"
    manifest_data = manifest_path.read_bytes()
    if sha256(manifest_data).hexdigest() != proposals.native_depth_manifest_sha256:
        raise RegionProposalError("native depth manifest source hash changed")
    import json

    manifest = json.loads(manifest_data)
    if (
        manifest.get("rgb_sha256") != proposals.rgb_sha256
        or manifest.get("depth_sha256") != proposals.depth_sha256
        or manifest.get("image_size_px") != [proposals.image_width_px, proposals.image_height_px]
        or manifest.get("frame_index") != proposals.frame_index
        or manifest.get("camera_id") != proposals.camera_id
    ):
        raise RegionProposalError("proposal/depth RGB or frame identity mismatch")

    results: list[NativeProposalDepthPoint] = []
    for box in proposals.proposals[:max_regions]:
        if (
            box.source_kind != "native_rgb_region_proposal"
            or box.camera_id != proposals.camera_id
            or box.frame_index != proposals.frame_index
            or box.object_id is not None
            or box.label is not None
            or box.confidence is not None
            or box.proposal_score is not None
        ):
            raise RegionProposalError("proposal claims unverified source, identity, or confidence")
        validate_bbox_xyxy(
            box.bbox_xyxy_px,
            width=proposals.image_width_px,
            height=proposals.image_height_px,
        )
        try:
            evidence = project_metric_region(
                root, camera_id=proposals.camera_id, bbox_xyxy_px=box.bbox_xyxy_px
            )
        except MetricPixelError:
            # Insufficient/multilayer depth is NOT valid geometry; do not
            # fabricate a 3D point or silently reduce the quality threshold.
            continue
        results.append(
            NativeProposalDepthPoint(
                proposal_id=box.proposal_id,
                camera_id=proposals.camera_id,
                frame_index=proposals.frame_index,
                source_manifest_sha256=proposals.native_depth_manifest_sha256,
                source_rgb_sha256=proposals.rgb_sha256,
                source_depth_sha256=proposals.depth_sha256,
                measured_region=evidence,
            )
        )
    return tuple(results)
