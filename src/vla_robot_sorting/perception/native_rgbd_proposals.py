"""Offline 2D proposals from ONE checksum-audited native RGB-D frame.

Unlike S1-V6, this consumes SAPIEN native RGB with the depth manifest from
the SAME capture refresh. There is NO HDF5 episode, identity or pose claim.
The depth is audited for basic plausibility but never fused into a grasp goal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

import numpy as np
from PIL import Image

from vla_robot_sorting.perception.native_metric_quality import audit_native_metric_quality
from vla_robot_sorting.perception.region_proposals import (
    RegionProposal,
    RegionProposalError,
    _opencv,
    validate_bbox_xyxy,
)

CAMERAS = ("head_camera", "wrist_camera")


@dataclass(frozen=True, slots=True)
class NativeRGBDProposalSet:
    camera_id: str
    frame_index: int
    source_classification: str
    source_kind: str
    source_frame_association: str
    native_depth_manifest_sha256: str
    rgb_sha256: str
    depth_sha256: str
    calibration_manifest_sha256: str
    image_width_px: int
    image_height_px: int
    producer_method: str
    producer_version: str
    producer_config_sha256: str
    proposals: tuple[RegionProposal, ...]
    native_depth_metric_quality_audited: bool = True
    verified_hdf5_episode: bool = False
    verified_object_identity: bool = False
    object_pose_world: None = None
    controller_eligible: bool = False


def _spatially_balanced_boxes(
    candidates: set[tuple[int, int, int, int]],
    *,
    width: int,
    height: int,
    limit: int,
) -> list[tuple[int, int, int, int]]:
    """Deterministic 4x4 spatial round-robin, without semantic scoring.

    The old y-first sort truncated wrist proposals at the top of the image,
    yielding zero candidates in its lower quadrant. Each occupied tile gets
    one candidate before any tile gets a second candidate.
    """
    buckets: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for box in candidates:
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        key = (min(3, 4 * cy // height), min(3, 4 * cx // width))
        buckets.setdefault(key, []).append(box)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda b: (
                -(b[2] - b[0]) * (b[3] - b[1]),
                b[1],
                b[0],
                b[3],
                b[2],
            )
        )
    selected: list[tuple[int, int, int, int]] = []
    # A bounded round-robin is deterministic and does not invent confidence.
    ordered_keys = sorted(buckets)
    index = 0
    while len(selected) < limit and ordered_keys:
        remaining = []
        for key in ordered_keys:
            if index < len(buckets[key]):
                selected.append(buckets[key][index])
                if len(selected) >= limit:
                    break
            if index + 1 < len(buckets[key]):
                remaining.append(key)
        else:
            ordered_keys = remaining
            index += 1
            continue
        break
    return selected


def generate_native_rgbd_proposals(
    native_dir: str | Path,
    *,
    camera_id: str,
    max_proposals: int = 500,
    min_area_px: int = 64,
    max_area_fraction: float = 0.75,
) -> NativeRGBDProposalSet:
    """Read-only, source-linked RGB proposals, no independent HDF5 association."""
    if camera_id not in CAMERAS:
        raise RegionProposalError("unsupported native camera_id")
    if type(max_proposals) is not int or not 1 <= max_proposals <= 5000:
        raise RegionProposalError("max_proposals must be in [1, 5000]")
    if type(min_area_px) is not int or min_area_px < 1:
        raise RegionProposalError("min_area_px must be a positive integer")
    if (
        type(max_area_fraction) not in (float, int)
        or not isfinite(max_area_fraction)
        or not 0 < max_area_fraction <= 1
    ):
        raise RegionProposalError("max_area_fraction must be finite in (0, 1]")
    root = Path(native_dir).resolve()

    # The independent quality auditor MUST verify TWO cameras, digests, RGB
    # image and metric units. A previous Ubuntu export correctly fails here.
    report = audit_native_metric_quality(root)
    if report["camera_count"] != 2 or not all(
        item["controller_eligible"] is False for item in report["cameras"]
    ):
        raise RegionProposalError("native depth quality is not accepted")

    stem = f"{camera_id}_frame_000000"
    manifest_path = root / f"{stem}_depth-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("camera_id") != camera_id
        or manifest.get("frame_index") != 0
        or manifest.get("frame_association_source") != "probe.get_obs single-frame camera refresh"
    ):
        raise RegionProposalError("native camera/frame source identity is inconsistent")
    width, height = manifest["image_size_px"]
    if type(width) is not int or type(height) is not int:
        raise RegionProposalError("native image dimensions are not integers")
    max_area_px = int(width * height * max_area_fraction)
    if max_area_px < min_area_px:
        raise RegionProposalError("area gates exclude every proposal")
    rgb_file = root / f"{stem}_rgb.png"
    with Image.open(rgb_file) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    cv2 = _opencv()
    try:
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        mser = cv2.MSER_create(5, min_area_px, max_area_px)
        _regions, boxes = mser.detectRegions(gray)
        edges = cv2.Canny(gray, 50, 150)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except (AttributeError, TypeError, cv2.error) as exc:
        raise RegionProposalError("OpenCV MSER/Canny runtime incompatible") from exc
    candidates: set[tuple[int, int, int, int]] = set()
    for x, y, box_width, box_height in boxes:
        try:
            box = validate_bbox_xyxy(
                (int(x), int(y), int(x + box_width), int(y + box_height)),
                width=width,
                height=height,
            )
        except RegionProposalError:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if min_area_px <= area <= max_area_px:
            candidates.add(box)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        try:
            box = validate_bbox_xyxy(
                (int(x), int(y), int(x + box_width), int(y + box_height)),
                width=width,
                height=height,
            )
        except RegionProposalError:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if min_area_px <= area <= max_area_px:
            candidates.add(box)
    ordered = _spatially_balanced_boxes(candidates, width=width, height=height, limit=max_proposals)
    proposal_set = tuple(
        RegionProposal(
            proposal_id=f"native-{camera_id}-000000-rp-{index:04d}",
            camera_id=camera_id,
            frame_index=0,
            bbox_xyxy_px=box,
            source_kind="native_rgb_region_proposal",
        )
        for index, box in enumerate(ordered)
    )
    config = {
        "method": "native_opencv_mser_canny",
        "version": 2,
        "max_proposals": max_proposals,
        "min_area_px": min_area_px,
        "max_area_px": max_area_px,
        "max_area_fraction": max_area_fraction,
        "ordering": "spatial_round_robin_4x4_center_area_desc",
    }
    calibration = {
        key: manifest[key]
        for key in (
            "intrinsic_cv",
            "extrinsic_cv_world_to_camera_cv",
            "cam2world_gl",
            "camera_to_world_cv",
        )
    }
    return NativeRGBDProposalSet(
        camera_id=camera_id,
        frame_index=0,
        source_classification="recorded_sim_observation",
        source_kind="native_rgbd_single_refresh_region_proposals",
        source_frame_association=manifest["frame_association_source"],
        native_depth_manifest_sha256=sha256(manifest_raw).hexdigest(),
        rgb_sha256=manifest["rgb_sha256"],
        depth_sha256=manifest["depth_sha256"],
        calibration_manifest_sha256=sha256(
            json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        image_width_px=width,
        image_height_px=height,
        producer_method=config["method"],
        producer_version=cv2.__version__,
        producer_config_sha256=sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        proposals=proposal_set,
    )


def native_proposals_to_dict(value: NativeRGBDProposalSet) -> dict:
    return asdict(value)
