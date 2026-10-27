"""Deterministic class-agnostic RGB region proposals, not object detection.

The proposals preserve recorded-frame provenance and carry no class, identity,
confidence, metric pose, or controller permission.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from typing import Any

import numpy as np

from vla_robot_sorting.perception.robotwin_hdf5 import CAMERA_IDS


class RegionProposalError(ValueError):
    """Proposal generation or evidence evaluation failed closed."""


@dataclass(frozen=True, slots=True)
class RegionProposal:
    proposal_id: str
    camera_id: str
    frame_index: int
    bbox_xyxy_px: tuple[int, int, int, int]
    label: None = None
    object_id: None = None
    confidence: None = None
    proposal_score: None = None
    score_semantics: str = "unscored_class_agnostic_candidate"
    source_kind: str = "rgb_region_proposal"


@dataclass(frozen=True, slots=True)
class RegionProposalSet:
    camera_id: str
    frame_index: int
    frame_count: int
    image_width_px: int
    image_height_px: int
    source_classification: str
    episode_sha256: str
    rgb_sha256: str
    calibration_sha256: str
    proposal_method: str
    proposal_method_version: str
    implementation: str
    implementation_version: str
    config_sha256: str
    confidence_semantics: str
    proposals: tuple[RegionProposal, ...]
    automatic_object_detection: bool = False
    object_identity_inferred: bool = False
    metric_pose_available: bool = False
    controller_eligible: bool = False


def _valid_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RegionProposalError(f"{field} must be a lowercase SHA256 digest")
    return value


def validate_bbox_xyxy(bbox: Any, *, width: int, height: int) -> tuple[int, int, int, int]:
    if (
        not isinstance(bbox, (tuple, list))
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
    ):
        raise RegionProposalError("bbox must contain four integer pixel coordinates")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RegionProposalError("bbox is empty or outside image bounds")
    return (x1, y1, x2, y2)


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RegionProposalError(
            "OpenCV is unavailable; install a locally approved CPU OpenCV runtime"
        ) from exc
    return cv2


def _frame_identity(frame: Any) -> tuple[str, int, int, int, int, str, str, str]:
    try:
        camera_id = frame.camera_id
        frame_index = frame.frame_index
        frame_count = frame.frame_count
        image = frame.rgb_image
        width, height = image.width, image.height
        encoded_rgb = frame.rgb_bytes
        provenance = frame.provenance
        source = provenance.source_classification
        episode_sha = _valid_sha256(provenance.episode_sha256, "episode_sha256")
        rgb_sha = _valid_sha256(provenance.rgb_sha256, "rgb_sha256")
        calibration_sha = _valid_sha256(provenance.calibration_sha256, "calibration_sha256")
    except (AttributeError, TypeError, ValueError) as exc:
        if isinstance(exc, RegionProposalError):
            raise
        raise RegionProposalError("recorded RGB frame provenance is incomplete") from exc
    if camera_id not in CAMERA_IDS:
        raise RegionProposalError("unsupported or missing camera_id")
    if type(frame_index) is not int or frame_index < 0 or type(frame_count) is not int:
        raise RegionProposalError("frame index/count must be valid integers")
    if frame_index >= frame_count or width <= 0 or height <= 0:
        raise RegionProposalError("frame index or RGB dimensions are invalid")
    if source != "recorded_sim_observation":
        raise RegionProposalError("source must be recorded_sim_observation")
    if sha256(encoded_rgb).hexdigest() != rgb_sha:
        raise RegionProposalError("RGB checksum does not match recorded frame bytes")
    if image.mode != "RGB":
        raise RegionProposalError("decoded RGB frame must use RGB color mode")
    if (image.width, image.height) != (width, height):
        raise RegionProposalError("decoded RGB dimensions are inconsistent")
    return camera_id, frame_index, frame_count, width, height, episode_sha, rgb_sha, calibration_sha


def generate_rgb_region_proposals(
    frame: Any,
    *,
    max_proposals: int = 500,
    min_area_px: int = 64,
    max_area_fraction: float = 0.75,
) -> RegionProposalSet:
    """Generate deterministic MSER box proposals from one exact recorded RGB frame.

    MSER supplies image regions, not object classes. Proposal confidence and
    object identity are intentionally absent. No model weights are loaded.
    """
    if type(max_proposals) is not int or not 1 <= max_proposals <= 5000:
        raise RegionProposalError("max_proposals must be an integer in [1, 5000]")
    if type(min_area_px) is not int or min_area_px < 1:
        raise RegionProposalError("min_area_px must be a positive integer")
    if (
        isinstance(max_area_fraction, bool)
        or not isinstance(max_area_fraction, (int, float))
        or not isfinite(max_area_fraction)
        or not 0 < max_area_fraction <= 1
    ):
        raise RegionProposalError("max_area_fraction must be finite in (0, 1]")
    try:
        (
            camera_id,
            frame_index,
            frame_count,
            width,
            height,
            episode_sha,
            rgb_sha,
            calibration_sha,
        ) = _frame_identity(frame)
    except ValueError as exc:
        if isinstance(exc, RegionProposalError):
            raise
        raise RegionProposalError("recorded RGB frame provenance is invalid") from exc

    cv2 = _opencv()
    max_area_px = int(width * height * max_area_fraction)
    if max_area_px < min_area_px:
        raise RegionProposalError("area limits exclude every possible proposal")
    rgb = np.asarray(frame.rgb_image, dtype=np.uint8)
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    try:
        mser = cv2.MSER_create(5, min_area_px, max_area_px)
        _regions, boxes = mser.detectRegions(gray)
        edges = cv2.Canny(gray, 50, 150)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except (AttributeError, TypeError, cv2.error) as exc:
        raise RegionProposalError("installed OpenCV does not support MSER/Canny proposals") from exc

    candidates: set[tuple[int, int, int, int]] = set()
    for x, y, box_width, box_height in boxes:
        bbox = (int(x), int(y), int(x + box_width), int(y + box_height))
        try:
            bbox = validate_bbox_xyxy(bbox, width=width, height=height)
        except RegionProposalError:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if min_area_px <= area <= max_area_px:
            candidates.add(bbox)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        bbox = (int(x), int(y), int(x + box_width), int(y + box_height))
        try:
            bbox = validate_bbox_xyxy(bbox, width=width, height=height)
        except RegionProposalError:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if min_area_px <= area <= max_area_px:
            candidates.add(bbox)
    ordered = sorted(candidates, key=lambda b: (b[1], b[0], b[3], b[2]))
    ordered = ordered[:max_proposals]
    proposals = tuple(
        RegionProposal(
            proposal_id=f"{camera_id}-{frame_index:06d}-rp-{index:04d}",
            camera_id=camera_id,
            frame_index=frame_index,
            bbox_xyxy_px=bbox,
        )
        for index, bbox in enumerate(ordered)
    )
    implementation_version = str(cv2.__version__)
    config = {
        "method": "opencv_mser_canny_regions",
        "version": 1,
        "delta": 5,
        "min_area_px": min_area_px,
        "max_area_px": max_area_px,
        "max_proposals": max_proposals,
        "ordering": "y1,x1,y2,x2",
    }
    return RegionProposalSet(
        camera_id=camera_id,
        frame_index=frame_index,
        frame_count=frame_count,
        image_width_px=width,
        image_height_px=height,
        source_classification="recorded_sim_observation",
        episode_sha256=episode_sha,
        rgb_sha256=rgb_sha,
        calibration_sha256=calibration_sha,
        proposal_method="opencv_mser_canny_regions",
        proposal_method_version="1",
        implementation="opencv",
        implementation_version=implementation_version,
        config_sha256=sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        confidence_semantics=(
            "no confidence or calibrated score; boxes are unranked class-agnostic proposals"
        ),
        proposals=proposals,
    )


def box_iou_xyxy(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def evaluate_region_proposal_recall(
    proposal_sets: Sequence[RegionProposalSet],
    annotations: Sequence[Any],
    *,
    thresholds: tuple[float, ...] = (0.25, 0.5),
) -> dict[str, Any]:
    """Measure proposal overlap against checksum-matched manual boxes only."""
    if not proposal_sets:
        raise RegionProposalError("at least one proposal set is required")
    if not annotations:
        raise RegionProposalError("manual annotations are required for evaluation")
    if any(not isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise RegionProposalError("IoU thresholds must be finite values in [0, 1]")
    by_frame: dict[tuple[str, int], RegionProposalSet] = {}
    for proposal_set in proposal_sets:
        key = (proposal_set.camera_id, proposal_set.frame_index)
        if key in by_frame:
            raise RegionProposalError("duplicate camera/frame proposal set")
        by_frame[key] = proposal_set
    groups: dict[tuple[str, int], list[Any]] = {}
    seen_ids: set[tuple[str, int, str]] = set()
    for annotation in annotations:
        try:
            key = (annotation.camera_id, annotation.frame_index)
            source = annotation.provenance.source_kind
            episode_sha = annotation.provenance.episode_sha256
            rgb_sha = annotation.provenance.rgb_sha256
            object_id = annotation.object_id
            bbox = annotation.bbox_xyxy_px
            width, height = annotation.image_width_px, annotation.image_height_px
        except AttributeError as exc:
            raise RegionProposalError("manual annotation provenance is incomplete") from exc
        if source != "manual_visual_annotation":
            raise RegionProposalError("evaluation accepts manual annotations only")
        if key not in by_frame:
            raise RegionProposalError("annotation camera/frame has no matching proposal set")
        proposal_set = by_frame[key]
        if episode_sha != proposal_set.episode_sha256 or rgb_sha != proposal_set.rgb_sha256:
            raise RegionProposalError("annotation source image checksum does not match proposals")
        if (width, height) != (proposal_set.image_width_px, proposal_set.image_height_px):
            raise RegionProposalError("annotation image dimensions do not match proposals")
        bbox = validate_bbox_xyxy(bbox, width=width, height=height)
        identity = (key[0], key[1], object_id)
        if identity in seen_ids:
            raise RegionProposalError("duplicate manual annotation object_id")
        seen_ids.add(identity)
        groups.setdefault(key, []).append((object_id, bbox))

    if set(groups) != set(by_frame):
        raise RegionProposalError("proposal and annotation camera/frame coverage differs")
    best_ious: list[float] = []
    per_annotation: list[dict[str, Any]] = []
    total_proposals = 0
    for key in sorted(groups):
        proposal_set = by_frame[key]
        boxes = [proposal.bbox_xyxy_px for proposal in proposal_set.proposals]
        total_proposals += len(boxes)
        for object_id, gold_box in groups[key]:
            best = max((box_iou_xyxy(gold_box, box) for box in boxes), default=0.0)
            best_ious.append(best)
            per_annotation.append(
                {
                    "camera_id": key[0],
                    "frame_index": key[1],
                    "annotation_id": object_id,
                    "best_proposal_iou": best,
                }
            )
    result: dict[str, Any] = {
        "image_count": len(by_frame),
        "annotation_count": len(best_ious),
        "proposal_count": total_proposals,
        "annotation_source_kind": "manual_visual_annotation",
        "automatic_detection_evaluated": False,
        "object_identity_evaluated": False,
        "mean_best_iou": sum(best_ious) / len(best_ious),
        "per_annotation": per_annotation,
        "controller_eligible": False,
        "claim_boundary": (
            "2D class-agnostic proposal overlap on the exact annotated frames only; "
            "not object detection, identity, segmentation, pose, or controller evidence."
        ),
    }
    for threshold in thresholds:
        name = f"{threshold:.2f}".replace(".", "_")
        result[f"recall_at_iou_{name}"] = sum(value >= threshold for value in best_ious) / len(
            best_ious
        )
    return result


def build_automatic_detector(model_id: str, *, weights_path: str | None = None) -> Any:
    """No automatic detector has been provenance- and runtime-verified for this workstream."""
    del weights_path
    if not isinstance(model_id, str) or not model_id.strip():
        raise RegionProposalError("model_id is required")
    raise RegionProposalError(
        "no verified automatic detector is registered; "
        "region proposals cannot be used as detections"
    )


def proposal_set_to_dict(proposal_set: RegionProposalSet) -> dict[str, Any]:
    return asdict(proposal_set)
