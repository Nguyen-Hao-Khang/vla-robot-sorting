# Scoped imported native evidence source; no semantic changes.
"""Read-only, identity-agnostic bridge from same-source native RGB-D evidence.

This module deliberately DOES NOT produce ObjectEvidence or SceneSnapshot.
Native box proposals and masked color pixels do not certify an object ID,
6-DoF pose, object dimensions, a sensor timestamp or actuation authority.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from json import dumps
from math import isfinite
from typing import Any

CAMERAS = ("head_camera", "wrist_camera")
_FRAME_ASSOCIATION = "probe.get_obs single-frame camera refresh"
_SOURCE_KIND = "native_rgbd_single_refresh_region_proposals"
_COLOR_METHOD = "canonical_hsv_color_largest_component_not_object_detector_v1"


class NativeEvidenceBridgeError(ValueError):
    """An unverified source, incompatible frame or inflated claim was supplied."""


def _require(condition: bool, explanation: str) -> None:
    if not condition:
        raise NativeEvidenceBridgeError(explanation)


def _sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {label} SHA256",
    )
    return value


def _nonprivileged(obj: Any, *, landmark: bool = False) -> None:
    empty_fields = ("object_id", "object_pose_world", "object_orientation")
    for key in empty_fields:
        _require(getattr(obj, key, "MISSING") is None, f"{key} is not verified")
    if landmark:
        _require(getattr(obj, "grasp_candidate_validated", None) is False,
                 "landmark cannot validate a grasp")
    else:
        _require(getattr(obj, "object_dimensions_m", "MISSING") is None,
                 "proposal cannot validate object dimensions")
        _require(getattr(obj, "grasp_validated", None) is False,
                 "proposal cannot validate a grasp")
    _require(getattr(obj, "controller_eligible", None) is False,
             "camera-only evidence cannot authorize actuation")


def _json_dataclass(value: Any, name: str) -> dict[str, Any]:
    _require(is_dataclass(value) and not isinstance(value, type),
             f"{name} must be source-typed dataclass evidence")
    result = asdict(value)
    _require(isinstance(result, dict), f"{name} must be a mapping")
    try:
        dumps(result, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise NativeEvidenceBridgeError(f"non-serializable/nonfinite {name}") from exc
    return result


def _bbox(value: Any, width: int, height: int) -> list[int]:
    _require(
        isinstance(value, (tuple, list)) and len(value) == 4
        and all(type(v) is int for v in value),
        "invalid measured bounding box",
    )
    x1, y1, x2, y2 = value
    _require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
             "bounding box exceeds source-native camera frame")
    return [x1, y1, x2, y2]


def build_unidentified_native_evidence(
    proposal_set: Any,
    depth_associations: Iterable[Any],
    color_landmarks: Iterable[Any],
) -> dict[str, Any]:
    """Compose one camera's proposals, observed depth and color surface cues.

    `depth_associations` must be from `associate_native_regions_with_observed_depth`;
    `color_landmarks` must be from `find_native_color_landmarks`. Reject claims
    of object identity, pose or motion eligibility rather than casting them.
    This is a *data boundary*, not an object/semantic grounding algorithm.
    """
    source = proposal_set
    camera = getattr(source, "camera_id", None)
    frame = getattr(source, "frame_index", None)
    _require(camera in CAMERAS, "unsupported native camera")
    _require(type(frame) is int and frame == 0, "wrong native camera frame")
    _require(getattr(source, "source_classification", None) == "recorded_sim_observation",
             "unverified observation source")
    _require(getattr(source, "source_kind", None) == _SOURCE_KIND,
             "unverified proposal source")
    _require(getattr(source, "source_frame_association", None) == _FRAME_ASSOCIATION,
             "unverified camera frame association")
    _require(getattr(source, "native_depth_metric_quality_audited", None) is True,
             "metric depth not independently audited")
    _require(getattr(source, "verified_hdf5_episode", None) is False,
             "cross-session HDF5 fusion forbidden")
    _require(getattr(source, "verified_object_identity", None) is False,
             "identity must not be claimed by native region proposals")
    _require(getattr(source, "object_pose_world", "MISSING") is None,
             "source cannot claim object pose")
    _require(getattr(source, "controller_eligible", None) is False,
             "source cannot authorize motion")
    manifest_sha = _sha(getattr(source, "native_depth_manifest_sha256", None), "manifest")
    rgb_sha = _sha(getattr(source, "rgb_sha256", None), "RGB")
    depth_sha = _sha(getattr(source, "depth_sha256", None), "depth")
    calibration_sha = _sha(
        getattr(source, "calibration_manifest_sha256", None), "calibration"
    )
    producer_sha = _sha(getattr(source, "producer_config_sha256", None), "producer")
    width = getattr(source, "image_width_px", None)
    height = getattr(source, "image_height_px", None)
    _require(type(width) is int and type(height) is int and width > 0 and height > 0,
             "invalid native frame dimensions")

    proposal_boxes: dict[str, list[int]] = {}
    for box in tuple(getattr(source, "proposals", ())):
        _require(getattr(box, "camera_id", None) == camera
                 and getattr(box, "frame_index", None) == frame
                 and getattr(box, "source_kind", None) == "native_rgb_region_proposal",
                 "inconsistent source proposal frame")
        _require(
            all(getattr(box, attr, "MISSING") is None
                for attr in ("object_id", "label", "confidence", "proposal_score")),
            "unverified proposal identity/confidence claim",
        )
        identifier = getattr(box, "proposal_id", None)
        _require(isinstance(identifier, str) and identifier and identifier not in proposal_boxes,
                 "missing/duplicate proposal ID")
        proposal_boxes[identifier] = _bbox(box.bbox_xyxy_px, width, height)

    depth_rows: list[dict[str, Any]] = []
    seen_depth: set[str] = set()
    for point in tuple(depth_associations):
        identifier = getattr(point, "proposal_id", None)
        _require(isinstance(identifier, str) and identifier in proposal_boxes,
                 "depth has no matching source proposal")
        _require(identifier not in seen_depth, "duplicate depth association")
        seen_depth.add(identifier)
        _require(
            getattr(point, "camera_id", None) == camera
            and getattr(point, "frame_index", None) == frame
            and getattr(point, "source_manifest_sha256", None) == manifest_sha
            and getattr(point, "source_rgb_sha256", None) == rgb_sha
            and getattr(point, "source_depth_sha256", None) == depth_sha,
            "depth association source hash/frame mismatch",
        )
        _require(
            getattr(point, "geometry_claim", None)
            == "one_observed_pixel_in_unsegmented_RGB_rectangle",
            "measured region cannot be promoted to object geometry",
        )
        _nonprivileged(point)
        depth_rows.append({
            "proposal_id": identifier,
            "bbox_xyxy_px": proposal_boxes[identifier],
            "geometry_claim": point.geometry_claim,
            "measured_region": _json_dataclass(point.measured_region, "measured_region"),
            "verified_object_identity": False,
            "object_id": None,
            "object_pose_world": None,
            "controller_eligible": False,
        })

    colors: list[dict[str, Any]] = []
    seen_colors: set[str] = set()
    for landmark in tuple(color_landmarks):
        _require(getattr(landmark, "camera_id", None) == camera
                 and getattr(landmark, "source_frame_index", None) == frame,
                 "landmark camera/frame mismatch")
        _require(getattr(landmark, "source_manifest_sha256", None) == manifest_sha
                 and getattr(landmark, "rgb_sha256", None) == rgb_sha
                 and getattr(landmark, "depth_sha256", None) == depth_sha,
                 "landmark source SHA mismatch")
        _require(getattr(landmark, "selection_method", None) == _COLOR_METHOD,
                 "unverified landmark selection method")
        _nonprivileged(landmark, landmark=True)
        kind = getattr(landmark, "landmark_kind", None)
        _require(isinstance(kind, str) and kind and kind not in seen_colors,
                 "missing/duplicate color landmark kind")
        seen_colors.add(kind)
        total = getattr(landmark, "mask_pixel_count", None)
        fraction = getattr(landmark, "valid_depth_fraction", None)
        _require(type(total) is int and total >= 30, "unusable native color mask")
        _require(isinstance(fraction, (int, float)) and isfinite(fraction)
                 and .60 <= fraction <= 1.0, "unusable landmark depth support")
        for key in ("median_masked_depth_m", "masked_depth_iqr_m"):
            number = getattr(landmark, key, None)
            _require(isinstance(number, (int, float)) and isfinite(number) and number >= 0,
                     "invalid masked metric depth")
        _require(landmark.median_masked_depth_m >= 0.05,
                 "masked metric depth must be at least 50mm")
        _require(landmark.masked_depth_iqr_m <= 0.0800001,
                 "masked depth uncertainty exceeds native quality gate")
        colors.append({
            "landmark_kind": kind,
            "selection_method": _COLOR_METHOD,
            "bbox_xyxy_px": _bbox(landmark.bbox_xyxy_px, width, height),
            "mask_pixel_count": total,
            "valid_depth_fraction": float(fraction),
            "median_masked_depth_m": float(landmark.median_masked_depth_m),
            "masked_depth_iqr_m": float(landmark.masked_depth_iqr_m),
            "observed_pixel": _json_dataclass(landmark.observed_pixel, "observed_pixel"),
            "object_id": None,
            "object_pose_world": None,
            "controller_eligible": False,
        })

    result = {
        "schema": "s1i.unidentified-native-visual-evidence.v1",
        "camera_id": camera,
        "frame_index": frame,
        "observation_source": "recorded_native",
        "source_kind": _SOURCE_KIND,
        "source_frame_association": _FRAME_ASSOCIATION,
        "source_manifest_sha256": manifest_sha,
        "rgb_sha256": rgb_sha,
        "depth_sha256": depth_sha,
        "calibration_manifest_sha256": calibration_sha,
        "producer_config_sha256": producer_sha,
        "producer_method": getattr(source, "producer_method", None),
        "producer_version": getattr(source, "producer_version", None),
        "image_size_px": [width, height],
        "proposals_total": len(proposal_boxes),
        "depth_points": depth_rows,
        "color_landmarks": colors,
        "source_timestamp_s": None,
        "scene_revision": None,
        "observation_revision": None,
        "object_identity_verified": False,
        "object_geometry_verified": False,
        "object_orientation_verified": False,
        "scene_snapshot_created": False,
        "semantic_grounding_authorized": False,
        "grasp_planning_authorized": False,
        "controller_eligible": False,
        "reason": (
            "Unsegmented RGB rectangles and color surfaces provide observed pixels, "
            "not verified object identity, dimensions, orientation or a timestamped scene. "
            "Do not construct ObjectEvidence/SceneSnapshot or authorize control."
        ),
    }
    try:
        dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativeEvidenceBridgeError("result contains unverified data types") from exc
    return result


def seal_visual_evidence(
    cameras: Iterable[dict[str, Any]], *, project_git_head: str | None = None
) -> dict[str, Any]:
    """Build a two-camera report; preserve separate camera provenance."""
    rows = tuple(cameras)
    _require(len(rows) == 2, "exactly head and wrist camera reports are required")
    _require({row.get("camera_id") for row in rows} == set(CAMERAS),
             "both head and wrist are required")
    for row in rows:
        _require(row.get("schema") == "s1i.unidentified-native-visual-evidence.v1",
                 "unsupported per-camera evidence schema")
        for gate in (
            "object_identity_verified", "object_geometry_verified", "object_orientation_verified",
            "scene_snapshot_created", "semantic_grounding_authorized",
            "grasp_planning_authorized", "controller_eligible",
        ):
            _require(row.get(gate) is False, f"camera report illegally promoted {gate}")
    if project_git_head is not None:
        _require(isinstance(project_git_head, str) and len(project_git_head) == 40
                 and all(c in "0123456789abcdef" for c in project_git_head),
                 "invalid project source Git SHA")
    _require(
        rows[0].get("source_manifest_sha256") != rows[1].get("source_manifest_sha256"),
        "head and wrist require distinct source camera manifests",
    )
    camera_rows = sorted(rows, key=lambda row: CAMERAS.index(row["camera_id"]))
    manifest = {
        "schema": "s1i.native-visual-evidence-bridge.v1",
        "classification": "SOURCE_VERIFIED_UNIDENTIFIED_PIXELS_ONLY",
        "cameras": camera_rows,
        "camera_count": 2,
        "project_git_head": project_git_head,
        "source_frame_index": 0,
        "cross_camera_timestamp_alignment_proven": False,
        "verified_object_identity": False,
        "scene_snapshot_created": False,
        "semantic_to_grasp_executed": False,
        "native_physics_steps": 0,
        "gemini_api_requests": 0,
        "controller_eligible": False,
        "accepted_native_manipulation_claim": False,
    }
    payload = dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["report_sha256"] = sha256(payload.encode("utf-8")).hexdigest()
    return manifest
