"""2D object observations and calibrated camera rays from checksum-verified RGB frames.

Manual annotations remain image-space evidence. A ray has direction but no metric
range; neither type is an object pose or controller/grasp authorization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path
from typing import Any


class Observation2DError(ValueError):
    """A visual annotation or calibration record is invalid or mismatched."""


@dataclass(frozen=True, slots=True)
class Observation2DProvenance:
    source_kind: str
    annotation_set_id: str
    annotator_id: str
    annotation_file_sha256: str
    episode_sha256: str
    rgb_sha256: str
    source_image_path: str
    camera_id: str
    frame_index: int


@dataclass(frozen=True, slots=True)
class ObjectObservation2D:
    object_id: str
    camera_id: str
    frame_index: int
    image_width_px: int
    image_height_px: int
    label: str
    bbox_xyxy_px: tuple[int, int, int, int] | None
    mask_rle: dict[str, Any] | None
    visibility: str
    confidence: float | None
    provenance: Observation2DProvenance
    metric_pose_world: None = None
    controller_eligible: bool = False


@dataclass(frozen=True, slots=True)
class CameraRay:
    camera_id: str
    frame_index: int
    pixel_uv: tuple[float, float]
    origin_world: tuple[float, float, float]
    direction_world: tuple[float, float, float]
    metric_depth_m: None = None
    controller_eligible: bool = False


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Observation2DError(f"{name} must be a SHA256 digest")
    if any(char not in "0123456789abcdef" for char in value):
        raise Observation2DError(f"{name} must be lowercase hexadecimal SHA256")
    return value


def _matrix(value: Any, rows: int, cols: int, name: str) -> tuple[tuple[float, ...], ...]:
    try:
        result = tuple(tuple(float(cell) for cell in row) for row in value)
    except (TypeError, ValueError) as exc:
        raise Observation2DError(f"{name} must be a finite {rows}x{cols} matrix") from exc
    if len(result) != rows or any(len(row) != cols for row in result):
        raise Observation2DError(f"{name} must be a finite {rows}x{cols} matrix")
    if not all(isfinite(cell) for row in result for cell in row):
        raise Observation2DError(f"{name} must be a finite {rows}x{cols} matrix")
    return result


def ray_from_pixel(frame: Any, pixel_uv: tuple[float, float]) -> CameraRay:
    """Back-project one pixel through CV intrinsics and camera-to-world calibration."""

    if not getattr(frame, "camera_id", None):
        raise Observation2DError("camera_id is missing")
    try:
        width, height = frame.rgb_image.width, frame.rgb_image.height
        index = frame.frame_index
        u, v = float(pixel_uv[0]), float(pixel_uv[1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise Observation2DError("frame, frame_index, dimensions and pixel are required") from exc
    if width <= 0 or height <= 0 or index < 0:
        raise Observation2DError("frame dimensions and index must be valid")
    if not isfinite(u) or not isfinite(v) or not (0 <= u < width and 0 <= v < height):
        raise Observation2DError("pixel must be finite and inside the recorded image bounds")

    k = _matrix(frame.intrinsic_matrix, 3, 3, "intrinsic_matrix")
    transform = _matrix(frame.camera_to_world_cv, 4, 4, "camera_to_world_cv")
    if k[0][0] <= 0 or k[1][1] <= 0 or abs(k[0][1]) > 1e-8 or abs(k[1][0]) > 1e-8:
        raise Observation2DError("intrinsic_matrix has unsupported focal length or skew")
    if any(abs(transform[3][i] - target) > 1e-6 for i, target in enumerate((0, 0, 0, 1))):
        raise Observation2DError("camera_to_world_cv must be homogeneous")
    rotation = tuple(tuple(transform[row][col] for col in range(3)) for row in range(3))
    for row in range(3):
        if abs(sum(value * value for value in rotation[row]) - 1.0) > 2e-4:
            raise Observation2DError("camera_to_world_cv rotation must be orthonormal")
        for other in range(row):
            if abs(sum(rotation[row][i] * rotation[other][i] for i in range(3))) > 2e-4:
                raise Observation2DError("camera_to_world_cv rotation must be orthonormal")
    det = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(det - 1.0) > 2e-4:
        raise Observation2DError("camera_to_world_cv rotation must be right-handed")

    x = (u - k[0][2]) / k[0][0]
    y = (v - k[1][2]) / k[1][1]
    camera_norm = sqrt(x * x + y * y + 1.0)
    camera_direction = (x / camera_norm, y / camera_norm, 1.0 / camera_norm)
    direction = tuple(
        sum(rotation[row][col] * camera_direction[col] for col in range(3))
        for row in range(3)
    )
    direction_norm = sqrt(sum(value * value for value in direction))
    direction = tuple(value / direction_norm for value in direction)
    return CameraRay(
        camera_id=frame.camera_id,
        frame_index=index,
        pixel_uv=(u, v),
        origin_world=(transform[0][3], transform[1][3], transform[2][3]),
        direction_world=direction,
    )


def load_manual_annotations(path: str | Path, frame: Any) -> tuple[ObjectObservation2D, ...]:
    """Load annotations only when episode, exact RGB bytes, camera, and frame match."""

    annotation_path = Path(path)
    try:
        raw = annotation_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise Observation2DError(f"cannot read annotation manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise Observation2DError("annotation manifest must be a JSON object")
    if manifest.get("schema") != "damh.manual-visual-annotations.v1":
        raise Observation2DError("unsupported annotation schema")
    if manifest.get("source_kind") != "manual_visual_annotation":
        raise Observation2DError("only manual_visual_annotation source is accepted")
    if manifest.get("source_classification") != "recorded_sim_observation":
        raise Observation2DError("annotation source must be a recorded_sim_observation")
    annotation_set_id = manifest.get("annotation_set_id")
    annotator_id = manifest.get("annotator_id")
    if not isinstance(annotation_set_id, str) or not annotation_set_id.strip():
        raise Observation2DError("annotation_set_id is required")
    if not isinstance(annotator_id, str) or not annotator_id.strip():
        raise Observation2DError("annotator_id is required")
    episode_sha = _digest(manifest.get("episode_sha256"), "episode_sha256")
    if episode_sha != frame.provenance.episode_sha256:
        raise Observation2DError("annotation episode SHA256 does not match the RGB frame")
    frame_records = manifest.get("frames")
    if not isinstance(frame_records, list):
        raise Observation2DError("annotation frames must be a list")
    matches = [
        item for item in frame_records
        if isinstance(item, dict)
        and item.get("camera_id") == frame.camera_id
        and item.get("frame_index") == frame.frame_index
    ]
    if len(matches) != 1:
        raise Observation2DError("annotation camera/frame must match exactly once")
    record = matches[0]
    if record.get("image_size_px") != [frame.rgb_image.width, frame.rgb_image.height]:
        raise Observation2DError("annotation dimensions do not match recorded RGB")
    if record.get("rgb_sha256") != frame.provenance.rgb_sha256:
        raise Observation2DError("annotation RGB SHA256 does not match exact recorded bytes")
    annotation_file_sha = sha256(raw).hexdigest()
    width, height = frame.rgb_image.width, frame.rgb_image.height
    raw_objects = record.get("objects", [])
    if not isinstance(raw_objects, list):
        raise Observation2DError("annotation objects must be a list")
    observations: list[ObjectObservation2D] = []
    seen_ids: set[str] = set()
    for item in raw_objects:
        if not isinstance(item, dict):
            raise Observation2DError("each annotated object must be a JSON object")
        object_id, label = item.get("object_id"), item.get("label")
        if not isinstance(object_id, str) or not object_id.strip():
            raise Observation2DError("object_id is required")
        if object_id in seen_ids:
            raise Observation2DError("duplicate object_id in annotation camera/frame")
        seen_ids.add(object_id)
        if not isinstance(label, str) or not label.strip():
            raise Observation2DError("label is required")
        box = item.get("bbox_xyxy_px")
        if box is not None:
            if (
                not isinstance(box, list) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
            ):
                raise Observation2DError("bbox_xyxy_px must be four integer pixel coordinates")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise Observation2DError("bbox_xyxy_px is outside image bounds or empty")
            box = (x1, y1, x2, y2)
        confidence = item.get("confidence")
        if confidence is not None:
            raise Observation2DError(
                "manual annotations cannot carry detector confidence; use null"
            )
        visibility = item.get("visibility", "visible")
        if visibility not in {"visible", "partially_occluded", "occluded"}:
            raise Observation2DError("unsupported visibility value")
        mask = item.get("mask_rle")
        if mask is not None and not isinstance(mask, dict):
            raise Observation2DError("mask_rle must be an object or null")
        provenance = Observation2DProvenance(
            source_kind="manual_visual_annotation",
            annotation_set_id=annotation_set_id,
            annotator_id=annotator_id,
            annotation_file_sha256=annotation_file_sha,
            episode_sha256=episode_sha,
            rgb_sha256=frame.provenance.rgb_sha256,
            source_image_path=frame.provenance.episode_path,
            camera_id=frame.camera_id,
            frame_index=frame.frame_index,
        )
        observations.append(
            ObjectObservation2D(
                object_id=object_id,
                camera_id=frame.camera_id,
                frame_index=frame.frame_index,
                image_width_px=width,
                image_height_px=height,
                label=label,
                bbox_xyxy_px=box,
                mask_rle=mask,
                visibility=visibility,
                confidence=None,
                provenance=provenance,
            )
        )
    return tuple(observations)


def draw_observation_overlay(frame: Any, observations: tuple[ObjectObservation2D, ...]) -> Any:
    """Return an RGB preview with boxes labeled as manual visual annotations."""

    from PIL import ImageDraw

    image = frame.rgb_image.copy()
    draw = ImageDraw.Draw(image)
    for item in observations:
        if item.camera_id != frame.camera_id or item.frame_index != frame.frame_index:
            raise Observation2DError("overlay observation camera/frame mismatch")
        if item.bbox_xyxy_px is not None:
            draw.rectangle(item.bbox_xyxy_px, outline=(255, 40, 40), width=2)
            draw.text(
                (item.bbox_xyxy_px[0], max(0, item.bbox_xyxy_px[1] - 12)),
                f"{item.label} [manual]",
                fill=(255, 255, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
    return image


def observations_json(frame: Any, observations: tuple[ObjectObservation2D, ...]) -> dict[str, Any]:
    """Serialize image-space evidence with explicit non-pose/non-controller status."""

    return {
        "schema": "damh.object-observations-2d.v1",
        "source_classification": "recorded_sim_observation",
        "camera_id": frame.camera_id,
        "frame_index": frame.frame_index,
        "frame_count": frame.frame_count,
        "image_size_px": [frame.rgb_image.width, frame.rgb_image.height],
        "episode_sha256": frame.provenance.episode_sha256,
        "rgb_sha256": frame.provenance.rgb_sha256,
        "calibration_sha256": frame.provenance.calibration_sha256,
        "observation_timestamp_s": frame.provenance.timestamp_s,
        "observation_timestamp_source": frame.provenance.timestamp_source,
        "scene_revision": frame.provenance.scene_revision,
        "scene_revision_source": frame.provenance.scene_revision_source,
        "calibration": {
            "intrinsic_cv": frame.intrinsic_matrix,
            "extrinsic_world_to_camera_cv": frame.extrinsic_world_to_camera_cv,
            "camera_to_world_gl": frame.camera_to_world_gl,
            "camera_to_world_cv": frame.camera_to_world_cv,
        },
        "depth_available": frame.provenance.depth_available,
        "observations": [asdict(item) for item in observations],
        "metric_object_pose_available": False,
        "controller_eligible": False,
        "limitations": [
            "object labels and boxes are manual visual annotations, not detector output",
            "one pixel defines only a ray; no metric depth or object pose is inferred",
            "observations do not authorize grasp planning or robot control",
        ],
    }
