"""Independently audit a source-locked, read-only RoboTwin native RGB-D export.

This is artifact integrity and camera-calibration verification only. It does not
validate object identity, pose, depth accuracy, successful manipulation or robot
control. Raw source must have been captured by Ubuntu SAPIEN; no synthetic
fixture can qualify as native capture evidence.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CAMERAS = ("head_camera", "wrist_camera")
CV_FROM_GL = np.diag((1.0, -1.0, -1.0, 1.0))


class DepthEvidenceError(ValueError):
    """Capture manifest, data, or transform integrity check failed."""


def _fail_if(condition: bool, reason: str) -> None:
    if condition:
        raise DepthEvidenceError(reason)


def _resolve_artifact(directory: Path, recorded_path: Any, suffix: str) -> Path:
    # Native manifests contain Ubuntu absolute paths. Cross-OS bridge exports
    # relocate their flat RGB/depth files to Windows; accept only the exact
    # expected basename, and verify the relocated bytes by SHA256 below.
    _fail_if(not isinstance(recorded_path, str) or not recorded_path, "artifact path is missing")
    recorded_name = recorded_path.replace("\\\\", "/").replace("\\", "/").rsplit("/", 1)[-1]
    _fail_if(recorded_name != suffix, f"artifact basename must be {suffix}")
    path = (directory / suffix).resolve()
    _fail_if(path.parent != directory.resolve(), "artifact escaped the evidence directory")
    _fail_if(not path.is_file(), "artifact does not exist")
    return path


def _check_digest(path: Path, expected: Any) -> str:
    _fail_if(
        not isinstance(expected, str)
        or len(expected) != 64
        or any(ch not in "0123456789abcdef" for ch in expected),
        "invalid SHA256 digest",
    )
    actual = sha256(path.read_bytes()).hexdigest()
    _fail_if(actual != expected, f"SHA256 mismatch: {path.name}")
    return actual


def _finite_matrix(record: dict[str, Any], key: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        matrix = np.asarray(record[key], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise DepthEvidenceError(f"invalid calibration {key}") from exc
    _fail_if(matrix.shape != shape or not np.isfinite(matrix).all(), f"invalid calibration {key}")
    return matrix


def audit_native_depth_bundle(directory: Path) -> dict[str, Any]:
    """Verify both camera captures independently; raise on any missing/false evidence."""

    directory = directory.resolve()
    _fail_if(not directory.is_dir(), "native depth evidence directory is missing")
    existing = sorted(directory.glob("*_depth-manifest.json"))
    _fail_if(len(existing) != 2, "expected exactly two depth manifests")
    result: list[dict[str, Any]] = []
    for camera_id in CAMERAS:
        name = f"{camera_id}_frame_000000"
        manifest_path = directory / f"{name}_depth-manifest.json"
        _fail_if(not manifest_path.is_file(), f"{camera_id} manifest missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DepthEvidenceError(f"cannot read {camera_id} manifest") from exc
        _fail_if(not isinstance(manifest, dict), "manifest must be an object")
        expected_values = {
            "schema": "damh.robotwin-native-rgbd-capture.v1",
            "status": "capture_written",
            "source_classification": "recorded_sim_observation",
            "camera_id": camera_id,
            "frame_index": 0,
            "frame_index_verified_by_caller": True,
            "native_capture_executed": True,
            "controller_eligible": False,
            "depth_units": "millimetres",
            "pinned_sapien_version": "3.0.0b1",
            "installed_sapien_version": "3.0.0b1",
        }
        for key, value in expected_values.items():
            _fail_if(
                type(manifest.get(key)) is not type(value) or manifest.get(key) != value,
                f"{camera_id}: wrong/missing {key}",
            )
        _fail_if(
            manifest.get("frame_association_source")
            != "probe.get_obs single-frame camera refresh",
            f"{camera_id}: unverified frame association provenance",
        )
        _fail_if(
            manifest.get("timestamp_semantics")
            != "probe wall-clock time; not a simulator sensor timestamp",
            f"{camera_id}: timestamp claim is unsupported",
        )
        _fail_if(
            manifest.get("scene_revision") is not None,
            "unknown scene revision cannot be invented",
        )

        rgb_path = _resolve_artifact(directory, manifest.get("rgb_path"), f"{name}_rgb.png")
        depth_path = _resolve_artifact(
            directory, manifest.get("depth_path"), f"{name}_depth_mm.npy"
        )
        rgb_digest = _check_digest(rgb_path, manifest.get("rgb_sha256"))
        depth_digest = _check_digest(depth_path, manifest.get("depth_sha256"))

        with Image.open(rgb_path) as image:
            _fail_if(image.format != "PNG" or image.mode != "RGBA", "RGB image format invalid")
            image_size = image.size
        try:
            depth = np.load(depth_path, allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise DepthEvidenceError("cannot parse numpy depth artifact") from exc
        _fail_if(depth.ndim != 2 or depth.dtype != np.float64, "depth must be 2D float64")
        _fail_if(
            not np.isfinite(depth).all() or np.any(depth < 0),
            "depth has negative or nonfinite millimetres",
        )
        _fail_if(depth.shape != image_size[::-1], "RGB/depth dimensions differ")
        valid_count = int(np.count_nonzero(depth > 0))
        _fail_if(valid_count <= 0, "no positive depth pixels")
        _fail_if(manifest.get("image_size_px") != list(image_size), "wrong image_size_px")
        _fail_if(manifest.get("depth_shape") != list(depth.shape), "wrong depth_shape")
        _fail_if(
            manifest.get("depth_valid_pixel_count") != valid_count,
            "depth pixel count mismatch",
        )
        intrinsic = _finite_matrix(manifest, "intrinsic_cv", (3, 3))
        extrinsic = _finite_matrix(manifest, "extrinsic_cv_world_to_camera_cv", (3, 4))
        model = _finite_matrix(manifest, "cam2world_gl", (4, 4))
        cv = _finite_matrix(manifest, "camera_to_world_cv", (4, 4))
        _fail_if(intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0, "invalid focal lengths")
        _fail_if(not np.allclose(model[3], [0, 0, 0, 1], atol=1e-6), "invalid GL pose")
        world_to_cv = np.vstack((extrinsic, [0, 0, 0, 1]))
        try:
            independent_cv = np.linalg.inv(world_to_cv)
        except np.linalg.LinAlgError as exc:
            raise DepthEvidenceError("extrinsic is singular") from exc
        err = max(
            float(np.max(np.abs(independent_cv - model @ CV_FROM_GL))),
            float(np.max(np.abs(cv - independent_cv))),
        )
        _fail_if(err > 2e-4, "camera GL/CV extrinsic mismatch")
        _fail_if(
            not isinstance(manifest.get("cv_gl_max_abs_error"), (int, float))
            or abs(float(manifest["cv_gl_max_abs_error"]) - err) > 2e-4,
            "recorded calibration error mismatch",
        )
        result.append({
            "camera_id": camera_id,
            "frame_index": 0,
            "rgb_sha256": rgb_digest,
            "depth_sha256": depth_digest,
            "positive_depth_pixels": valid_count,
            "image_size_px": list(image_size),
            "independent_calibration_max_error": err,
            "controller_eligible": False,
        })
    return {
        "status": "ARTIFACT_INTEGRITY_PASS_NOT_TASK_ACCEPTANCE",
        "scope": "read_only_native_RGBD_artifact_audit",
        "native_physics_or_task_success_verified": False,
        "camera_count": len(result),
        "cameras": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Native depth evidence directory from Ubuntu")
    args = parser.parse_args()
    try:
        result = audit_native_depth_bundle(args.directory)
    except (DepthEvidenceError, OSError) as exc:
        print(json.dumps({"status": "ARTIFACT_INTEGRITY_FAIL", "reason": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
