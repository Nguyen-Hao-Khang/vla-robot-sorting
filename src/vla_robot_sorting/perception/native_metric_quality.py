"""Independent source-locked native RGB-D quality gates beyond artifact integrity.

These gates certify only basic sensor-value plausibility, NOT geometry accuracy,
object identity, scene alignment, 6DoF pose, or any physical task outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vla_robot_sorting.perception.native_depth_audit import (
    DepthEvidenceError,
    audit_native_depth_bundle,
)
from vla_robot_sorting.perception.native_depth_capture import NATIVE_METRIC_DEPTH_CONVENTION


class NativeMetricQualityError(ValueError):
    """Refuse RGB-D artifacts without defensible metric source provenance."""


def audit_native_metric_quality(directory: str | Path) -> dict:
    """Audit two native camera frames, require source-v2, and check basic scales."""
    root = Path(directory).resolve()
    integrity = audit_native_depth_bundle(root)
    cameras: list[dict] = []
    for camera in ("head_camera", "wrist_camera"):
        name = f"{camera}_frame_000000"
        manifest = json.loads((root / f"{name}_depth-manifest.json").read_text(encoding="utf-8"))
        if manifest.get("depth_convention") != NATIVE_METRIC_DEPTH_CONVENTION:
            raise NativeMetricQualityError(f"{camera}: unverified or legacy metric depth formula")
        depth = np.load(root / f"{name}_depth_mm.npy", allow_pickle=False)
        positive = depth[depth > 0]
        if positive.size < 4:
            raise NativeMetricQualityError(f"{camera}: insufficient positive measured depth")
        median_mm = float(np.median(positive))
        # 50mm floor is conservative policy, not asserted sensor near-clip calibration.
        plausible_fraction = float(np.count_nonzero(positive >= 50.0) / positive.size)
        if median_mm < 50.0 or plausible_fraction < 0.5:
            raise NativeMetricQualityError(f"{camera}: suspicious sub-50mm depth scale")
        with Image.open(root / f"{name}_rgb.png") as img:
            rgb = np.asarray(img.convert("RGBA"), dtype=np.uint8)[..., :3]
        # Explicit low-information flag: the failed source RGB was in 0..1,
        # because HDR/color max >1 skipped conversion to uint8.
        peak_rgb = int(np.max(rgb))
        if peak_rgb <= 3:
            raise NativeMetricQualityError(f"{camera}: suspicious 8-bit RGB range <=3")
        cameras.append(
            {
                "camera_id": camera,
                "positive_depth_pixels": int(positive.size),
                "median_depth_mm": median_mm,
                "fraction_positive_pixels_at_least_50mm": plausible_fraction,
                "rgb_max_8bit": peak_rgb,
                "source_convention": NATIVE_METRIC_DEPTH_CONVENTION,
                "controller_eligible": False,
            }
        )
    return {
        "status": "NATIVE_RGBD_PLAUSIBILITY_PASS_NOT_METRIC_ACCURACY_OR_TASK_ACCEPTANCE",
        "artifact_audit_status": integrity["status"],
        "camera_count": len(cameras),
        "cameras": cameras,
        "object_identity_verified": False,
        "metric_accuracy_verified": False,
        "controller_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        data = audit_native_metric_quality(args.directory)
    except (DepthEvidenceError, NativeMetricQualityError, ValueError, OSError) as exc:
        print(json.dumps({"status": "NATIVE_RGBD_PLAUSIBILITY_REJECTED", "reason": str(exc)}))
        return 2
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
