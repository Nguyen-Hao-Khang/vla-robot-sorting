"""One same-task RGB-D capture before physical manipulation, never identity authority.

The camera pictures and actor registry share a native RoboTwin task instance.
SAPIEN does not attest simultaneous hardware sensor timestamps; no cross-camera
correspondence or vision-derived target label is asserted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_robot_sorting.perception.native_depth_capture import capture_sapien_depth_frame
from vla_robot_sorting.perception.native_metric_quality import audit_native_metric_quality


def capture_task_native_pair(task: Any, directory: Path) -> dict[str, Any]:
    """Refuse incomplete, repeated or unverifiable camera refresh; no physics actions."""
    output = directory.resolve()
    if output.exists():
        raise ValueError("native capture path must not exist before observation")
    cameras = task.cameras
    if not (
        getattr(cameras, "single_arm_mode", False)
        and getattr(cameras, "collect_head_camera", False)
        and getattr(cameras, "collect_wrist_camera", False)
    ):
        raise ValueError("native single arm head/wrist capture unavailable")
    names = cameras.static_camera_name
    handles = cameras.static_camera_list
    if len(names) != len(handles):
        raise ValueError("camera name/handle pairing malformed")
    heads = [handle for name, handle in zip(names, handles, strict=True)
             if name == "head_camera"]
    wrist = getattr(cameras, "wrist_camera", None)
    if len(heads) != 1 or heads[0] is None or wrist is None:
        raise ValueError("exactly one head and wrist camera required")
    observed = {"head_camera": heads[0], "wrist_camera": wrist}
    calls: list[str] = []
    original_update = cameras.update_picture

    def tracked_update() -> None:
        # Preserve vendor picture acquisition behavior; count only captured frames.
        def watched_take(camera: Any, name: str) -> None:
            camera.take_picture()
            calls.append(name)

        if cameras.collect_wrist_camera:
            watched_take(wrist, "wrist_camera")
        for name, camera in zip(names, handles, strict=True):
            watched_take(camera, name)

    try:
        cameras.update_picture = tracked_update
        task.get_obs()  # no step() or motion command in this adapter
    finally:
        cameras.update_picture = original_update
    if calls.count("head_camera") != 1 or calls.count("wrist_camera") != 1:
        raise ValueError("cameras not refreshed exactly once for same task")
    for camera_id, camera in observed.items():
        manifest_path = capture_sapien_depth_frame(
            camera,
            camera_id=camera_id,
            frame_index=0,
            frame_association_source="probe.get_obs single-frame camera refresh",
            output_dir=output,
            frame_index_verified=True,
            scene_revision=None,
        )
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if raw.get("status") != "capture_written":
            raise ValueError("native RGBD capture rejected: " + camera_id)
    audit = audit_native_metric_quality(output)
    if audit.get("camera_count") != 2 or audit.get("controller_eligible") is not False:
        raise ValueError("native RGBD metric quality rejected")
    return {
        "capture_root": str(output),
        "camera_count": 2,
        "metric_quality_status": audit["status"],
        "source": "same-task RGBD read; actor ID from separate privileged registry",
        "sensor_timestamp_sync_verified": False,
        "visual_object_identity_verified": False,
        "controller_eligible_from_rgbd": False,
    }
