"""Replay and audit one checksum-verified native RoboTwin HDF5 camera frame."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from vla_robot_sorting.perception.robotwin_hdf5 import (
    NativeEpisode,
    NativeEpisodeError,
)


def _normalize_manifest_path(value: str) -> str:
    return "/" + value.replace("\\", "/").lstrip("/")


def _verify_export_manifest(
    report_path: Path, episode_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    if not report_path.is_file():
        raise NativeEpisodeError(f"export report does not exist: {report_path}")
    try:
        raw = report_path.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeEpisodeError(f"cannot read export report: {exc}") from exc
    try:
        files_root = (report_path.parent / "files").resolve()
        relative = episode_path.resolve().relative_to(files_root).as_posix()
    except ValueError as exc:
        raise NativeEpisodeError(
            "episode is not under the export report's files directory"
        ) from exc
    suffix = "/files/" + relative
    match = next(
        (
            item
            for item in report.get("files", [])
            if _normalize_manifest_path(str(item.get("exported", ""))).endswith(suffix)
        ),
        None,
    )
    if match is None:
        raise NativeEpisodeError("episode path is not listed in export report")
    if match.get("sha256") != expected_sha256:
        raise NativeEpisodeError("export report SHA256 does not match --expected-sha256")
    if int(match.get("size_bytes", -1)) != episode_path.stat().st_size:
        raise NativeEpisodeError("export report file size does not match the episode")
    return report, sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode one recorded RoboTwin HDF5 RGB frame and validate its CV/OpenGL calibration. "
            "This does not perform detection, metric object localization, or robot control."
        )
    )
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--camera", choices=("head_camera", "wrist_camera"), required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output PNG preview path")
    parser.add_argument("--metadata-output", type=Path, help="defaults to OUTPUT with .json suffix")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        export_report, export_report_sha256 = _verify_export_manifest(
            args.export_report, args.episode, args.expected_sha256
        )
        episode = NativeEpisode(args.episode, expected_sha256=args.expected_sha256)
        audit = episode.audit_camera_conventions()
        frame = episode.read_frame(args.camera, args.frame)
    except (NativeEpisodeError, OSError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.rgb_image.save(args.output, format="PNG")
    preview_sha256 = sha256(args.output.read_bytes()).hexdigest()
    metadata_path = args.metadata_output or args.output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = asdict(frame.provenance)
    provenance["export_classification"] = export_report.get(
        "classification", "classification_missing"
    )
    metadata = {
        "schema": "damh.native-camera-replay.v1",
        "source_classification": "recorded_sim_observation",
        "export_classification": provenance["export_classification"],
        "export_report_path": str(args.export_report.resolve()),
        "export_report_sha256": export_report_sha256,
        "original_source_path": next(
            item.get("original")
            for item in export_report.get("files", [])
            if _normalize_manifest_path(str(item.get("exported", ""))).endswith(
                "/files/"
                + args.episode.resolve()
                .relative_to((args.export_report.parent / "files").resolve())
                .as_posix()
            )
        ),
        "provenance": provenance,
        "frame": {
            "index": frame.frame_index,
            "count": frame.frame_count,
            "camera_id": frame.camera_id,
            "image_encoding_in_hdf5": frame.rgb_encoding,
            "decoded_image_format": "PNG",
            "decoded_pixel_format": frame.rgb_image.mode,
            "width_px": frame.rgb_image.width,
            "height_px": frame.rgb_image.height,
            "rgb_sha256": frame.provenance.rgb_sha256,
            "preview_path": str(args.output.resolve()),
            "preview_sha256": preview_sha256,
        },
        "dataset_inventory": [
            {"path": path, "shape": shape, "dtype": dtype}
            for path, shape, dtype in frame.dataset_inventory
        ],
        "calibration": {
            "intrinsic_cv": frame.intrinsic_matrix,
            "extrinsic_cv_world_to_camera": frame.extrinsic_world_to_camera_cv,
            "cam2world_gl": frame.camera_to_world_gl,
            "camera_to_world_cv": frame.camera_to_world_cv,
            "calibration_sha256": frame.provenance.calibration_sha256,
            "convention_audit": asdict(audit),
        },
        "recorded_robot_fields": {
            "source_datasets": [
                "joint_action/arm",
                "joint_action/gripper",
                "endpose/arm",
            ],
            "joint_action_arm": frame.robot_fields.joint_action_arm,
            "joint_action_gripper": frame.robot_fields.joint_action_gripper,
            "endpose_arm": frame.robot_fields.endpose_arm,
            "field_semantics": (
                "Values retain RoboTwin HDF5 field names; joint_action/arm is not "
                "asserted to be measured joint state, and endpose/arm is robot pose, "
                "not an object pose."
            ),
        },
        "unavailable_evidence": {
            "depth_available": frame.provenance.depth_available,
            "depth_dataset_shape": frame.provenance.depth_dataset_shape,
            "metric_object_pose_available": frame.provenance.object_poses_available,
            "timestamp_s": frame.provenance.timestamp_s,
            "timestamp_source": frame.provenance.timestamp_source,
            "scene_revision": frame.provenance.scene_revision,
            "scene_revision_source": frame.provenance.scene_revision_source,
            "rgb_detection_performed": False,
            "image_derived_metric_localization_performed": False,
            "scene_snapshot_created": False,
            "reason": (
                "The episode supplies RGB and calibration but no depth, object detections, "
                "object poses, scene revision, or observation timestamp; the reader will not "
                "invent these values."
            ),
        },
        "execution": {
            "network_calls": 0,
            "model_calls": 0,
            "simulator_calls": 0,
            "physics_steps": 0,
            "robot_commands_sent": 0,
            "controller_eligible": False,
            "executed": False,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "preview": str(args.output.resolve()),
                "metadata": str(metadata_path.resolve()),
                "camera": frame.camera_id,
                "frame": frame.frame_index,
                "frame_count": frame.frame_count,
                "rgb_size": [frame.rgb_image.width, frame.rgb_image.height],
                "rgb_sha256": frame.provenance.rgb_sha256,
                "cv_gl_max_abs_error": audit.max_abs_error,
                "export_classification": provenance["export_classification"],
                "depth_available": frame.provenance.depth_available,
                "object_pose_available": frame.provenance.object_poses_available,
                "controller_eligible": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
