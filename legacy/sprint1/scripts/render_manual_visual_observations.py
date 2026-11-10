"""Render verified manual annotations over exact recorded RoboTwin RGB frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from vla_robot_sorting.perception.robotwin_hdf5 import NativeEpisode, NativeEpisodeError
from vla_robot_sorting.perception.visual_observation import (
    Observation2DError,
    draw_observation_overlay,
    load_manual_annotations,
    observations_json,
    ray_from_pixel,
)


def _verify_export_manifest(
    report_path: Path, episode_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    try:
        raw = report_path.read_bytes()
        report = json.loads(raw)
        files_root = (report_path.parent / "files").resolve()
        relative = episode_path.resolve().relative_to(files_root).as_posix()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise Observation2DError(f"cannot verify export report/episode path: {exc}") from exc
    suffix = "/files/" + relative
    match = next(
        (
            item
            for item in report.get("files", [])
            if str(item.get("exported", "")).replace("\\", "/").endswith(suffix)
        ),
        None,
    )
    if match is None or match.get("sha256") != expected_sha256:
        raise Observation2DError("episode is absent from export manifest or SHA256 differs")
    if int(match.get("size_bytes", -1)) != episode_path.stat().st_size:
        raise Observation2DError("episode size differs from export manifest")
    return report, sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay manual image-space annotations on checksum-verified recorded RGB. "
            "This is not an automatic detector and does not infer 3D pose."
        )
    )
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, report_sha256 = _verify_export_manifest(
            args.export_report, args.episode, args.expected_sha256
        )
        episode = NativeEpisode(args.episode, expected_sha256=args.expected_sha256)
        raw_annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        frame_keys = [
            (item["camera_id"], item["frame_index"]) for item in raw_annotations["frames"]
        ]
        if not frame_keys:
            raise Observation2DError("annotation manifest has no frames")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for camera_id, frame_index in frame_keys:
            frame = episode.read_frame(camera_id, frame_index)
            observations = load_manual_annotations(args.annotations, frame)
            overlay = draw_observation_overlay(frame, observations)
            stem = f"{camera_id}_frame_{frame_index:03d}"
            preview_path = args.output_dir / f"{stem}_manual-overlay.png"
            metadata_path = args.output_dir / f"{stem}_observations.json"
            overlay.save(preview_path, format="PNG")
            metadata = observations_json(frame, observations)
            metadata["camera_rays"] = [
                {
                    "object_id": item.object_id,
                    **asdict(ray_from_pixel(
                        frame,
                        (
                            (item.bbox_xyxy_px[0] + item.bbox_xyxy_px[2]) / 2,
                            (item.bbox_xyxy_px[1] + item.bbox_xyxy_px[3]) / 2,
                        ),
                    )),
                }
                for item in observations
                if item.bbox_xyxy_px is not None
            ]
            metadata.update({
                "export_classification": report.get("classification", "classification_missing"),
                "export_report_sha256": report_sha256,
                "annotation_manifest_path": str(args.annotations.resolve()),
                "preview_path": str(preview_path.resolve()),
                "preview_sha256": sha256(preview_path.read_bytes()).hexdigest(),
                "automatic_detector_used": False,
                "metric_3d_localization_performed": False,
            })
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            outputs.append({
                "camera_id": camera_id,
                "frame_index": frame_index,
                "object_count": len(observations),
                "preview": str(preview_path.resolve()),
                "metadata": str(metadata_path.resolve()),
            })
    except (NativeEpisodeError, Observation2DError, OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": "manual_observation_rendered",
        "source_classification": "recorded_sim_observation",
        "export_classification": report.get("classification", "classification_missing"),
        "automatic_detector_used": False,
        "metric_3d_localization_performed": False,
        "controller_eligible": False,
        "outputs": outputs,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
