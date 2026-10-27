#!/usr/bin/env python3
"""Create offline class-agnostic RGB proposals and audit them against manual boxes."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import ImageDraw

# Do not accidentally import a DIFFERENT editable DAMH worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_robot_sorting.perception.region_proposals import (
    RegionProposalError,
    evaluate_region_proposal_recall,
    generate_rgb_region_proposals,
    proposal_set_to_dict,
)
from vla_robot_sorting.perception.robotwin_hdf5 import NativeEpisode
from vla_robot_sorting.perception.visual_observation import (
    Observation2DError,
    load_manual_annotations,
)


def _verify_export_report(
    report_path: Path, episode_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    try:
        raw = report_path.read_bytes()
        report = json.loads(raw)
        files_root = (report_path.parent / "files").resolve()
        relative = episode_path.resolve().relative_to(files_root).as_posix()
        records = report["files"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise RegionProposalError("cannot verify export report and episode path") from exc
    if not isinstance(report, dict) or not isinstance(records, list):
        raise RegionProposalError("export report schema is invalid")
    suffix = "/files/" + relative
    matches = [
        item
        for item in records
        if isinstance(item, dict)
        and str(item.get("exported", "")).replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1 or matches[0].get("sha256") != expected_sha256:
        raise RegionProposalError("episode path/SHA256 does not match export report")
    if matches[0].get("size_bytes") != episode_path.stat().st_size:
        raise RegionProposalError("episode size does not match export report")
    return report, sha256(raw).hexdigest()


def _draw_preview(frame, proposals, annotations, path: Path, preview_limit: int) -> str:
    image = frame.rgb_image.copy()
    draw = ImageDraw.Draw(image)
    for proposal in proposals.proposals[:preview_limit]:
        x1, y1, x2, y2 = proposal.bbox_xyxy_px
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 180, 255), width=1)
    for annotation in annotations:
        if annotation.bbox_xyxy_px is None:
            continue
        x1, y1, x2, y2 = annotation.bbox_xyxy_px
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 35, 35), width=2)
        draw.text(
            (x1, max(0, y1 - 12)),
            f"{annotation.label} [manual]",
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    image.save(path, format="PNG")
    return sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate untrained RGB region proposals from the manual-annotation frames "
            "of one checksum-verified RoboTwin episode."
        )
    )
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-proposals", type=int, default=500)
    parser.add_argument("--min-area-px", type=int, default=64)
    parser.add_argument("--max-area-fraction", type=float, default=0.75)
    parser.add_argument("--preview-proposals", type=int, default=60)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    report, report_sha = _verify_export_report(
        args.export_report, args.episode, args.expected_sha256
    )
    output_dir = args.output_dir.resolve()
    immutable_root = args.export_report.parent.resolve()
    if output_dir == immutable_root or immutable_root in output_dir.parents:
        raise RegionProposalError("output directory cannot be inside immutable episode export")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RegionProposalError("output directory is not empty; refusing to overwrite evidence")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode = NativeEpisode(args.episode, expected_sha256=args.expected_sha256)
    calibration_audit = episode.audit_camera_conventions()
    raw_manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict) or not isinstance(raw_manifest.get("frames"), list):
        raise RegionProposalError("annotation manifest frames are invalid")
    frame_keys = [
        (record.get("camera_id"), record.get("frame_index"))
        for record in raw_manifest["frames"]
        if isinstance(record, dict)
    ]
    if len(frame_keys) != len(raw_manifest["frames"]) or not frame_keys:
        raise RegionProposalError("annotation manifest has malformed or no frame records")
    if len(set(frame_keys)) != len(frame_keys):
        raise RegionProposalError("annotation manifest repeats a camera/frame")

    proposal_sets = []
    observations = []
    frame_records = []
    for camera_id, frame_index in frame_keys:
        frame = episode.read_frame(camera_id, frame_index)
        manual = load_manual_annotations(args.annotations, frame)
        if not manual:
            raise RegionProposalError("each evaluated image must have manual box annotations")
        proposals = generate_rgb_region_proposals(
            frame,
            max_proposals=args.max_proposals,
            min_area_px=args.min_area_px,
            max_area_fraction=args.max_area_fraction,
        )
        proposal_sets.append(proposals)
        observations.extend(manual)
        stem = f"{camera_id}_frame_{frame_index:06d}"
        preview_path = output_dir / f"{stem}_proposals-preview.png"
        preview_sha = _draw_preview(frame, proposals, manual, preview_path, args.preview_proposals)
        frame_records.append(
            {
                "camera_id": camera_id,
                "frame_index": frame_index,
                "image_size_px": [frame.rgb_image.width, frame.rgb_image.height],
                "episode_sha256": frame.provenance.episode_sha256,
                "rgb_sha256": frame.provenance.rgb_sha256,
                "calibration_sha256": frame.provenance.calibration_sha256,
                "timestamp_s": frame.provenance.timestamp_s,
                "timestamp_source": frame.provenance.timestamp_source,
                "scene_revision": frame.provenance.scene_revision,
                "scene_revision_source": frame.provenance.scene_revision_source,
                "depth_available_in_hdf5": frame.provenance.depth_available,
                "proposal_set": proposal_set_to_dict(proposals),
                "manual_annotations": [
                    {
                        "object_id": item.object_id,
                        "appearance_label": item.label,
                        "bbox_xyxy_px": item.bbox_xyxy_px,
                        "visibility": item.visibility,
                        "confidence": None,
                        "source_kind": item.provenance.source_kind,
                        "annotation_file_sha256": item.provenance.annotation_file_sha256,
                    }
                    for item in manual
                ],
                "preview_path": preview_path.name,
                "preview_sha256": preview_sha,
                "preview_shows_first_n_proposals": min(
                    args.preview_proposals, len(proposals.proposals)
                ),
            }
        )
    evaluation = evaluate_region_proposal_recall(proposal_sets, observations)
    evidence = {
        "schema": "damh.rgb-region-proposal-evidence.v1",
        "status": "REGION_PROPOSAL_EVALUATION_ONLY",
        "source_classification": "recorded_sim_observation",
        "episode_path": str(args.episode.resolve()),
        "episode_sha256": episode.sha256,
        "episode_size_bytes": args.episode.stat().st_size,
        "export_classification": report.get("classification", "missing"),
        "export_report_path": str(args.export_report.resolve()),
        "export_report_sha256": report_sha,
        "annotation_manifest_path": str(args.annotations.resolve()),
        "annotation_manifest_sha256": sha256(args.annotations.read_bytes()).hexdigest(),
        "annotation_source_kind": raw_manifest.get("source_kind"),
        "annotation_set_id": raw_manifest.get("annotation_set_id"),
        "annotation_method": raw_manifest.get("annotation_method"),
        "camera_convention_audit": {
            "camera_count": calibration_audit.camera_count,
            "frame_count_per_camera": calibration_audit.frame_count_per_camera,
            "checked_transform_count": calibration_audit.checked_transform_count,
            "max_abs_error": calibration_audit.max_abs_error,
            "tolerance": calibration_audit.tolerance,
            "convention": calibration_audit.convention,
        },
        "proposal_engine": {
            "method": "opencv_mser_canny_regions",
            "pretrained_weights_used": False,
            "automatic_object_detection_used": False,
            "object_identity_inferred": False,
            "proposal_confidence": None,
            "score_semantics": "unscored class-agnostic proposal boxes",
        },
        "evaluated_image_count": evaluation["image_count"],
        "manual_annotation_count": evaluation["annotation_count"],
        "evaluation": evaluation,
        "frames": frame_records,
        "depth_data_used": False,
        "metric_object_pose_inferred": False,
        "controller_eligible": False,
        "claim_boundary": (
            "IoU/recall evaluates class-agnostic RGB region coverage against a small, "
            "checksum-matched manual appearance-box set only. No automatic object detection, "
            "identity, segmentation, depth fusion, pose, or robot control is established."
        ),
    }
    out_path = output_dir / "region-proposal-evidence.json"
    # Git enforces LF for *.json. Emit LF bytes even on Windows, otherwise
    # SHA256 in the manifest changes silently during git add/checkout.
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    evidence["evidence_json_path"] = str(out_path)
    evidence["evidence_json_sha256"] = sha256(out_path.read_bytes()).hexdigest()
    # Include preview and evidence digests in a small stable run manifest.
    manifest = {
        "schema": "damh.rgb-region-proposal-run.v1",
        "status": evidence["status"],
        "evidence_json_path": out_path.name,
        "evidence_json_sha256": evidence["evidence_json_sha256"],
        "summary": evaluation,
        "preview_files": [
            {"path": record["preview_path"], "sha256": record["preview_sha256"]}
            for record in frame_records
        ],
        "controller_eligible": False,
    }
    with (output_dir / "run-manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return evidence


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = run(args)
    except (RegionProposalError, Observation2DError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": "damh.rgb-region-proposal-run.v1",
                    "status": "REGION_PROPOSAL_REJECTED",
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                    "controller_eligible": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "schema": "damh.rgb-region-proposal-run.v1",
                "status": evidence["status"],
                "evaluation": evidence["evaluation"],
                "evidence_json_path": evidence["evidence_json_path"],
                "evidence_json_sha256": evidence["evidence_json_sha256"],
                "controller_eligible": False,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
