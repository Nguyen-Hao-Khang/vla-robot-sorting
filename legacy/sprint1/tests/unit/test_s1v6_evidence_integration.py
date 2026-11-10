"""Independent integration checks for copied/committed S1-V6 evidence.

Checks true JSON, relocation-safe local paths, SHA256 and measured proposal
overlap without depending on OpenCV (OpenCV generation is tested separately).
This does not establish automatic object recognition or task acceptance.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _iou(a: list[int], b: list[int]) -> float:
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def test_integrated_proposals_evidence_is_valid_json_and_reproducible_math():
    folder = Path(__file__).resolve().parents[2] / "experiments/evidence/s1v6_rgb_proposals"
    manifest = json.loads((folder / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_json_path"] == "region-proposal-evidence.json"
    source = folder / manifest["evidence_json_path"]
    assert _sha(source) == manifest["evidence_json_sha256"]
    evidence = json.loads(source.read_text(encoding="utf-8"))
    assert evidence["status"] == "REGION_PROPOSAL_EVALUATION_ONLY"
    assert evidence["depth_data_used"] is False
    assert evidence["metric_object_pose_inferred"] is False
    assert evidence["controller_eligible"] is False
    assert evidence["evaluation"]["automatic_detection_evaluated"] is False
    assert evidence["evaluated_image_count"] == 2
    assert evidence["manual_annotation_count"] == 8

    scores: list[float] = []
    count = 0
    for frame in evidence["frames"]:
        assert frame["depth_available_in_hdf5"] is False
        proposals = frame["proposal_set"]["proposals"]
        count += len(proposals)
        for proposal in proposals:
            assert proposal["object_id"] is None
            assert proposal["confidence"] is None
            assert proposal["label"] is None
        for annotation in frame["manual_annotations"]:
            scores.append(
                max(
                    (_iou(annotation["bbox_xyxy_px"], p["bbox_xyxy_px"]) for p in proposals),
                    default=0.0,
                )
            )
    assert count == 780
    assert len(scores) == 8
    summary = evidence["evaluation"]
    assert abs(summary["mean_best_iou"] - sum(scores) / len(scores)) < 1e-12
    assert summary["recall_at_iou_0_25"] == sum(s >= 0.25 for s in scores) / 8
    assert summary["recall_at_iou_0_50"] == sum(s >= 0.5 for s in scores) / 8

    for preview in manifest["preview_files"]:
        assert preview["path"] == Path(preview["path"]).name
        assert _sha(folder / preview["path"]) == preview["sha256"]
