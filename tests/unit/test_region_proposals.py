from __future__ import annotations

import importlib.util
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from vla_robot_sorting.perception.region_proposals import (
    RegionProposalError,
    build_automatic_detector,
    evaluate_region_proposal_recall,
    generate_rgb_region_proposals,
    validate_bbox_xyxy,
)


def _frame(camera_id="head_camera", frame_index=0, source="recorded_sim_observation"):
    image = Image.new("RGB", (120, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((22, 18, 70, 70), fill=(20, 70, 190))
    draw.ellipse((82, 14, 108, 44), fill=(200, 30, 20))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    rgb_bytes = buffer.getvalue()
    return SimpleNamespace(
        camera_id=camera_id,
        frame_index=frame_index,
        frame_count=2,
        rgb_image=image,
        rgb_bytes=rgb_bytes,
        provenance=SimpleNamespace(
            source_classification=source,
            episode_sha256="a" * 64,
            rgb_sha256=sha256(rgb_bytes).hexdigest(),
            calibration_sha256="c" * 64,
            depth_available=False,
            controller_eligible=False,
        ),
    )


def _manual_annotation(
    frame, box, *, camera_id=None, frame_index=None, source="manual_visual_annotation"
):
    return SimpleNamespace(
        object_id="human-label-01",
        camera_id=camera_id or frame.camera_id,
        frame_index=frame.frame_index if frame_index is None else frame_index,
        image_width_px=frame.rgb_image.width,
        image_height_px=frame.rgb_image.height,
        label="blue bottle-like appearance",
        bbox_xyxy_px=box,
        confidence=None,
        provenance=SimpleNamespace(
            source_kind=source,
            episode_sha256=frame.provenance.episode_sha256,
            rgb_sha256=frame.provenance.rgb_sha256,
            annotation_file_sha256="b" * 64,
        ),
    )


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="optional OpenCV unavailable")
def test_mser_proposals_are_unlabelled_and_bind_to_recorded_rgb():
    frame = _frame()
    result = generate_rgb_region_proposals(frame, max_proposals=100)

    assert result.camera_id == "head_camera"
    assert result.frame_index == 0
    assert result.rgb_sha256 == frame.provenance.rgb_sha256
    assert result.source_classification == "recorded_sim_observation"
    assert result.proposal_method == "opencv_mser_canny_regions"
    assert result.automatic_object_detection is False
    assert result.controller_eligible is False
    assert result.proposals
    for proposal in result.proposals:
        assert 0 <= proposal.bbox_xyxy_px[0] < proposal.bbox_xyxy_px[2] <= 120
        assert 0 <= proposal.bbox_xyxy_px[1] < proposal.bbox_xyxy_px[3] <= 90
        assert proposal.label is None
        assert proposal.object_id is None
        assert proposal.confidence is None
        assert proposal.score_semantics == "unscored_class_agnostic_candidate"


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="optional OpenCV unavailable")
def test_replay_proposal_order_is_deterministic():
    frame = _frame()
    first = generate_rgb_region_proposals(frame, max_proposals=100)
    second = generate_rgb_region_proposals(frame, max_proposals=100)
    assert [p.bbox_xyxy_px for p in first.proposals] == [p.bbox_xyxy_px for p in second.proposals]


@pytest.mark.parametrize(
    "bbox",
    [(-1, 0, 2, 2), (0, 0, 0, 2), (3, 0, 2, 2), (0, 0, 121, 4), (0, 0, 4.0, 4), (False, 0, 4, 4)],
)
def test_bbox_validation_fails_closed(bbox):
    with pytest.raises(RegionProposalError):
        validate_bbox_xyxy(bbox, width=120, height=90)


def test_proposal_input_requires_recorded_provenance_and_matching_rgb_digest():
    frame = _frame(source="synthetic_fixture")
    with pytest.raises(RegionProposalError, match="recorded_sim_observation"):
        generate_rgb_region_proposals(frame)

    frame = _frame()
    frame.provenance.rgb_sha256 = "0" * 64
    with pytest.raises(RegionProposalError, match="RGB checksum"):
        generate_rgb_region_proposals(frame)


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="optional OpenCV unavailable")
def test_recall_evaluation_uses_verified_manual_boxes_and_exact_frame_scope():
    frame = _frame()
    proposals = generate_rgb_region_proposals(frame, max_proposals=100)
    annotation = _manual_annotation(frame, (22, 18, 71, 71))
    score = evaluate_region_proposal_recall([proposals], [annotation])

    assert score["image_count"] == 1
    assert score["annotation_count"] == 1
    assert score["annotation_source_kind"] == "manual_visual_annotation"
    assert score["automatic_detection_evaluated"] is False
    assert score["object_identity_evaluated"] is False
    assert score["controller_eligible"] is False
    assert 0.0 <= score["mean_best_iou"] <= 1.0
    assert score["recall_at_iou_0_25"] in (0.0, 1.0)
    assert score["recall_at_iou_0_50"] in (0.0, 1.0)


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="optional OpenCV unavailable")
def test_evaluation_rejects_unverified_or_mismatched_manual_annotations():
    frame = _frame()
    proposals = generate_rgb_region_proposals(frame, max_proposals=100)
    bad_source = _manual_annotation(frame, (20, 15, 71, 71), source="detector_output")
    with pytest.raises(RegionProposalError, match="manual"):
        evaluate_region_proposal_recall([proposals], [bad_source])

    bad_frame = _manual_annotation(frame, (20, 15, 71, 71), frame_index=1)
    with pytest.raises(RegionProposalError, match="camera/frame"):
        evaluate_region_proposal_recall([proposals], [bad_frame])


def test_unverified_detector_is_never_substituted_for_region_proposals():
    with pytest.raises(RegionProposalError, match="no verified automatic detector"):
        build_automatic_detector("yolov10n", weights_path="cached/model.onnx")


def test_missing_camera_and_calibration_fail_closed():
    frame = _frame()
    frame.camera_id = None
    with pytest.raises(RegionProposalError, match="camera_id"):
        generate_rgb_region_proposals(frame)

    frame = _frame()
    frame.provenance.calibration_sha256 = None
    with pytest.raises(RegionProposalError, match="calibration_sha256"):
        generate_rgb_region_proposals(frame)


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="optional OpenCV unavailable")
def test_manual_annotation_must_match_rgb_checksum():
    frame = _frame()
    proposals = generate_rgb_region_proposals(frame, max_proposals=100)
    annotation = _manual_annotation(frame, (22, 18, 71, 71))
    annotation.provenance.rgb_sha256 = "0" * 64
    with pytest.raises(RegionProposalError, match="checksum"):
        evaluate_region_proposal_recall([proposals], [annotation])
