from __future__ import annotations

from types import SimpleNamespace

import pytest

from vla_robot_sorting.perception.visual_observation import (
    Observation2DError,
    load_manual_annotations,
    observations_json,
    ray_from_pixel,
)


def _frame(*, camera_id: str = "head_camera", frame_index: int = 0):
    return SimpleNamespace(
        camera_id=camera_id,
        frame_index=frame_index,
        frame_count=3,
        rgb_image=SimpleNamespace(width=100, height=80),
        intrinsic_matrix=((100.0, 0.0, 50.0), (0.0, 100.0, 40.0), (0.0, 0.0, 1.0)),
        extrinsic_world_to_camera_cv=(
            (0.0, 1.0, 0.0, -2.0),
            (-1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, -3.0),
        ),
        camera_to_world_gl=(
            (0.0, 1.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0, -2.0),
            (0.0, 0.0, -1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        camera_to_world_cv=(
            (0.0, -1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        provenance=SimpleNamespace(
            episode_sha256="a" * 64,
            rgb_sha256="b" * 64,
            calibration_sha256="c" * 64,
            episode_path="recorded/episode0.hdf5",
            timestamp_s=None,
            timestamp_source=None,
            scene_revision=None,
            scene_revision_source=None,
            depth_available=False,
        ),
    )


def test_ray_from_pixel_returns_world_ray_without_metric_depth():
    ray = ray_from_pixel(_frame(), (50.0, 40.0))

    assert ray.origin_world == (1.0, 2.0, 3.0)
    assert ray.direction_world == pytest.approx((0.0, 0.0, 1.0))
    assert ray.metric_depth_m is None
    assert ray.controller_eligible is False


@pytest.mark.parametrize("pixel", [(-0.1, 20), (100, 20), (20, 80), (float("nan"), 4)])
def test_ray_rejects_invalid_or_out_of_frame_pixels(pixel):
    with pytest.raises(Observation2DError):
        ray_from_pixel(_frame(), pixel)


def test_ray_rejects_nonfinite_or_malformed_calibration():
    frame = _frame()
    frame.camera_to_world_cv = ((1, 0), (0, 1))

    with pytest.raises(Observation2DError, match="camera_to_world"):
        ray_from_pixel(frame, (10, 10))


def test_annotations_require_exact_camera_frame_and_image_digest(tmp_path):
    frame = _frame()
    manifest = {
        "schema": "damh.manual-visual-annotations.v1",
        "annotation_set_id": "fixture-annotations-01",
        "annotator_id": "assistant_visual_review",
        "source_kind": "manual_visual_annotation",
        "source_classification": "recorded_sim_observation",
        "episode_sha256": frame.provenance.episode_sha256,
        "frames": [{
            "camera_id": "head_camera",
            "frame_index": 0,
            "rgb_sha256": frame.provenance.rgb_sha256,
            "image_size_px": [100, 80],
            "objects": [{
                "object_id": "visual-01",
                "label": "blue bottle-like object",
                "bbox_xyxy_px": [10, 12, 30, 44],
                "visibility": "visible",
                "confidence": None,
                "mask_rle": None,
            }],
        }],
    }
    path = tmp_path / "annotations.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    annotations = load_manual_annotations(path, frame)

    assert len(annotations) == 1
    assert annotations[0].confidence is None
    assert annotations[0].provenance.source_kind == "manual_visual_annotation"
    assert annotations[0].bbox_xyxy_px == (10, 12, 30, 44)


def test_annotations_reject_mismatched_frame_and_unverified_source(tmp_path):
    frame = _frame()
    base = {
        "schema": "damh.manual-visual-annotations.v1",
        "annotation_set_id": "x",
        "annotator_id": "assistant_visual_review",
        "source_kind": "manual_visual_annotation",
        "source_classification": "recorded_sim_observation",
        "episode_sha256": frame.provenance.episode_sha256,
        "frames": [{
            "camera_id": "head_camera",
            "frame_index": 1,
            "rgb_sha256": frame.provenance.rgb_sha256,
            "objects": [],
        }],
    }
    path = tmp_path / "annotations.json"
    path.write_text(__import__("json").dumps(base), encoding="utf-8")

    with pytest.raises(Observation2DError, match="frame"):
        load_manual_annotations(path, frame)

    base["frames"][0]["frame_index"] = 0
    base["source_kind"] = "detector_output"
    path.write_text(__import__("json").dumps(base), encoding="utf-8")
    with pytest.raises(Observation2DError, match="manual_visual_annotation"):
        load_manual_annotations(path, frame)


def test_annotations_reject_out_of_bounds_box_and_fabricated_manual_confidence(tmp_path):
    frame = _frame()
    manifest = {
        "schema": "damh.manual-visual-annotations.v1",
        "annotation_set_id": "x",
        "annotator_id": "assistant_visual_review",
        "source_kind": "manual_visual_annotation",
        "source_classification": "recorded_sim_observation",
        "episode_sha256": frame.provenance.episode_sha256,
        "frames": [{
            "camera_id": "head_camera",
            "frame_index": 0,
            "rgb_sha256": frame.provenance.rgb_sha256,
            "image_size_px": [100, 80],
            "objects": [{
                "object_id": "bad",
                "label": "object",
                "bbox_xyxy_px": [90, 2, 101, 50],
                "visibility": "visible",
                "confidence": 0.99,
            }],
        }],
    }
    path = tmp_path / "annotations.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    with pytest.raises(Observation2DError, match="bounds"):
        load_manual_annotations(path, frame)

    manifest["frames"][0]["objects"][0]["bbox_xyxy_px"] = [10, 2, 20, 50]
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(Observation2DError, match="confidence"):
        load_manual_annotations(path, frame)


def test_serialized_observation_keeps_missing_time_revision_depth_and_calibration_explicit():
    record = observations_json(_frame(), ())

    assert record["observation_timestamp_s"] is None
    assert record["observation_timestamp_source"] is None
    assert record["scene_revision"] is None
    assert record["depth_available"] is False
    assert record["calibration"]["intrinsic_cv"][0][0] == 100.0
    assert record["calibration"]["camera_to_world_cv"][2][3] == 3.0
    assert record["metric_object_pose_available"] is False
    assert record["controller_eligible"] is False


def test_manual_annotations_reject_malformed_objects_and_duplicate_ids(tmp_path):
    import json

    frame = _frame()
    object_record = {
        "object_id": "same-visual-id",
        "label": "manual appearance",
        "bbox_xyxy_px": [1, 2, 12, 20],
        "visibility": "visible",
        "confidence": None,
    }
    manifest = {
        "schema": "damh.manual-visual-annotations.v1",
        "annotation_set_id": "review-01",
        "annotator_id": "manual-review",
        "source_kind": "manual_visual_annotation",
        "source_classification": "recorded_sim_observation",
        "episode_sha256": frame.provenance.episode_sha256,
        "frames": [{
            "camera_id": frame.camera_id,
            "frame_index": frame.frame_index,
            "rgb_sha256": frame.provenance.rgb_sha256,
            "image_size_px": [100, 80],
            "objects": [object_record, dict(object_record)],
        }],
    }
    path = tmp_path / "manual.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Observation2DError, match="duplicate"):
        load_manual_annotations(path, frame)

    for invalid_objects in ("not a list", [None], [42]):
        manifest["frames"][0]["objects"] = invalid_objects
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(Observation2DError, match=r"objects|JSON object"):
            load_manual_annotations(path, frame)
