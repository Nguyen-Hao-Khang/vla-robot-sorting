"""S1-V7: source-only RGB/depth fusion guard. All data synthetic offline."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from vla_robot_sorting.perception.native_depth_capture import NATIVE_METRIC_DEPTH_CONVENTION
from vla_robot_sorting.perception.proposal_depth_alignment import (
    ProposalDepthAlignmentError,
    verify_proposal_depth_alignment,
)
from vla_robot_sorting.perception.region_proposals import RegionProposal, RegionProposalSet


def _proposals():
    return RegionProposalSet(
        camera_id="head_camera",
        frame_index=0,
        frame_count=158,
        image_width_px=320,
        image_height_px=240,
        source_classification="recorded_sim_observation",
        episode_sha256="a" * 64,
        rgb_sha256="b" * 64,
        calibration_sha256="c" * 64,
        proposal_method="opencv_mser_canny_regions",
        proposal_method_version="1",
        implementation="opencv",
        implementation_version="5.0.0",
        config_sha256="d" * 64,
        confidence_semantics="unscored_class_agnostic_proposals",
        proposals=(
            RegionProposal(
                proposal_id="head_camera-000000-rp-0000",
                camera_id="head_camera",
                frame_index=0,
                bbox_xyxy_px=(20, 30, 60, 90),
            ),
        ),
    )


def _native(tmp_path, monkeypatch, **overrides):
    from vla_robot_sorting.perception import proposal_depth_alignment as module

    root = tmp_path / "native-depth"
    root.mkdir()
    manifest = {
        "camera_id": "head_camera",
        "frame_index": 0,
        "image_size_px": [320, 240],
        "depth_convention": NATIVE_METRIC_DEPTH_CONVENTION,
        "source_episode_sha256": "a" * 64,
        "source_calibration_sha256": "c" * 64,
        "rgb_sha256": "b" * 64,
        "source_frame_association": "same_verified_observation_frame",
    }
    manifest.update(overrides)
    (root / "head_camera_frame_000000_depth-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    # Metric-quality source validation is independently unit tested. This
    # fake only exercises the alignment policy AFTER successful auditing.
    monkeypatch.setattr(
        module,
        "audit_native_metric_quality",
        lambda _directory: {
            "status": "NATIVE_RGBD_PLAUSIBILITY_PASS_NOT_METRIC_ACCURACY_OR_TASK_ACCEPTANCE"
        },
    )
    return root


def test_exact_matching_source_only_grants_alignment_not_pose_or_execution(tmp_path, monkeypatch):
    root = _native(tmp_path, monkeypatch)
    result = verify_proposal_depth_alignment(_proposals(), root)
    assert result.source_alignment_verified is True
    assert result.object_identity_verified is False
    assert result.metric_object_pose_verified is False
    assert result.grasp_candidate_validated is False
    assert result.controller_eligible is False
    assert len(result.native_manifest_sha256) == 64


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source_episode_sha256", None),
        ("source_episode_sha256", "f" * 64),
        ("rgb_sha256", "f" * 64),
        ("source_calibration_sha256", "f" * 64),
        ("source_frame_association", "separate_native_capture"),
        ("depth_convention", "legacy_color_alpha_scaled_depth"),
        ("image_size_px", [640, 360]),
        ("frame_index", 1),
        ("camera_id", "wrist_camera"),
    ],
)
def test_any_native_source_or_frame_mismatch_rejects_fusion(tmp_path, monkeypatch, key, value):
    root = _native(tmp_path, monkeypatch, **{key: value})
    with pytest.raises(ProposalDepthAlignmentError, match="source/frame association"):
        verify_proposal_depth_alignment(_proposals(), root)


@pytest.mark.parametrize(
    "change",
    [
        {"object_identity_inferred": True},
        {"metric_pose_available": True},
        {"controller_eligible": True},
        {"rgb_sha256": "not_sha256"},
        {"frame_index": 1},
    ],
)
def test_proposal_claims_and_provenance_fail_closed(tmp_path, monkeypatch, change):
    root = _native(tmp_path, monkeypatch)
    with pytest.raises(ProposalDepthAlignmentError):
        verify_proposal_depth_alignment(replace(_proposals(), **change), root)


def test_labelled_or_unsafe_region_fails_even_with_matching_provenance(tmp_path, monkeypatch):
    root = _native(tmp_path, monkeypatch)
    p = _proposals()
    bad = replace(p.proposals[0], object_id="invented_bottle")
    with pytest.raises(ProposalDepthAlignmentError, match="unverified identity"):
        verify_proposal_depth_alignment(replace(p, proposals=(bad,)), root)

    bad = replace(p.proposals[0], bbox_xyxy_px=(-1, 0, 30, 40))
    with pytest.raises(ValueError, match="bbox"):
        verify_proposal_depth_alignment(replace(p, proposals=(bad,)), root)


def test_metric_auditor_failure_never_permits_alignment(tmp_path, monkeypatch):
    from vla_robot_sorting.perception import proposal_depth_alignment as module

    root = _native(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "audit_native_metric_quality", lambda _: {"status": "FAIL"})
    with pytest.raises(ProposalDepthAlignmentError, match="metric audit"):
        verify_proposal_depth_alignment(_proposals(), root)


def test_historical_export_lacks_cross_session_attestation(tmp_path, monkeypatch):
    root = _native(tmp_path, monkeypatch, source_episode_sha256=None)
    with pytest.raises(ProposalDepthAlignmentError, match="source_episode_sha256"):
        verify_proposal_depth_alignment(_proposals(), root)


def test_independent_native_audit_error_is_normalized_to_alignment_rejection(tmp_path, monkeypatch):
    from vla_robot_sorting.perception import proposal_depth_alignment as module
    from vla_robot_sorting.perception.native_metric_quality import NativeMetricQualityError

    root = _native(tmp_path, monkeypatch)

    def reject(_):
        raise NativeMetricQualityError("unverified legacy RGB-D scale")

    monkeypatch.setattr(module, "audit_native_metric_quality", reject)
    with pytest.raises(ProposalDepthAlignmentError, match="native metric audit failed"):
        verify_proposal_depth_alignment(_proposals(), root)
