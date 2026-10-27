from __future__ import annotations

from math import sqrt

import pytest

from vla_robot_sorting.contracts.grasping import GraspRequest, ParallelJawGripper
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    DestinationRegion,
    GeometrySource,
    ObjectVisibility,
)
from vla_robot_sorting.contracts.semantic import SemanticProvenance, SemanticRequest, SourceLanguage
from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator
from vla_robot_sorting.orchestration.semantic_to_grasp import (
    BridgeFailureCode,
    SemanticToGraspBridge,
)
from vla_robot_sorting.perception.scene_observation import (
    CameraCalibration,
    ObjectEvidence,
    SceneEvidenceKind,
    SceneObservationRecord,
    build_scene_snapshot,
)
from vla_robot_sorting.semantics.vlm import parse_semantic_payload


def calibration(*, rotation=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))):
    return CameraCalibration(
        camera_id="head_camera",
        camera_frame_id="head_camera",
        world_frame_id="world",
        width_px=320,
        height_px=240,
        intrinsic_matrix=((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
        camera_to_world=(
            (*rotation[0], 1.0),
            (*rotation[1], 2.0),
            (*rotation[2], 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        transform_revision="tf-4",
        timestamp_s=12.5,
        calibration_sha256="c" * 64,
    )


def record(*, depth_sha256="b" * 64, timestamp_s=12.5, source=SceneEvidenceKind.FIXTURE):
    return SceneObservationRecord(
        observation_id="episode0:head:frame-10",
        source=source,
        scene_revision="scene-10",
        timestamp_s=timestamp_s,
        rgb_sha256="a" * 64,
        depth_sha256=depth_sha256,
        calibration=calibration(),
        artifact_sha256="d" * 64,
    )


def semantic_result(observation_revision):
    payload = {
        "decision": "structured_success",
        "normalized_intent_id": "sort:red-block:left-tray",
        "operation": "pick_and_place",
        "target": {
            "reference": "red block",
            "category": "block",
            "attributes": {"color": "red", "shape": None, "size": None, "location": None},
            "relation": None,
            "relation_reference": None,
        },
        "destination": {
            "reference": "left tray",
            "category": "tray",
            "attributes": {"color": None, "shape": None, "size": None, "location": "left"},
            "relation": None,
            "relation_reference": None,
        },
        "semantic_confidence": 0.96,
        "ambiguity": None,
        "reason": None,
    }
    request = SemanticRequest(
        "move red block to left tray", SourceLanguage.EN, observation_revision
    )
    provenance = SemanticProvenance(
        "fixture", "no_model_call", "fixture-v1", "a" * 64, "b" * 64, False
    )
    return parse_semantic_payload(payload, request=request, provenance=provenance)


def estimated_object(
    *, visibility=ObjectVisibility.VISIBLE, kind=SceneEvidenceKind.IMAGE_ESTIMATE, pose_world=None
):
    return ObjectEvidence(
        object_id="red-block",
        semantic_labels=("red block", "block"),
        geometry=BoxGeometry((0.04, 0.04, 0.08)),
        geometry_source=GeometrySource.ESTIMATED_PROXY,
        evidence_kind=kind,
        visibility=visibility,
        identity_confidence=0.92,
        pixel_center_uv=(60.0, 50.0),
        depth_m=2.0 if visibility is ObjectVisibility.VISIBLE else None,
        orientation_camera_xyzw=(0.0, 0.0, 0.0, 1.0),
        pose_world=pose_world,
    )


def test_fixture_rgbd_pixel_is_transformed_and_integrated_into_geometry_only_grasp_proposal():
    destination = DestinationRegion(
        "left-tray",
        Pose6D((0.3, 0.0, 0.04), (0.0, 0.0, 0.0, 1.0), "world"),
        (0.03, 0.03, 0.03),
        ("left tray", "khay trái"),
    )
    snapshot = build_scene_snapshot(
        record(),
        (estimated_object(),),
        max_age_s=1.0,
        now_s=12.5,
        destinations=(destination,),
    )
    target = snapshot.object("red-block")

    assert target.pose.position_xyz == pytest.approx((1.0, 2.2, 5.0))
    assert target.pose.orientation_xyzw == pytest.approx((0.0, 0.0, sqrt(0.5), sqrt(0.5)))
    assert target.pose.frame_id == snapshot.frame_id == "world"
    assert target.metadata["evidence_kind"] == "image_estimate"
    assert target.geometry_source is GeometrySource.ESTIMATED_PROXY
    assert target.metadata["control_eligible"] is False
    assert snapshot.metadata["rgb_sha256"] == "a" * 64
    assert snapshot.metadata["depth_sha256"] == "b" * 64
    assert snapshot.metadata["observation_source"] == "fixture"

    generated = GeometryFirstGraspGenerator().generate(
        GraspRequest(
            object_id="red-block",
            scene_revision=snapshot.scene_revision,
            transform_revision=snapshot.transform_revision,
            gripper=ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08),
        ),
        snapshot,
    )
    assert generated.succeeded
    assert generated.candidates
    assert all(c.grasp_pose.frame_id == "world" for c in generated.candidates)
    semantic = semantic_result(snapshot.metadata["observation_revision"])
    bridged = SemanticToGraspBridge().propose(
        semantic, snapshot, ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08)
    )
    assert bridged.succeeded
    assert bridged.proposal.source_evidence_kind == "image_estimate"
    assert bridged.proposal.control_eligible is False


def test_image_estimate_rejects_missing_depth():
    with pytest.raises(ValueError, match="depth"):
        build_scene_snapshot(
            record(depth_sha256=None),
            (estimated_object(),),
            max_age_s=1.0,
            now_s=12.5,
        )


def test_occluded_object_is_preserved_as_unavailable_and_never_grasped():
    snapshot = build_scene_snapshot(
        record(source=SceneEvidenceKind.RECORDED_SIM_OBSERVATION),
        (
            estimated_object(
                visibility=ObjectVisibility.OCCLUDED,
                kind=SceneEvidenceKind.RECORDED_SIM_OBSERVATION,
                pose_world=Pose6D((1.0, 2.2, 5.0), (0.0, 0.0, 0.0, 1.0), "world"),
            ),
        ),
        max_age_s=1.0,
        now_s=12.5,
        destinations=(
            DestinationRegion(
                "left-tray",
                Pose6D((0.3, 0.0, 0.04), (0.0, 0.0, 0.0, 1.0), "world"),
                (0.03, 0.03, 0.03),
                ("left tray",),
            ),
        ),
    )
    assert snapshot.object("red-block").visibility is ObjectVisibility.OCCLUDED
    result = SemanticToGraspBridge().propose(
        semantic_result(snapshot.metadata["observation_revision"]),
        snapshot,
        ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08),
    )
    assert result.failure_code is BridgeFailureCode.TARGET_UNAVAILABLE


def test_stale_observation_and_camera_frame_timestamp_are_rejected():
    with pytest.raises(ValueError, match="stale"):
        build_scene_snapshot(record(), (estimated_object(),), max_age_s=0.1, now_s=13.0)

    with pytest.raises(ValueError, match="timestamp"):
        build_scene_snapshot(
            record(timestamp_s=12.4), (estimated_object(),), max_age_s=1.0, now_s=12.5
        )


def test_non_rigid_or_wrong_calibration_is_rejected():
    with pytest.raises(ValueError, match="rotation"):
        CameraCalibration(
            camera_id="head_camera",
            camera_frame_id="head_camera",
            world_frame_id="world",
            width_px=320,
            height_px=240,
            intrinsic_matrix=((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
            camera_to_world=(
                (2.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            transform_revision="tf-bad",
            timestamp_s=12.5,
            calibration_sha256="c" * 64,
        )


def test_privileged_geometry_keeps_distinct_provenance_and_needs_no_depth():
    gt_record = record(depth_sha256=None, source=SceneEvidenceKind.SIM_GROUND_TRUTH)
    privileged = ObjectEvidence(
        object_id="red-block",
        semantic_labels=("red block",),
        geometry=BoxGeometry((0.04, 0.04, 0.08)),
        geometry_source=GeometrySource.SIM_GROUND_TRUTH,
        evidence_kind=SceneEvidenceKind.SIM_GROUND_TRUTH,
        visibility=ObjectVisibility.VISIBLE,
        identity_confidence=1.0,
        pose_world=Pose6D((0.1, 0.2, 0.04), (0.0, 0.0, 0.0, 1.0), "world"),
    )
    snapshot = build_scene_snapshot(gt_record, (privileged,), max_age_s=1.0, now_s=12.5)
    assert snapshot.metadata["observation_source"] == "sim_ground_truth"
    assert snapshot.object("red-block").metadata["evidence_kind"] == "sim_ground_truth"
    assert snapshot.object("red-block").metadata["control_eligible"] is False


def test_projection_rejects_skew_not_supported_by_backprojection() -> None:
    with pytest.raises(ValueError, match="zero skew"):
        CameraCalibration(
            camera_id="head", camera_frame_id="head", world_frame_id="world",
            width_px=320, height_px=240,
            intrinsic_matrix=((100.0, 15.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
            camera_to_world=(
                (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0),
            ),
            transform_revision="tf-skew", timestamp_s=12.5,
            calibration_sha256="a" * 64,
        )


def test_scene_refuses_forged_privileged_object_inside_fixture() -> None:
    privileged = ObjectEvidence(
        object_id="red-block",
        semantic_labels=("red block",),
        geometry=BoxGeometry((0.04, 0.04, 0.08)),
        geometry_source=GeometrySource.SIM_GROUND_TRUTH,
        evidence_kind=SceneEvidenceKind.SIM_GROUND_TRUTH,
        visibility=ObjectVisibility.VISIBLE,
        identity_confidence=1.0,
        pose_world=Pose6D((0.1, 0.2, 0.04), (0.0, 0.0, 0.0, 1.0), "world"),
    )
    with pytest.raises(ValueError, match="source conflicts"):
        build_scene_snapshot(record(), (privileged,), max_age_s=1.0, now_s=12.5)


def test_image_pose_must_come_from_projection_not_world_pose_override() -> None:
    with pytest.raises(ValueError, match="must be projected"):
        build_scene_snapshot(
            record(),
            (estimated_object(
                pose_world=Pose6D((999.0, 999.0, 999.0),
                                  (0.0, 0.0, 0.0, 1.0), "world"),
            ),),
            max_age_s=1.0, now_s=12.5,
        )
