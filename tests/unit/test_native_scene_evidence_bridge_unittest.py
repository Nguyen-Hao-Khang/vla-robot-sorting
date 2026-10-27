# Scoped imported native evidence source; no semantic changes.
"""Unittest regression runnable even when Ubuntu RoboTwin lacks pytest."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace

from vla_robot_sorting.perception.native_scene_evidence_bridge import (
    NativeEvidenceBridgeError,
    build_unidentified_native_evidence,
    seal_visual_evidence,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64


@dataclass(frozen=True)
class Region:
    depth_m: float = 0.45
    observed_pixel_uv: tuple[int, int] = (25, 25)


@dataclass(frozen=True)
class Pixel:
    depth_m: float = 0.44
    pixel_uv: tuple[int, int] = (30, 30)


def fixture(camera: str = "head_camera"):
    manifest = A if camera == "head_camera" else E
    p = SimpleNamespace(
        camera_id=camera,
        frame_index=0,
        source_kind="native_rgbd_single_refresh_region_proposals",
        source_classification="recorded_sim_observation",
        source_frame_association="probe.get_obs single-frame camera refresh",
        native_depth_metric_quality_audited=True,
        verified_hdf5_episode=False,
        verified_object_identity=False,
        object_pose_world=None,
        controller_eligible=False,
        native_depth_manifest_sha256=manifest,
        rgb_sha256=B,
        depth_sha256=C,
        calibration_manifest_sha256=D,
        producer_config_sha256=E,
        producer_method="native_opencv_mser_canny",
        producer_version="4.11",
        image_width_px=100,
        image_height_px=80,
        proposals=(
            SimpleNamespace(
                proposal_id=f"native-{camera}-000000-rp-0000",
                camera_id=camera, frame_index=0,
                source_kind="native_rgb_region_proposal",
                bbox_xyxy_px=(10, 10, 50, 50),
                object_id=None, label=None, confidence=None, proposal_score=None,
            ),
        ),
    )
    d = SimpleNamespace(
        proposal_id=p.proposals[0].proposal_id,
        camera_id=camera, frame_index=0,
        source_manifest_sha256=manifest, source_rgb_sha256=B, source_depth_sha256=C,
        measured_region=Region(),
        geometry_claim="one_observed_pixel_in_unsegmented_RGB_rectangle",
        object_id=None, object_pose_world=None, object_dimensions_m=None,
        object_orientation=None, grasp_validated=False,
        controller_eligible=False,
    )
    landmark = SimpleNamespace(
        camera_id=camera, source_frame_index=0,
        source_manifest_sha256=manifest, rgb_sha256=B, depth_sha256=C,
        landmark_kind="orange_surface",
        selection_method="canonical_hsv_color_largest_component_not_object_detector_v1",
        bbox_xyxy_px=(20, 20, 45, 45),
        mask_pixel_count=200, valid_depth_fraction=.80,
        median_masked_depth_m=.44, masked_depth_iqr_m=.01,
        observed_pixel=Pixel(),
        object_id=None, object_pose_world=None, object_orientation=None,
        grasp_candidate_validated=False, controller_eligible=False,
    )
    return p, d, landmark


class NativeEvidenceTests(unittest.TestCase):
    def test_observed_pixel_evidence_never_becomes_object(self):
        p, d, landmark = fixture()
        report = build_unidentified_native_evidence(p, [d], [landmark])
        self.assertEqual(report["proposals_total"], 1)
        self.assertEqual(report["depth_points"][0]["measured_region"]["depth_m"], .45)
        self.assertIsNone(report["depth_points"][0]["object_id"])
        self.assertIsNone(report["color_landmarks"][0]["object_pose_world"])
        self.assertFalse(report["scene_snapshot_created"])
        self.assertFalse(report["semantic_grounding_authorized"])
        self.assertFalse(report["controller_eligible"])
        self.assertIsNone(report["source_timestamp_s"])

    def test_two_camera_sealing_has_separate_provenance(self):
        head = build_unidentified_native_evidence(*_wrap(fixture("head_camera")))
        wrist = build_unidentified_native_evidence(*_wrap(fixture("wrist_camera")))
        result = seal_visual_evidence([wrist, head])
        self.assertEqual([r["camera_id"] for r in result["cameras"]],
                         ["head_camera", "wrist_camera"])
        self.assertEqual(result["classification"],
                         "SOURCE_VERIFIED_UNIDENTIFIED_PIXELS_ONLY")
        self.assertEqual(len(result["report_sha256"]), 64)
        self.assertFalse(result["controller_eligible"])
        with self.assertRaises(NativeEvidenceBridgeError):
            seal_visual_evidence([head, head])

    def test_missing_identity_and_timestamp_cannot_be_filled_by_adapter(self):
        p, d, landmark = fixture()
        r = build_unidentified_native_evidence(p, [d], [landmark])
        for field in ("scene_revision", "observation_revision", "source_timestamp_s"):
            self.assertIsNone(r[field])
        self.assertFalse(r["object_geometry_verified"])
        self.assertFalse(r["object_orientation_verified"])
        self.assertFalse(r["grasp_planning_authorized"])

    def test_cross_camera_depth_is_rejected(self):
        p, d, landmark = fixture()
        d.camera_id = "wrist_camera"
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_cross_session_depth_sha_rejected(self):
        p, d, landmark = fixture()
        d.source_rgb_sha256 = D
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_forged_depth_object_identity_rejected(self):
        p, d, landmark = fixture()
        d.object_id = "pillbottle"
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_forged_proposal_confidence_rejected(self):
        p, d, landmark = fixture()
        p.proposals[0].confidence = .99
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_forged_landmark_pose_rejected(self):
        p, d, landmark = fixture()
        landmark.object_pose_world = (1, 2, 3)
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_wrong_landmark_manifest_rejected(self):
        p, d, landmark = fixture()
        landmark.source_manifest_sha256 = E
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_native_only_control_claim_rejected(self):
        p, d, landmark = fixture()
        p.controller_eligible = True
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_source_quality_audit_required(self):
        p, d, landmark = fixture()
        p.native_depth_metric_quality_audited = False
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_empty_depth_is_allowed_without_fabrication(self):
        p, _unused_depth, _unused_landmark = fixture()
        result = build_unidentified_native_evidence(p, [], [])
        self.assertEqual(result["depth_points"], [])
        self.assertEqual(result["color_landmarks"], [])
        self.assertEqual(result["proposals_total"], 1)
        self.assertFalse(result["object_identity_verified"])

    def test_nonfinite_depth_payload_fails(self):
        p, d, landmark = fixture()
        d.measured_region = replace(d.measured_region, depth_m=float("nan"))
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_duplicate_proposal_id_fails(self):
        p, d, landmark = fixture()
        p.proposals = (p.proposals[0], p.proposals[0])
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_landmark_not_object_identity(self):
        p, d, landmark = fixture()
        landmark.grasp_candidate_validated = True
        with self.assertRaises(NativeEvidenceBridgeError):
            build_unidentified_native_evidence(p, [d], [landmark])

    def test_missing_both_camera_reports_rejected(self):
        head = build_unidentified_native_evidence(*_wrap(fixture()))
        with self.assertRaises(NativeEvidenceBridgeError):
            seal_visual_evidence([head])


def _wrap(f):
    p, d, landmark = f
    return p, [d], [landmark]


if __name__ == "__main__":
    unittest.main()
