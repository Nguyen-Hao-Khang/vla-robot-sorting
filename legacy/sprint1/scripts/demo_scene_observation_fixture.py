"""Replay a labelled fixture through camera geometry, scene state and grasp proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vla_robot_sorting.contracts.grasping import ParallelJawGripper
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    DestinationRegion,
    GeometrySource,
    ObjectVisibility,
)
from vla_robot_sorting.contracts.semantic import SemanticProvenance, SemanticRequest, SourceLanguage
from vla_robot_sorting.orchestration.semantic_to_grasp import SemanticToGraspBridge
from vla_robot_sorting.perception.scene_observation import (
    CameraCalibration,
    ObjectEvidence,
    SceneEvidenceKind,
    SceneObservationRecord,
    build_scene_snapshot,
)
from vla_robot_sorting.semantics.vlm import parse_semantic_payload

FIXTURE_RGB = b"DAMH-S1V synthetic RGB fixture; no camera capture"
FIXTURE_DEPTH = b"DAMH-S1V synthetic depth fixture; 0.70 m at pixel (160,120)"
FIXTURE_CALIBRATION = b"fx=fy=320; cx=160; cy=120; camera_to_world=identity"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def semantic_result(observation_revision: str):
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
        "fixture", "no_model_call", "fixture-v1", digest(FIXTURE_RGB), digest(FIXTURE_DEPTH), False
    )
    return parse_semantic_payload(payload, request=request, provenance=provenance)


def run_demo(output_path: Path) -> dict[str, object]:
    calibration = CameraCalibration(
        camera_id="fixture_head",
        camera_frame_id="fixture_head",
        world_frame_id="world",
        width_px=320,
        height_px=240,
        intrinsic_matrix=((320.0, 0.0, 160.0), (0.0, 320.0, 120.0), (0.0, 0.0, 1.0)),
        camera_to_world=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        transform_revision="fixture-tf-v1",
        timestamp_s=100.0,
        calibration_sha256=digest(FIXTURE_CALIBRATION),
    )
    record = SceneObservationRecord(
        observation_id="fixture-observation-1",
        source=SceneEvidenceKind.FIXTURE,
        scene_revision="fixture-scene-1",
        timestamp_s=100.0,
        rgb_sha256=digest(FIXTURE_RGB),
        depth_sha256=digest(FIXTURE_DEPTH),
        calibration=calibration,
        artifact_sha256=digest(FIXTURE_RGB + FIXTURE_DEPTH + FIXTURE_CALIBRATION),
        metadata={"fixture_label": "synthetic_pixel_depth_and_calibration"},
    )
    target = ObjectEvidence(
        object_id="red-block",
        semantic_labels=("red block", "block"),
        geometry=BoxGeometry((0.04, 0.04, 0.08)),
        geometry_source=GeometrySource.ESTIMATED_PROXY,
        evidence_kind=SceneEvidenceKind.IMAGE_ESTIMATE,
        visibility=ObjectVisibility.VISIBLE,
        identity_confidence=0.92,
        pixel_center_uv=(160.0, 120.0),
        depth_m=0.70,
    )
    destination = DestinationRegion(
        "left-tray",
        Pose6D((0.30, 0.0, 0.04), (0.0, 0.0, 0.0, 1.0), "world"),
        (0.03, 0.03, 0.03),
        ("left tray",),
    )
    scene = build_scene_snapshot(
        record,
        (target,),
        destinations=(destination,),
        max_age_s=0.1,
        now_s=100.0,
        table_height_m=0.0,
    )
    result = SemanticToGraspBridge().propose(
        semantic_result(record.observation_id),
        scene,
        ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08),
    )
    if not result.succeeded or result.proposal is None:
        raise RuntimeError(f"fixture bridge failed: {result.reason}")
    report: dict[str, object] = {
        "classification": "SYNTHETIC_FIXTURE_REPLAY_NOT_RECORDED_NATIVE",
        "detection_performed": False,
        "network_calls": 0,
        "native_simulator_calls": 0,
        "physics_steps": 0,
        "executed": False,
        "control_eligible": result.proposal.control_eligible,
        "observation_source": scene.metadata["observation_source"],
        "scene_revision": scene.scene_revision,
        "transform_revision": scene.transform_revision,
        "observation_revision": scene.metadata["observation_revision"],
        "rgb_sha256": scene.metadata["rgb_sha256"],
        "depth_sha256": scene.metadata["depth_sha256"],
        "calibration_sha256": scene.metadata["calibration_sha256"],
        "camera_to_world": scene.metadata["camera_to_world"],
        "target_evidence_kind": scene.object("red-block").metadata["evidence_kind"],
        "target_world_pose": {
            "position_xyz_m": scene.object("red-block").pose.position_xyz,
            "orientation_xyzw": scene.object("red-block").pose.orientation_xyzw,
            "frame_id": scene.object("red-block").pose.frame_id,
        },
        "destination_id": result.proposal.destination_id,
        "geometry_candidate_count": len(result.proposal.candidates),
        "geometry_feasibility_checked": result.proposal.feasibility_checked,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/s1v/scene-observation-fixture.json"),
    )
    args = parser.parse_args()
    report = run_demo(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
