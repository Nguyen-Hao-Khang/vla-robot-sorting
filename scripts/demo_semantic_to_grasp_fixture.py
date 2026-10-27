"""Replay synthetic VI/EN semantic-to-grasp geometry and an offline HTML visualization.

No Gemini, RGB detection, IK, collision certification, RoboTwin or robot motion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from vla_robot_sorting.contracts.grasping import ParallelJawGripper
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    DestinationRegion,
    GeometrySource,
    ObjectVisibility,
    SceneObject,
    SceneSnapshot,
)
from vla_robot_sorting.contracts.semantic import (
    SemanticProvenance,
    SemanticRequest,
    SourceLanguage,
)
from vla_robot_sorting.orchestration.semantic_to_grasp import (
    BridgeFailureCode,
    SemanticToGraspBridge,
)
from vla_robot_sorting.semantics.vlm import parse_semantic_payload
from vla_robot_sorting.visualization.semantic_report import render_fixture_dashboard


def fixture_scene() -> SceneSnapshot:
    return SceneSnapshot(
        scene_revision="fixture-scene-1",
        transform_revision="fixture-tf-1",
        frame_id="world",
        objects=(
            SceneObject(
                "red-block",
                Pose6D((0.2, 0.0, 0.10), (0.0, 0.0, 0.0, 1.0), "world"),
                BoxGeometry((0.04, 0.04, 0.08)),
                GeometrySource.ESTIMATED_PROXY,
                ("red block", "khối đỏ", "block"),
                0.92,
            ),
            SceneObject(
                "blue-block",
                Pose6D((-0.2, 0.0, 0.10), (0.0, 0.0, 0.0, 1.0), "world"),
                BoxGeometry((0.04, 0.04, 0.08)),
                GeometrySource.ESTIMATED_PROXY,
                ("blue block", "khối xanh", "block"),
                0.91,
            ),
        ),
        destinations=(
            DestinationRegion(
                "left-tray",
                Pose6D((-0.2, 0.3, 0.04), (0.0, 0.0, 0.0, 1.0), "world"),
                (0.03, 0.03, 0.03),
                ("left tray", "khay trái", "tray"),
            ),
        ),
        table_height_m=0.0,
        metadata={
            "observation_revision": "fixture-obs-1",
            "observation_source": "fixture",
            "claim_boundary": "synthetic scene and explicit alias labels only",
        },
    )


def fixture_semantic(command: str, language: SourceLanguage, reference: str):
    """Parse a fixed fixture payload using the production strict semantic parser."""
    request = SemanticRequest(command, language, "fixture-obs-1")
    provenance = SemanticProvenance(
        provider="fixture_only",
        model_id="no_model_call",
        prompt_version="fixture-v1",
        prompt_sha256="a" * 64,
        schema_sha256="b" * 64,
        fallback_used=False,
    )
    payload = {
        "decision": "structured_success",
        "normalized_intent_id": "sort:red-block:left-tray",
        "operation": "pick_and_place",
        "target": {
            "reference": reference,
            "category": "block",
            "attributes": {
                "color": None,
                "shape": None,
                "size": None,
                "location": None,
            },
            "relation": None,
            "relation_reference": None,
        },
        "destination": {
            "reference": "khay trái" if language is SourceLanguage.VI else "left tray",
            "category": "tray",
            "attributes": {
                "color": None,
                "shape": None,
                "size": None,
                "location": "left" if language is SourceLanguage.EN else None,
            },
            "relation": None,
            "relation_reference": None,
        },
        "semantic_confidence": 0.95,
        "ambiguity": None,
        "reason": None,
    }
    return parse_semantic_payload(payload, request=request, provenance=provenance)


def _scene_record(scene: SceneSnapshot) -> dict[str, object]:
    """Export only geometry actually present in the named synthetic scene."""
    return {
        "scene_revision": scene.scene_revision,
        "transform_revision": scene.transform_revision,
        "observation_revision": scene.metadata["observation_revision"],
        "observation_source": scene.metadata["observation_source"],
        "frame_id": scene.frame_id,
        "table_height_m": scene.table_height_m,
        "objects": [
            {
                "object_id": obj.object_id,
                "pose": asdict(obj.pose),
                "size_xyz_m": list(obj.geometry.size_xyz_m),
                "identity_confidence": obj.identity_confidence,
                "visibility": obj.visibility.value,
                "semantic_labels": list(obj.semantic_labels),
                "geometry_source": obj.geometry_source.value,
            }
            for obj in scene.objects
        ],
        "destinations": [
            {
                "destination_id": item.destination_id,
                "pose": asdict(item.pose),
                "tolerance_xyz_m": list(item.tolerance_xyz_m),
                "semantic_labels": list(item.semantic_labels),
            }
            for item in scene.destinations
        ],
    }


def run_fixture() -> dict[str, object]:
    scene = fixture_scene()
    gripper = ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08)
    bridge = SemanticToGraspBridge()
    cases = (
        (SourceLanguage.EN, "Put the red block in the left tray", "red block"),
        (SourceLanguage.VI, "Đưa khối đỏ vào khay trái", "khối đỏ"),
    )
    records = []
    for language, command, reference in cases:
        semantic = fixture_semantic(command, language, reference)
        response = bridge.propose(semantic, scene, gripper)
        if not response.succeeded or response.proposal is None:
            raise RuntimeError(f"fixture {language.value} rejected: {response.reason}")
        proposal = response.proposal
        records.append(
            {
                "language": language.value,
                "command": command,
                "semantic_source": semantic.provenance.provider,
                "semantic_decision": semantic.decision.value,
                "normalized_intent_id": semantic.normalized_intent_id,
                "semantic_confidence": semantic.semantic_confidence,
                "target": proposal.target_id,
                "destination": proposal.destination_id,
                "scene_revision": proposal.scene_revision,
                "transform_revision": proposal.transform_revision,
                "observation_revision": proposal.observation_revision,
                "observation_source": proposal.observation_source,
                "geometry_source": proposal.geometry_source,
                "candidate_count": len(proposal.candidates),
                "candidate_ids": list(proposal.candidate_ids),
                "candidates": [asdict(candidate) for candidate in proposal.candidates],
                "top_geometry_candidate": {
                    "candidate_id": proposal.candidates[0].candidate_id,
                    "grasp_pose": asdict(proposal.candidates[0].grasp_pose),
                    "pregrasp_pose": asdict(proposal.candidates[0].pregrasp_pose),
                },
                "feasibility_checked": proposal.feasibility_checked,
                "executed": False,
            }
        )
    if records[0]["candidate_ids"] != records[1]["candidate_ids"]:
        raise RuntimeError("VI/EN fixture candidate proposals differ")

    en_semantic = fixture_semantic(cases[0][1], SourceLanguage.EN, "red block")
    assert en_semantic.target is not None
    altered = (
        (
            "Stale observation revision",
            scene,
            replace(en_semantic, observation_revision="fixture-obs-stale"),
            BridgeFailureCode.OBSERVATION_STALE,
        ),
        (
            "Target occluded",
            replace(
                scene,
                objects=(
                    replace(scene.objects[0], visibility=ObjectVisibility.OCCLUDED),
                    scene.objects[1],
                ),
            ),
            en_semantic,
            BridgeFailureCode.TARGET_UNAVAILABLE,
        ),
        (
            "Ambiguous target reference",
            scene,
            replace(en_semantic, target=replace(en_semantic.target, reference="block")),
            BridgeFailureCode.GROUNDING_FAILED,
        ),
    )
    rejections = []
    for label, altered_scene, semantic, expected in altered:
        result = bridge.propose(semantic, altered_scene, gripper)
        if result.succeeded or result.failure_code is not expected:
            raise RuntimeError(f"unsafe adversarial outcome in {label}: {result}")
        rejections.append(
            {
                "label": label,
                "observation_revision": semantic.observation_revision,
                "failure_code": result.failure_code.value,
                "reason": result.reason,
                "scene": _scene_record(altered_scene),
                "candidate_count": 0,
                "executed": False,
            }
        )
    return {
        "schema": "semantic-to-grasp-fixture-demo-v2",
        "source": "synthetic_fixture",
        "network_calls": 0,
        "native_simulation_calls": 0,
        "physics_steps": 0,
        "claim_boundary": (
            "Strict semantic-schema-to-geometry integration only; no live VLM, "
            "RGB detection, IK feasibility, collision planning or physical execution"
        ),
        "scene": _scene_record(scene),
        "cases": records,
        "rejections": rejections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence output")
    parser.add_argument(
        "--html-output", type=Path, help="Standalone dashboard path (default: next to JSON)"
    )
    args = parser.parse_args()
    output = run_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    html_path = args.html_output or args.output.with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_fixture_dashboard(output), encoding="utf-8")
    print("SEMANTIC_GEOMETRY_FIXTURE_PASS")
    print(f"cases={len(output['cases'])} safe_rejections={len(output['rejections'])} "
          "languages=en,vi native_calls=0 physics_steps=0")
    print(f"json={args.output}")
    print(f"visualization={html_path} (open in browser; synthetic fixture only)")


if __name__ == "__main__":
    main()
