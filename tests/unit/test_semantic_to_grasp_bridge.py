"""Contract tests for geometry-only VI/EN vertical integration."""

from __future__ import annotations

from dataclasses import replace

import pytest

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


def gripper() -> ParallelJawGripper:
    return ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08)


def scene() -> SceneSnapshot:
    def obj(name: str, alias: str, x: float) -> SceneObject:
        return SceneObject(
            name,
            Pose6D((x, 0.0, 0.10), (0.0, 0.0, 0.0, 1.0), "world"),
            BoxGeometry((0.04, 0.04, 0.08)),
            GeometrySource.ESTIMATED_PROXY,
            (alias, "block"),
            0.92,
        )

    return SceneSnapshot(
        "scene-7",
        "tf-2",
        "world",
        (
            obj("red-block", "red block", -0.1),
            obj("blue-block", "blue block", 0.1),
        ),
        (
            DestinationRegion(
                "left-tray",
                Pose6D((-0.2, 0.2, 0.04), (0.0, 0.0, 0.0, 1.0), "world"),
                (0.03, 0.03, 0.03),
                ("left tray", "khay trái"),
            ),
        ),
        table_height_m=0.0,
        metadata={"observation_source": "fixture", "observation_revision": "obs-7"},
    )


def semantic(
    *,
    language: SourceLanguage = SourceLanguage.EN,
    target: str = "red block",
    destination: str = "left tray",
    color: str | None = "red",
    decision: str = "structured_success",
):
    payload = {
        "decision": decision,
        "normalized_intent_id": "sort:red-block:left-tray"
        if decision == "structured_success"
        else None,
        "operation": "pick_and_place" if decision == "structured_success" else None,
        "target": {
            "reference": target,
            "category": "block",
            "attributes": {
                "color": color,
                "shape": None,
                "size": None,
                "location": None,
            },
            "relation": None,
            "relation_reference": None,
        },
        "destination": {
            "reference": destination,
            "category": "tray",
            "attributes": {
                "color": None,
                "shape": None,
                "size": None,
                "location": "left",
            },
            "relation": None,
            "relation_reference": None,
        },
        "semantic_confidence": 0.96,
        "ambiguity": (
            {"reason": "target ambiguous", "clarification_slots": ["target"]}
            if decision == "clarify"
            else None
        ),
        "reason": "not safe" if decision in ("abort", "reobserve") else None,
    }
    provenance = SemanticProvenance(
        "fixture", "no_model_call", "fixture-v1", "a" * 64, "b" * 64, False
    )
    request = SemanticRequest("move red block / đưa khối đỏ", language, "obs-7")
    return parse_semantic_payload(payload, request=request, provenance=provenance)


@pytest.mark.parametrize("language", [SourceLanguage.EN, SourceLanguage.VI])
def test_bilingual_schema_to_geometry_proposals(language: SourceLanguage) -> None:
    output = SemanticToGraspBridge().propose(semantic(language=language), scene(), gripper())
    assert output.succeeded
    assert output.proposal is not None
    assert output.proposal.target_id == "red-block"
    assert output.proposal.destination_id == "left-tray"
    assert output.proposal.scene_revision == "scene-7"
    assert output.proposal.transform_revision == "tf-2"
    assert output.proposal.observation_source == "fixture"
    assert output.proposal.geometry_source == "estimated_proxy"
    assert not output.proposal.feasibility_checked
    assert len(output.proposal.candidates) >= 4
    assert all(item.grasp_pose.frame_id == "world" for item in output.proposal.candidates)


@pytest.mark.parametrize("decision", ["clarify", "abort", "reobserve"])
def test_non_success_semantics_never_emits_grasp(decision: str) -> None:
    s = semantic(decision=decision)
    got = SemanticToGraspBridge().propose(s, scene(), gripper())
    assert not got.succeeded
    assert got.failure_code is BridgeFailureCode.SEMANTIC_REJECTED


def test_stale_observation_fails_closed() -> None:
    s = replace(
        scene(), metadata={"observation_source": "fixture", "observation_revision": "obs-8"}
    )
    got = SemanticToGraspBridge().propose(semantic(), s, gripper())
    assert got.failure_code is BridgeFailureCode.OBSERVATION_STALE


def test_missing_or_unknown_scene_provenance_fails_closed() -> None:
    for source in (None, "unspecified"):
        metadata = {"observation_revision": "obs-7"}
        if source:
            metadata["observation_source"] = source
        got = SemanticToGraspBridge().propose(
            semantic(), replace(scene(), metadata=metadata), gripper()
        )
        assert got.failure_code is BridgeFailureCode.PROVENANCE_MISSING


@pytest.mark.parametrize("target", ["unknown", "block"])
def test_missing_or_ambiguous_target_fails_closed(target: str) -> None:
    s = semantic(target=target, color=None)
    got = SemanticToGraspBridge().propose(s, scene(), gripper())
    assert got.failure_code is BridgeFailureCode.GROUNDING_FAILED


def test_unsupported_attribute_or_relation_fails_closed() -> None:
    from vla_robot_sorting.contracts.semantic import GroundingConstraint, ObjectAttributes

    original = semantic()
    mismatch = replace(
        original,
        target=GroundingConstraint(
            reference="red block", category="block", attributes=ObjectAttributes(color="green")
        ),
    )
    assert (
        SemanticToGraspBridge().propose(mismatch, scene(), gripper()).failure_code
        is BridgeFailureCode.CONSTRAINT_UNSUPPORTED
    )
    relation = replace(
        original,
        target=replace(original.target, relation="left_of", relation_reference="blue-block"),
    )
    assert (
        SemanticToGraspBridge().propose(relation, scene(), gripper()).failure_code
        is BridgeFailureCode.CONSTRAINT_UNSUPPORTED
    )


def test_occluded_or_unconfident_target_fails_closed() -> None:
    base = scene()
    first = base.objects[0]
    occluded = replace(
        base, objects=(replace(first, visibility=ObjectVisibility.OCCLUDED), base.objects[1])
    )
    got = SemanticToGraspBridge().propose(semantic(), occluded, gripper())
    assert got.failure_code is BridgeFailureCode.TARGET_UNAVAILABLE
    low = replace(base, objects=(replace(first, identity_confidence=0.2), base.objects[1]))
    got = SemanticToGraspBridge().propose(semantic(), low, gripper())
    assert got.failure_code is BridgeFailureCode.CONFIDENCE_LOW


def test_unreachable_geometry_is_not_reported_as_feasible_motion() -> None:
    base = scene()
    bad_gripper = ParallelJawGripper(0.20, 0.21, 0.01, 0.02, 0.08)
    got = SemanticToGraspBridge().propose(semantic(), base, bad_gripper)
    assert got.failure_code is BridgeFailureCode.NO_GRASP


def test_low_semantic_confidence_fails_closed() -> None:
    got = SemanticToGraspBridge().propose(
        replace(semantic(), semantic_confidence=0.1), scene(), gripper()
    )
    assert got.failure_code is BridgeFailureCode.CONFIDENCE_LOW


@pytest.mark.parametrize("reference", ["unknown-tray", "tray"])
def test_missing_or_ambiguous_destination_fails_closed(reference: str) -> None:
    from vla_robot_sorting.contracts.perception import DestinationRegion

    data = scene()
    # Both destinations deliberately share an ambiguous "tray" alias.
    second = DestinationRegion(
        "right-tray",
        Pose6D((0.2, 0.2, 0.04), (0.0, 0.0, 0.0, 1.0), "world"),
        (0.03, 0.03, 0.03),
        ("right tray", "tray"),
    )
    base = data.destinations[0]
    first = replace(base, semantic_labels=(*base.semantic_labels, "tray"))
    changed = replace(data, destinations=(first, second))
    intent = semantic(destination=reference)
    got = SemanticToGraspBridge().propose(intent, changed, gripper())
    assert got.failure_code is BridgeFailureCode.GROUNDING_FAILED
    assert got.proposal is None
