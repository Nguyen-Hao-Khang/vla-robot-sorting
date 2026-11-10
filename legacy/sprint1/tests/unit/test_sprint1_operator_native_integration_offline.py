"""Pure offline same-task actor-registry baseline; never physical task acceptance."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from vla_robot_sorting.contracts.grasping import GraspRequest, ParallelJawGripper
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    CylinderGeometry,
    DestinationRegion,
    GeometrySource,
    SceneObject,
    SceneSnapshot,
)
from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator
from vla_robot_sorting.orchestration.native_command_preflight import compile_native_command
from vla_robot_sorting.orchestration.unified_sprint1 import (
    NativeBinding,
    PipelineStatus,
    UnifiedSprint1Pipeline,
)

ROOT = Path(__file__).resolve().parents[2]
GATE = json.loads(
    (ROOT / "configs/s1.3/grasp_recovery_gates.json").read_text(encoding="utf-8")
)
GRIPPER = ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08)


def scene() -> SceneSnapshot:
    return SceneSnapshot(
        scene_revision="native-test-1",
        transform_revision="native-static",
        frame_id="world",
        table_height_m=0.0,
        objects=(
            SceneObject(
                "pillbottle",
                Pose6D((0.2, 0.0, 0.14), (0.0, 0.0, 0.0, 1.0), "world"),
                CylinderGeometry(0.03, 0.12),
                GeometrySource.SIM_GROUND_TRUTH,
                ("pillbottle", "pill bottle"),
                1.0,
            ),
        ),
        destinations=(
            DestinationRegion(
                "pad",
                Pose6D((-0.2, 0.2, 0.02), (0.0, 0.0, 0.0, 1.0), "world"),
                (0.04, 0.04, 0.01),
                ("pad", "blue pad"),
            ),
        ),
        metadata={
            "observation_revision": "native-test-1",
            "observation_source": "sim_ground_truth",
            "identity_provenance": "simulator_native_actor_registry",
            "destination_provenance": "simulator_native_actor_registry",
            "evidence_session": "unique-native-test-session",
        },
    )


def generator() -> GeometryFirstGraspGenerator:
    return GeometryFirstGraspGenerator(
        cylinder_side_grasp_height_fractions=tuple(
            float(v) for v in GATE["cylinder_side_grasp_height_fractions"]
        ),
        box_side_grasp_height_fractions=tuple(
            float(v) for v in GATE["box_side_grasp_height_fractions"]
        ),
        box_parallel_jaw_roll_variants=GATE["box_parallel_jaw_roll_variants"],
    )


@pytest.mark.parametrize(
    ("language", "command"),
    [
        ("vi", "Gắp chai thuốc và đặt lên tấm đệm"),
        ("en", "Pick up the pill bottle and place it on the pad"),
    ],
)
def test_command_semantic_candidates_match_native_b5_geometry_set(language, command):
    observation = scene()
    semantic, digest = compile_native_command(command, language, "native-test-1")
    binding = NativeBinding(
        "move_pillbottle_pad", 0, "native-test-1",
        "native-static", "native-test-1", "unique-native-test-session",
        "pillbottle", "pad",
    )
    report = UnifiedSprint1Pipeline(
        grasp_generator=generator(),
        minimum_table_clearance_m=float(GATE["minimum_table_clearance_m"]),
    ).run(
        command=command,
        language=semantic.source_language,
        scene=observation,
        semantic=semantic,
        semantic_command_sha256=digest,
        binding=binding,
        gripper=GRIPPER,
    )
    native = generator().generate(
        GraspRequest(
            object_id="pillbottle",
            scene_revision="native-test-1",
            transform_revision="native-static",
            gripper=GRIPPER,
            minimum_table_clearance_m=float(GATE["minimum_table_clearance_m"]),
        ),
        observation,
    )
    assert native.succeeded
    assert report.status is PipelineStatus.NATIVE_PENDING
    assert not report.executed
    assert report.target_id == "pillbottle"
    assert report.destination_id == "pad"
    assert set(report.candidate_ids) == {c.candidate_id for c in native.candidates}
    assert report.candidate_count == len(native.candidates)
    assert digest == sha256(command.encode("utf-8")).hexdigest()


def test_invalid_missing_destination_refused():
    observation = scene()
    observation = SceneSnapshot(
        observation.scene_revision, observation.transform_revision,
        observation.frame_id, observation.objects, (),
        observation.table_height_m, observation.metadata,
    )
    command = "Gắp chai thuốc và đặt lên tấm đệm"
    semantic, digest = compile_native_command(command, "vi", "native-test-1")
    report = UnifiedSprint1Pipeline().run(
        command=command, language=semantic.source_language,
        scene=observation, semantic=semantic,
        semantic_command_sha256=digest, gripper=GRIPPER,
    )
    assert report.status is PipelineStatus.REJECTED
    assert report.stage == "grounding_or_grasp"
