from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    DestinationRegion,
    GeometrySource,
    SceneObject,
    SceneSnapshot,
)
from vla_robot_sorting.semantics.grounding import (
    CatalogDestinationGrounder,
    CatalogGrounder,
    GroundingFailureCode,
    GroundingRequest,
)


def make_scene() -> SceneSnapshot:
    def obj(object_id: str, labels: tuple[str, ...], x: float) -> SceneObject:
        return SceneObject(
            object_id=object_id,
            pose=Pose6D((x, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0), "world"),
            geometry=BoxGeometry((0.04, 0.04, 0.08)),
            geometry_source=GeometrySource.ESTIMATED_PROXY,
            semantic_labels=labels,
            identity_confidence=0.85,
        )

    return SceneSnapshot(
        scene_revision="scene-7",
        transform_revision="tf-2",
        frame_id="world",
        objects=(
            obj("red-block", ("red block", "khối đỏ"), -0.1),
            obj("blue-block", ("blue block", "khối xanh"), 0.1),
        ),
        destinations=(
            DestinationRegion(
                destination_id="left-tray",
                pose=Pose6D((-0.2, 0.2, 0.05), (0.0, 0.0, 0.0, 1.0), "world"),
                tolerance_xyz_m=(0.03, 0.03, 0.03),
                semantic_labels=("left tray", "khay trái", "tray"),
            ),
            DestinationRegion(
                destination_id="right-tray",
                pose=Pose6D((0.2, 0.2, 0.05), (0.0, 0.0, 0.0, 1.0), "world"),
                tolerance_xyz_m=(0.03, 0.03, 0.03),
                semantic_labels=("right tray", "khay phải", "tray"),
            ),
        ),
    )


def test_catalog_grounder_supports_bilingual_aliases() -> None:
    scene = make_scene()

    result = CatalogGrounder().ground(
        GroundingRequest("khối đỏ", "scene-7", language="vi"),
        scene,
    )

    assert result.succeeded
    assert result.object_id == "red-block"
    assert result.confidence == 0.85


def test_catalog_grounder_reports_ambiguity_separately() -> None:
    scene = make_scene()

    result = CatalogGrounder().ground(
        GroundingRequest("block", "scene-7", language="en"),
        scene,
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is GroundingFailureCode.AMBIGUOUS
    assert set(result.failure.candidate_ids) == {"red-block", "blue-block"}


def test_catalog_grounder_rejects_stale_scene() -> None:
    scene = make_scene()

    result = CatalogGrounder().ground(
        GroundingRequest("red block", "old-scene"),
        scene,
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is GroundingFailureCode.STALE_SCENE


def test_catalog_grounder_reports_not_found() -> None:
    scene = make_scene()

    result = CatalogGrounder().ground(
        GroundingRequest("yellow mug", "scene-7"),
        scene,
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is GroundingFailureCode.NOT_FOUND


def test_destination_grounder_supports_bilingual_aliases() -> None:
    scene = make_scene()

    result = CatalogDestinationGrounder().ground(
        GroundingRequest("khay trái", "scene-7", language="vi"),
        scene,
    )

    assert result.succeeded
    assert result.destination_id == "left-tray"
    assert result.confidence == 1.0


def test_destination_grounder_reports_ambiguity_separately() -> None:
    scene = make_scene()

    result = CatalogDestinationGrounder().ground(
        GroundingRequest("tray", "scene-7", language="en"),
        scene,
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is GroundingFailureCode.AMBIGUOUS
    assert set(result.failure.candidate_ids) == {"left-tray", "right-tray"}


def test_destination_grounder_rejects_stale_scene() -> None:
    scene = make_scene()

    result = CatalogDestinationGrounder().ground(
        GroundingRequest("left tray", "old-scene"),
        scene,
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is GroundingFailureCode.STALE_SCENE
