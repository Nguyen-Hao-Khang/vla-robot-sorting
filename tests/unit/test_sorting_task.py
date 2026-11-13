"""Procedural sorting scenes and command generation (tools/twin/sorting_task.py)."""

from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))

import sorting_task as st  # noqa: E402
import twin_scene as ts  # noqa: E402


def _scenes(n: int = 200, seed: int = 0):
    rng = random.Random(seed)
    return [st.sample_scene(rng) for _ in range(n)]


def test_scene_counts_and_uniqueness() -> None:
    for scene in _scenes():
        assert 2 <= len(scene.objects) <= 4
        assert 2 <= len(scene.containers) <= 3
        triples = {(o.shape, o.colour, o.size) for o in scene.objects}
        assert len(triples) == len(scene.objects)
        styles = {(c.colour, c.outline, c.depth) for c in scene.containers}
        assert len(styles) == len(scene.containers)


def test_objects_keep_minimum_gap_inside_area() -> None:
    (x0, x1), (y0, y1) = st.OBJECT_AREA
    for scene in _scenes():
        for a, b in itertools.combinations(scene.objects, 2):
            assert math.dist(a.xy, b.xy) >= st.MIN_OBJECT_GAP
        for o in scene.objects:
            assert x0 <= o.xy[0] <= x1 and y0 <= o.xy[1] <= y1


def test_object_references_are_unique_and_minimal() -> None:
    for scene in _scenes():
        for target in scene.objects:
            refs = st.object_references(scene, target.name)
            assert refs, target.name  # the full attribute triple is always unique
            for ref in refs:
                matches = [
                    o.name for o in scene.objects if all(o.attribute(c) == v for c, v in ref)
                ]
                assert matches == [target.name]
                for c, _ in ref:  # dropping any attribute makes the reference ambiguous
                    smaller = [(k, v) for k, v in ref if k != c]
                    if smaller:
                        others = [
                            o for o in scene.objects if all(o.attribute(k) == v for k, v in smaller)
                        ]
                        assert len(others) > 1


def test_container_references_identify_one_container() -> None:
    for scene in _scenes():
        for target in scene.containers:
            for attr, value in st.container_references(scene, target.name):
                hits = [c for c in scene.containers if getattr(c, attr) == value]
                assert hits == [target]


def test_commands_name_valid_targets_in_both_languages() -> None:
    rng = random.Random(4)
    kinds, languages = set(), set()
    for scene in _scenes(100, seed=3):
        for language in ("en", "vi"):
            cmd = st.make_command(scene, rng, language, class_share=0.5)
            assert cmd is not None
            kinds.add(cmd.kind)
            languages.add(cmd.language)
            assert cmd.destination in {c.name for c in scene.containers}
            for name in cmd.targets:
                obj = scene.object(name)
                assert all(obj.attribute(c) == v for c, v in cmd.object_reference)
            if cmd.kind == "object":
                assert len(cmd.targets) == 1
            if language == "vi":
                assert any(ord(ch) > 127 for ch in cmd.text)
    assert kinds == {"object", "class"} and languages == {"en", "vi"}


def test_commands_respect_available_objects_and_open_containers() -> None:
    scene = _scenes(1, seed=7)[0]
    first = scene.objects[0].name
    only = scene.containers[-1].name
    rng = random.Random(1)
    for _ in range(50):
        cmd = st.make_command(scene, rng, "en", 0.5, available={first}, open_containers={only})
        assert cmd is not None and cmd.targets == (first,) and cmd.destination == only
    assert st.make_command(scene, rng, "en", available=set()) is None


def test_example_phrases() -> None:
    ref = (("colour", "red"), ("shape", "cylinder"))
    assert st.describe_object(ref, "en", definite=True) == "the red cylinder"
    assert st.describe_object(ref, "vi", definite=True) == "khối trụ đỏ"
    assert st.describe_object((("colour", "red"),), "en", definite=False) == "a red object"
    assert st.describe_object((("colour", "red"),), "vi", definite=False) == "một vật màu đỏ"
    assert st.describe_object((("shape", "cube"),), "en", definite=False) == "a cube"
    box = st.SceneContainer("c", "white", "round", "deep", "left", (0.0, 0.0))
    assert st.describe_container(("outline", "round"), box, "en") == "round bin"
    assert st.describe_container(("depth", "deep"), box, "vi") == "khay sâu"
    assert st.describe_container(("position", "left"), box, "vi") == "thùng bên trái"


def test_to_level_builds_matching_specs() -> None:
    scene = _scenes(1, seed=11)[0]
    level = st.to_level(scene)
    assert {o.name for o in level.objects} == {o.name for o in scene.objects}
    assert level.spawn_xy() == {o.name: o.xy for o in scene.objects}
    for spec, src in zip(level.containers, scene.containers, strict=True):
        assert spec.kind == ("bin" if src.depth == "deep" else "tray")
        assert spec.outline == ("round" if src.outline == "round" else "rect")
        assert spec.capacity == 2


def test_round_container_inside_and_walls() -> None:
    c = ts.ContainerSpec(
        "r", "tray", "white", (1, 1, 1), (0.0, 0.0), (0.05, 0.05), 0.02, outline="round"
    )
    assert ts.is_inside(c, (0.03, 0.03, 0.75), 0.74)
    assert not ts.is_inside(c, (0.05, 0.05, 0.75), 0.74)  # inside the square, outside the circle
    boxes = ts.container_boxes(c)
    assert len(boxes) == 1 + ts.ROUND_SEGMENTS
    for (cx, cy, _), _half in boxes[1:]:
        assert math.isclose(math.hypot(cx, cy), 0.05 + ts.WALL_T / 2, rel_tol=1e-6)


def test_household_scenes_have_a_look_alike_pair_and_identity_references() -> None:
    for seed in range(50):
        scene = st.sample_scene(random.Random(seed), family="household")
        assert scene.family == "household" and 2 <= len(scene.objects) <= 4
        categories = [o.category for o in scene.objects]
        assert any(categories.count(c) == 2 for c in categories)
        assert len({o.name for o in scene.objects}) == len(scene.objects)
        for o in scene.objects:
            refs = st.object_references(scene, o.name)
            assert (("item", o.item),) in refs
            if categories.count(o.category) == 2:
                assert (("category", o.category),) not in refs


def test_household_commands_use_item_and_category_phrases() -> None:
    scene = st.Scene(
        tuple(
            st.SceneObject(
                item, cat, "", "", (0.1 * i, 0.0), 0.0, item=item, category=cat, family="household"
            )
            for i, (item, cat) in enumerate(
                [
                    ("fresh milk carton", "milk carton"),
                    ("soy milk carton", "milk carton"),
                    ("box 2", "numbered box"),
                ]
            )
        ),
        st.sample_scene(random.Random(0)).containers,
    )
    assert st.describe_object((("item", "soy milk carton"),), "vi", True) == "hộp sữa đậu nành"
    assert st.describe_object((("item", "box 2"),), "en", True) == "the box number 2"
    assert st.describe_object((("category", "milk carton"),), "en", False) == "a milk carton"
    assert st.describe_object((("category", "bottle"),), "vi", False) == "một chai nước"
    rng = random.Random(3)
    for _ in range(40):
        cmd = st.make_command(scene, rng, rng.choice(["en", "vi"]), 0.5)
        if cmd.kind == "class":
            assert cmd.object_reference[0][0] == "category"
            if cmd.object_reference[0][1] == "milk carton":
                assert set(cmd.targets) == {"fresh milk carton", "soy milk carton"}
        else:
            assert len(cmd.targets) == 1


def test_household_to_level_uses_textured_packages() -> None:
    scene = st.sample_scene(random.Random(1), family="household")
    level = st.to_level(scene)
    assert [o.name for o in level.objects] == [o.name for o in scene.objects]
    assert all(o.visual is not None and o.mass >= 0.05 for o in level.objects)
    soda = st.household_spec("soft drink bottle")
    assert soda.visual == ("bottle", "soda") and soda.shape == "bottle"


def test_kit_scenes_use_only_real_kit_pieces() -> None:
    catalogue = set(st.KIT_CATALOGUE)
    for seed in range(50):
        scene = st.sample_scene(random.Random(seed), family="kit")
        assert scene.family == "kit"
        assert all((o.shape, o.size, o.colour) in catalogue for o in scene.objects)
        level = st.to_level(scene)
        for spec, o in zip(level.objects, scene.objects, strict=True):
            narrow = 2 * min(spec.half_extents[:2]) if spec.half_extents else spec.grip_width
            assert narrow <= 0.045, (o.name, narrow)
    assert st.describe_object((("colour", "natural"), ("shape", "cube")), "en", True) == (
        "the natural wood cube"
    )
    assert st.describe_object((("colour", "natural"),), "vi", False) == "một vật màu gỗ mộc"
    assert st.describe_object((("colour", "purple"), ("shape", "cuboid")), "vi", True) == (
        "khối hộp chữ nhật tím"
    )
