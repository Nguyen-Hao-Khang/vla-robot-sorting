"""Procedural sorting scenes and bilingual commands (docs/task-spec.md).

Each scene draws 2-4 objects and 2-3 containers with random attributes:

* objects: shape (cube, cuboid, cylinder, triangular prism), colour (red, yellow, green, blue)
  and size (small, large); no two objects share all three attributes;
* containers: colour (white, grey, black, brown), outline (square, rectangle, round) and depth
  (shallow, deep), placed left / middle / right in the container row.

Commands come in two forms:

* object commands name exactly one object, using the shortest attribute set that identifies it
  among the objects in the scene ("put the red cylinder in the round tray");
* class commands name a class under one criterion (colour, shape or size), and any member of
  the class is a correct pick ("put a red object in the left tray").

The container is named by one attribute that is unique among the containers in the scene:
position, colour, outline or depth. Everything here is pure Python; ``to_level`` turns a scene
into a ``twin_scene.Level`` that the simulator can build.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

import twin_scene as ts

SHAPES = ("cube", "cuboid", "cylinder", "triangular prism")
COLOURS = ("red", "yellow", "green", "blue")
SIZES = ("small", "large")
CONTAINER_COLOURS = ("white", "grey", "black", "brown")
OUTLINES = ("square", "rectangle", "round")
DEPTHS = ("shallow", "deep")
CRITERIA = ("colour", "shape", "size")

RGB = {
    "red": (0.85, 0.10, 0.10),
    "yellow": (1.00, 0.85, 0.10),
    "green": (0.10, 0.70, 0.20),
    "blue": (0.10, 0.30, 0.90),
    "white": (0.95, 0.95, 0.95),
    "grey": (0.55, 0.55, 0.60),
    "black": (0.08, 0.08, 0.08),
    "brown": (0.55, 0.35, 0.20),
}

# Object geometry (metres): half extents chosen to fit the two-finger gripper.
OBJECT_DIMS = {
    ("cube", "small"): (0.015, 0.015, 0.015),
    ("cube", "large"): (0.022, 0.022, 0.022),
    ("cuboid", "small"): (0.015, 0.015, 0.025),
    ("cuboid", "large"): (0.020, 0.020, 0.035),
    ("cylinder", "small"): (0.015, 0.015, 0.020),
    ("cylinder", "large"): (0.020, 0.020, 0.030),
    ("triangular prism", "small"): (0.020, 0.020, 0.015),
    ("triangular prism", "large"): (0.026, 0.026, 0.022),
}

# Container geometry: inner half sizes, wall heights, x positions of the row slots.
INNER_HALF = {"square": (0.050, 0.050), "rectangle": (0.070, 0.045), "round": (0.050, 0.050)}
WALL_HEIGHT = {"shallow": 0.020, "deep": 0.060}
CONTAINER_X = {2: (-0.12, 0.12), 3: (-0.18, 0.0, 0.18)}
POSITION_NAMES = {2: ("left", "right"), 3: ("left", "middle", "right")}

# Free area for objects (x range, y range) and the minimum centre distance between objects.
OBJECT_AREA = ((-0.22, 0.22), (-0.07, 0.01))
MIN_OBJECT_GAP = 0.11

EN_OBJECT_TEMPLATES = (
    "put the {obj} in the {dest}",
    "pick up the {obj} and place it in the {dest}",
    "move the {obj} to the {dest}",
    "place the {obj} into the {dest}",
)
EN_CLASS_TEMPLATES = (
    "put {obj} in the {dest}",
    "pick {obj} and place it in the {dest}",
    "move {obj} to the {dest}",
)
VI_OBJECT_TEMPLATES = (
    "gắp {obj} bỏ vào {dest}",
    "đặt {obj} vào {dest}",
    "chuyển {obj} sang {dest}",
    "cho {obj} vào {dest}",
)
VI_CLASS_TEMPLATES = (
    "gắp {obj} bỏ vào {dest}",
    "lấy {obj} đặt vào {dest}",
    "chuyển {obj} sang {dest}",
)

VI_WORDS = {
    "cube": "khối lập phương",
    "cuboid": "khối hộp chữ nhật",
    "cylinder": "khối trụ",
    "triangular prism": "khối lăng trụ tam giác",
    "red": "đỏ",
    "yellow": "vàng",
    "green": "xanh lá",
    "blue": "xanh dương",
    "white": "trắng",
    "grey": "xám",
    "black": "đen",
    "brown": "nâu",
    "small": "nhỏ",
    "large": "lớn",
    "square": "vuông",
    "rectangle": "chữ nhật",
    "round": "tròn",
    "shallow": "nông",
    "deep": "sâu",
    "left": "bên trái",
    "middle": "ở giữa",
    "right": "bên phải",
}


@dataclass(frozen=True)
class SceneObject:
    name: str
    shape: str
    colour: str
    size: str
    xy: tuple[float, float]
    yaw: float

    def attribute(self, criterion: str) -> str:
        return {"colour": self.colour, "shape": self.shape, "size": self.size}[criterion]


@dataclass(frozen=True)
class SceneContainer:
    name: str
    colour: str
    outline: str
    depth: str
    position: str
    xy: tuple[float, float]


@dataclass(frozen=True)
class Scene:
    objects: tuple[SceneObject, ...]
    containers: tuple[SceneContainer, ...]

    def container(self, name: str) -> SceneContainer:
        return next(c for c in self.containers if c.name == name)

    def object(self, name: str) -> SceneObject:
        return next(o for o in self.objects if o.name == name)


@dataclass(frozen=True)
class Command:
    """One instruction. ``targets`` lists every object that is a correct pick."""

    text: str
    language: str
    kind: str  # "object" | "class"
    targets: tuple[str, ...]
    destination: str
    object_reference: tuple[tuple[str, str], ...]
    container_reference: tuple[str, str]


def object_name(shape: str, colour: str, size: str) -> str:
    return f"{size} {colour} {shape}"


def _sample_positions(rng: random.Random, count: int) -> list[tuple[float, float]]:
    (x0, x1), (y0, y1) = OBJECT_AREA
    for _ in range(1000):
        points: list[tuple[float, float]] = []
        for _ in range(count):
            for _ in range(200):
                p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
                if all(math.dist(p, q) >= MIN_OBJECT_GAP for q in points):
                    points.append(p)
                    break
        if len(points) == count:
            return points
    raise RuntimeError(f"cannot place {count} objects with gap {MIN_OBJECT_GAP} m")


def sample_scene(
    rng: random.Random,
    n_objects: tuple[int, int] = (2, 4),
    n_containers: tuple[int, int] = (2, 3),
) -> Scene:
    """Random scene: unique object attribute triples, containers that differ pairwise."""
    count = rng.randint(*n_objects)
    triples = rng.sample(list(itertools.product(SHAPES, COLOURS, SIZES)), count)
    positions = _sample_positions(rng, count)
    objects = tuple(
        SceneObject(
            object_name(shape, colour, size),
            shape,
            colour,
            size,
            xy,
            rng.uniform(-math.pi, math.pi),
        )
        for (shape, colour, size), xy in zip(triples, positions, strict=True)
    )
    k = rng.randint(*n_containers)
    while True:
        styles = [
            (rng.choice(CONTAINER_COLOURS), rng.choice(OUTLINES), rng.choice(DEPTHS))
            for _ in range(k)
        ]
        if len(set(styles)) == k:
            break
    containers = tuple(
        SceneContainer(
            f"{position} {colour} {depth} {outline} container",
            colour,
            outline,
            depth,
            position,
            (x, ts.CONTAINER_ROW_Y),
        )
        for (colour, outline, depth), position, x in zip(
            styles, POSITION_NAMES[k], CONTAINER_X[k], strict=True
        )
    )
    return Scene(objects, containers)


def _unique(values: list[tuple[str, ...]], index: int) -> bool:
    return values.count(values[index]) == 1


def object_references(scene: Scene, name: str) -> list[tuple[tuple[str, str], ...]]:
    """Minimal attribute subsets that pick out ``name`` among the scene objects."""
    target = scene.object(name)
    index = scene.objects.index(target)
    found: list[tuple[tuple[str, str], ...]] = []
    for r in (1, 2, 3):
        for subset in itertools.combinations(CRITERIA, r):
            if any(
                set(prev_keys) <= set(subset) for prev_keys in ([k for k, _ in f] for f in found)
            ):
                continue
            values = [tuple(o.attribute(c) for c in subset) for o in scene.objects]
            if _unique(values, index):
                found.append(tuple((c, target.attribute(c)) for c in subset))
    return found


def container_references(scene: Scene, name: str) -> list[tuple[str, str]]:
    """Single attributes (position, colour, outline, depth) unique to container ``name``."""
    target = scene.container(name)
    refs = [("position", target.position)]
    for attr in ("colour", "outline", "depth"):
        values = [getattr(c, attr) for c in scene.containers]
        if values.count(getattr(target, attr)) == 1:
            refs.append((attr, getattr(target, attr)))
    return refs


def class_members(scene: Scene, criterion: str, value: str) -> tuple[str, ...]:
    return tuple(o.name for o in scene.objects if o.attribute(criterion) == value)


def describe_object(reference: tuple[tuple[str, str], ...], language: str, definite: bool) -> str:
    """Noun phrase for an object reference, e.g. 'the small red cube' / 'khối lập phương đỏ nhỏ'."""
    attrs = dict(reference)
    shape = attrs.get("shape")
    if language == "vi":
        noun = VI_WORDS[shape] if shape else "vật"
        words = [noun]
        if "colour" in attrs:
            words.append(("" if shape else "màu ") + VI_WORDS[attrs["colour"]])
        if "size" in attrs:
            words.append(VI_WORDS[attrs["size"]])
        phrase = " ".join(words)
        return phrase if definite else f"một {phrase}"
    noun = shape or "object"
    words = [w for w in (attrs.get("size"), attrs.get("colour"), noun) if w]
    phrase = " ".join(words)
    if definite:
        return f"the {phrase}"
    return f"an {phrase}" if phrase[0] in "aeiou" else f"a {phrase}"


def describe_container(reference: tuple[str, str], container: SceneContainer, language: str) -> str:
    """Container phrase that uses only the referenced attribute (plus the noun)."""
    attr, value = reference
    deep = container.depth == "deep"
    if language == "vi":
        noun = "thùng" if deep and attr != "depth" else "khay"
        if attr == "position":
            return f"{noun} {VI_WORDS[value]}"
        if attr == "outline":
            return f"{noun} hình {VI_WORDS[value]}" if value != "round" else f"{noun} tròn"
        return f"{noun} {VI_WORDS[value]}"
    noun = "bin" if deep and attr != "depth" else "tray"
    if attr == "position":
        return f"{value} {noun}" if value != "middle" else f"middle {noun}"
    if attr == "outline":
        return f"{'rectangular' if value == 'rectangle' else value} {noun}"
    return f"{value} {noun}"


def make_command(
    scene: Scene,
    rng: random.Random,
    language: str,
    class_share: float = 0.3,
    available: set[str] | None = None,
    open_containers: set[str] | None = None,
) -> Command | None:
    """Sample one object or class command with a uniquely identified destination.

    ``available`` limits the picks to objects still on the table and ``open_containers`` to
    containers with free room; references stay unique among all scene objects and containers.
    Returns None when no valid command exists.
    """
    on_table = {o.name for o in scene.objects} if available is None else set(available)
    dests = [c for c in scene.containers if open_containers is None or c.name in open_containers]
    if not on_table or not dests:
        return None
    dest = rng.choice(dests)
    container_ref = rng.choice(container_references(scene, dest.name))
    dest_phrase = describe_container(container_ref, dest, language)
    if rng.random() < class_share:
        options = sorted(
            {(c, o.attribute(c)) for c in CRITERIA for o in scene.objects if o.name in on_table}
        )
        criterion, value = rng.choice(options)
        reference = ((criterion, value),)
        targets = tuple(t for t in class_members(scene, criterion, value) if t in on_table)
        obj_phrase = describe_object(reference, language, definite=False)
        templates = VI_CLASS_TEMPLATES if language == "vi" else EN_CLASS_TEMPLATES
        kind = "class"
    else:
        target = rng.choice([o for o in scene.objects if o.name in on_table])
        reference = rng.choice(object_references(scene, target.name))
        targets = (target.name,)
        obj_phrase = describe_object(reference, language, definite=True)
        if language == "en":
            obj_phrase = obj_phrase[len("the ") :]
        templates = VI_OBJECT_TEMPLATES if language == "vi" else EN_OBJECT_TEMPLATES
        kind = "object"
    text = rng.choice(templates).format(obj=obj_phrase, dest=dest_phrase)
    return Command(text, language, kind, targets, dest.name, reference, container_ref)


def to_level(scene: Scene, number: int = 0) -> ts.Level:
    """Build a twin_scene.Level so the simulator can create this scene."""
    objects = []
    for o in scene.objects:
        hx, hy, hz = OBJECT_DIMS[(o.shape, o.size)]
        rgb = RGB[o.colour]
        if o.shape == "cube":
            spec = ts.ObjectSpec(o.name, "cube", o.colour, rgb, hx, hz)
        elif o.shape == "cuboid":
            spec = ts.ObjectSpec(o.name, "box", o.colour, rgb, hx, hz, half_extents=(hx, hy, hz))
        elif o.shape == "cylinder":
            spec = ts.ObjectSpec(o.name, "cylinder", o.colour, rgb, hx, hz)
        else:
            spec = ts.ObjectSpec(o.name, "prism", o.colour, rgb, hx, hz)
        objects.append(spec)
    containers = []
    for c in scene.containers:
        hx, hy = INNER_HALF[c.outline]
        pitch = 0.035 if c.outline == "rectangle" else 0.025
        containers.append(
            ts.ContainerSpec(
                c.name,
                "bin" if c.depth == "deep" else "tray",
                c.colour,
                RGB[c.colour],
                c.xy,
                (hx, hy),
                WALL_HEIGHT[c.depth],
                ((-pitch, 0.0), (pitch, 0.0)),
                outline="round" if c.outline == "round" else "rect",
            )
        )
    spawns = tuple((o.name, o.xy[0], o.xy[1]) for o in scene.objects)
    return ts.Level(
        number, "procedural sorting scene", tuple(objects), tuple(containers), spawns=spawns
    )
