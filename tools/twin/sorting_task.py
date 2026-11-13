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
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import twin_scene as ts

SHAPES = ("cube", "cuboid", "cylinder", "triangular prism")
COLOURS = ("red", "yellow", "green", "blue")
SIZES = ("small", "large")
CONTAINER_COLOURS = ("white", "grey", "black", "brown")
OUTLINES = ("square", "rectangle", "round")
DEPTHS = ("shallow", "deep")

# Household extension (task specification 1.3): printed packages that differ by identity and
# fall into semantic categories. Each item: (category, English name, Vietnamese name).
HOUSEHOLD_ITEMS = {
    "fresh milk carton": ("milk carton", "fresh milk carton", "hộp sữa tươi"),
    "soy milk carton": ("milk carton", "soy milk carton", "hộp sữa đậu nành"),
    "box 1": ("numbered box", "box number 1", "hộp số 1"),
    "box 2": ("numbered box", "box number 2", "hộp số 2"),
    "water bottle": ("bottle", "water bottle", "chai nước suối"),
    "soft drink bottle": ("bottle", "soft drink bottle", "chai nước ngọt"),
}
HOUSEHOLD_CATEGORIES = {
    "milk carton": ("milk carton", "hộp sữa"),
    "numbered box": ("numbered box", "hộp có in số"),
    "bottle": ("bottle", "chai nước"),
}
FAMILIES = ("blocks", "kit", "household")
# Bottles come in two sizes; real bottles are wider than the gripper opens, so they are
# grasped by the neck and cap (twin_scene.BOTTLE_CAP_R). (radius, half height) in metres,
# estimated for 350 ml and 500 ml PET bottles and corrected once the bought bottles are measured.
# BOTTLE_MASS is the simulated mass: with the full-bottle masses (BOTTLE_MASS_REAL) every expert
# grasp by the cap slipped in RoboTwin (13 November 2026), so the simulation uses masses the
# simulated gripper holds. The real arm handles the real masses (task specification 2.2).
BOTTLE_ITEMS = ("water bottle", "soft drink bottle")
BOTTLE_SIZES = {"small": (0.029, 0.080), "large": (0.0325, 0.1025)}
BOTTLE_MASS = {"small": 0.12, "large": 0.15}
BOTTLE_MASS_REAL = {"small": 0.37, "large": 0.53}
SIZE_PAIR_SHARE = 0.25  # household scenes with a bottle that get its other size as well

# Real-kit family (task specification 1.4): the pieces of the commercial 30-piece wooden kit
# bought for the real arm: the shapes of the kit except the two arches. Colours: the kit paints
# small cuboids blue, red, green or purple and small prisms and half cylinders orange or
# yellow; the simulation offers every colour of KIT_COLOURS for these shapes (team decision,
# more variety for training), while large pieces, cubes and cylinders keep the kit's colours.
# Sizes are estimated from product photos and are corrected once the kit is measured.
KIT_COLOURS = ("red", "orange", "yellow", "green", "blue", "purple")
KIT_CATALOGUE = tuple(
    [("cube", "small", "natural"), ("cylinder", "small", "natural")]
    + [("cuboid", "small", c) for c in KIT_COLOURS]
    + [("cuboid", "large", c) for c in ("green", "natural")]
    + [("triangular prism", "small", c) for c in KIT_COLOURS]
    + [("triangular prism", "large", "natural")]
    + [("half cylinder", "small", c) for c in KIT_COLOURS]
)
KIT_DIMS = {  # half extents (m)
    ("cube", "small"): (0.015, 0.015, 0.015),
    ("cylinder", "small"): (0.015, 0.015, 0.015),
    ("cuboid", "small"): (0.015, 0.0075, 0.015),  # 3 x 1.5 x 3 cm, standing
    ("cuboid", "large"): (0.030, 0.015, 0.0075),  # 6 x 3 x 1.5 cm bar, lying
    # The kit's prisms are right-angled (base 4.2 / 6 cm) and lie on their base, so the real
    # gripper closes on the triangular faces 3 cm apart. The simulator's prism is equilateral
    # and stands on a triangular face (3 cm tall); the large one is cut to 5.2 cm sides so its
    # side-face grasp (1.5 x circumradius = 4.5 cm) stays in the range that holds reliably.
    ("triangular prism", "small"): (0.0242, 0.0242, 0.015),  # 4.2 cm sides
    ("triangular prism", "large"): (0.0300, 0.0300, 0.015),  # 5.2 cm sides
    ("half cylinder", "small"): (0.020, 0.015, 0.010),  # 4 cm across, 3 cm thick, 2 cm tall
}
HOUSEHOLD_MASS = {  # full packages (team decision: drinks are not emptied); bottles: BOTTLE_MASS
    "fresh milk carton": 0.20,
    "soy milk carton": 0.20,
    "box 1": 0.05,
    "box 2": 0.05,
}
HOUSEHOLD_CRITERIA = ("category", "item", "size")
EN_COLOUR = {"natural": "natural wood"}
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
    "orange": (0.95, 0.45, 0.10),
    "purple": (0.45, 0.20, 0.60),
    "natural": (0.85, 0.72, 0.55),
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
    "half cylinder": "khối bán trụ",
    "red": "đỏ",
    "yellow": "vàng",
    "green": "xanh lá",
    "blue": "xanh dương",
    "white": "trắng",
    "grey": "xám",
    "black": "đen",
    "brown": "nâu",
    "orange": "cam",
    "purple": "tím",
    "natural": "gỗ mộc",
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
    item: str = ""  # household items only: the key in HOUSEHOLD_ITEMS
    category: str = ""
    family: str = "blocks"

    def attribute(self, criterion: str) -> str:
        return {
            "colour": self.colour,
            "shape": self.shape,
            "size": self.size,
            "item": self.item,
            "category": self.category,
        }[criterion]


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

    @property
    def family(self) -> str:
        return self.objects[0].family

    @property
    def criteria(self) -> tuple[str, ...]:
        """Attributes that object references may use in this scene."""
        return HOUSEHOLD_CRITERIA if self.family == "household" else CRITERIA

    @property
    def class_criteria(self) -> tuple[str, ...]:
        """Criteria that class commands may sort by in this scene."""
        return ("category",) if self.family == "household" else CRITERIA


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
    family: str = "blocks",
) -> Scene:
    """Random scene: unique objects, containers that differ pairwise.

    ``blocks`` draws (shape, colour, size) triples; ``household`` draws distinct printed
    packages from HOUSEHOLD_ITEMS. Both use the same placement and container rules.
    """
    count = rng.randint(*n_objects)
    if family == "household":
        # At least one look-alike pair (both items of a category), so that identity, not
        # just category, decides which object a command means.
        pair_category = rng.choice(sorted(HOUSEHOLD_CATEGORIES))
        pair = sorted(i for i, v in HOUSEHOLD_ITEMS.items() if v[0] == pair_category)
        rest = sorted(i for i in HOUSEHOLD_ITEMS if i not in pair)
        items = pair + rng.sample(rest, count - len(pair))
        rng.shuffle(items)
        positions = _sample_positions(rng, count)
        objects = _household_objects(
            items, positions, [rng.uniform(-math.pi, math.pi) for _ in items], rng
        )
    elif family == "kit":
        pieces = rng.sample(KIT_CATALOGUE, count)
        positions = _sample_positions(rng, count)
        objects = tuple(
            SceneObject(
                object_name(shape, colour, size),
                shape,
                colour,
                size,
                xy,
                rng.uniform(-math.pi, math.pi),
                family="kit",
            )
            for (shape, size, colour), xy in zip(pieces, positions, strict=True)
        )
    elif family == "blocks":
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
    else:
        raise ValueError(f"unknown object family {family!r}")
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


def _household_objects(
    items: list[str], positions: list, yaws: list[float], rng: random.Random
) -> tuple[SceneObject, ...]:
    """Household scene objects; bottles get a size and sometimes their other size as well.

    Sizes come from a separate generator seeded by ``rng``'s state, so the items, positions
    and containers of a scene are the same as before bottle sizes existed (the reviewed
    household test commands keep their scenes).
    """
    sizes = random.Random(repr(rng.getstate()))
    objects = []
    for item, xy, yaw in zip(items, positions, yaws, strict=True):
        size = sizes.choice(sorted(BOTTLE_SIZES)) if item in BOTTLE_ITEMS else ""
        objects.append(_household_object(item, size, xy, yaw, item))
    bottles = [o for o in objects if o.size]
    if bottles and len(objects) < 4 and sizes.random() < SIZE_PAIR_SHARE:
        first = sizes.choice(bottles)
        other = "large" if first.size == "small" else "small"
        xy = _free_position(sizes, [o.xy for o in objects])
        if xy is not None:
            index = objects.index(first)
            objects[index] = replace(first, name=f"{first.size} {first.item}")
            twin = _household_object(
                first.item, other, xy, sizes.uniform(-math.pi, math.pi), f"{other} {first.item}"
            )
            objects.append(twin)
    return tuple(objects)


def _household_object(item: str, size: str, xy, yaw: float, name: str) -> SceneObject:
    category = HOUSEHOLD_ITEMS[item][0]
    return SceneObject(
        name, category, "", size, xy, yaw, item=item, category=category, family="household"
    )


def _free_position(rng: random.Random, taken: list) -> tuple[float, float] | None:
    (x0, x1), (y0, y1) = OBJECT_AREA
    for _ in range(200):
        p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        if all(math.dist(p, q) >= MIN_OBJECT_GAP for q in taken):
            return p
    return None


def _unique(values: list[tuple[str, ...]], index: int) -> bool:
    return values.count(values[index]) == 1


def object_references(scene: Scene, name: str) -> list[tuple[tuple[str, str], ...]]:
    """Minimal attribute subsets that pick out ``name`` among the scene objects."""
    target = scene.object(name)
    index = scene.objects.index(target)
    found: list[tuple[tuple[str, str], ...]] = []
    criteria = scene.criteria
    for r in range(1, len(criteria) + 1):
        for subset in itertools.combinations(criteria, r):
            if any(
                set(prev_keys) <= set(subset) for prev_keys in ([k for k, _ in f] for f in found)
            ):
                continue
            if any(target.attribute(c) == "" for c in subset):
                continue  # e.g. "size" for a carton
            if subset == ("size",) and scene.family == "household":
                continue  # "the large one" alone does not name a household object
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
    if "item" in attrs or "category" in attrs:
        if "item" in attrs:
            en, vi = HOUSEHOLD_ITEMS[attrs["item"]][1:]
        else:
            en, vi = HOUSEHOLD_CATEGORIES[attrs["category"]]
        if "size" in attrs:
            en, vi = f"{attrs['size']} {en}", f"{vi} {VI_WORDS[attrs['size']]}"
        if language == "vi":
            return vi if definite else f"một {vi}"
        if definite:
            return f"the {en}"
        return f"an {en}" if en[0] in "aeiou" else f"a {en}"
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
    colour = EN_COLOUR.get(attrs.get("colour", ""), attrs.get("colour"))
    words = [w for w in (attrs.get("size"), colour, noun) if w]
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
    bank: dict[str, dict[str, list[str]]] | None = None,
    bank_share: float = 0.7,
) -> Command | None:
    """Sample one object or class command with a uniquely identified destination.

    ``available`` limits the picks to objects still on the table and ``open_containers`` to
    containers with free room; references stay unique among all scene objects and containers.
    ``bank`` holds generated templates (``load_template_bank``); a template from it is used
    with probability ``bank_share``, otherwise a built-in one. Returns None when no valid
    command exists.
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
            {
                (c, o.attribute(c))
                for c in scene.class_criteria
                for o in scene.objects
                if o.name in on_table
            }
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
        templates = VI_OBJECT_TEMPLATES if language == "vi" else EN_OBJECT_TEMPLATES
        kind = "object"
    generated = (bank or {}).get(language, {}).get(kind, [])
    if generated and rng.random() < bank_share:
        if language == "en":
            dest_phrase = f"the {dest_phrase}"
        text = rng.choice(generated).format(obj=obj_phrase, dest=dest_phrase)
    else:
        if language == "en" and kind == "object":
            obj_phrase = obj_phrase[len("the ") :]
        text = rng.choice(templates).format(obj=obj_phrase, dest=dest_phrase)
    return Command(text, language, kind, targets, dest.name, reference, container_ref)


def load_template_bank(path: Path) -> dict[str, dict[str, list[str]]]:
    """Generated templates from configs/command_templates.json (empty if the file is absent)."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("templates", {})


def household_spec(item: str, size: str = "") -> ts.ObjectSpec:
    """Simulator object for a household item: the textured packages of scene levels 6 and 7,
    plus a soft drink bottle that shares the water bottle's geometry. Bottles take the radius,
    height and mass of their size (BOTTLE_SIZES)."""
    spec = _household_base(item)
    if item in BOTTLE_ITEMS:
        radius, half_height = BOTTLE_SIZES[size or "small"]
        return replace(
            spec, radius=radius, half_height=half_height, mass=BOTTLE_MASS[size or "small"]
        )
    return spec


def _household_base(item: str) -> ts.ObjectSpec:
    specs = {o.name: o for level in (ts.LEVELS[6], ts.LEVELS[7]) for o in level.objects}
    if item == "soft drink bottle":
        water = specs["water bottle"]
        return replace(
            water,
            name=item,
            color_name="dark brown",
            rgb=(0.35, 0.15, 0.08),
            aliases=("chai nước ngọt",),
            kind="small soft drink bottle with a red cap",
            visual=("bottle", "soda"),
            keywords=("soft", "soda"),
        )
    return replace(specs[item], mass=HOUSEHOLD_MASS.get(item, specs[item].mass))


def to_level(scene: Scene, number: int = 0) -> ts.Level:
    """Build a twin_scene.Level so the simulator can create this scene."""
    objects = []
    for o in scene.objects:
        if o.item:
            objects.append(replace(household_spec(o.item, o.size), name=o.name))
            continue
        dims = KIT_DIMS if o.family == "kit" else OBJECT_DIMS
        hx, hy, hz = dims[(o.shape, o.size)]
        rgb = RGB[o.colour]
        if o.shape == "cube":
            spec = ts.ObjectSpec(o.name, "cube", o.colour, rgb, hx, hz)
        elif o.shape == "cuboid":
            spec = ts.ObjectSpec(o.name, "box", o.colour, rgb, hx, hz, half_extents=(hx, hy, hz))
        elif o.shape == "cylinder":
            spec = ts.ObjectSpec(o.name, "cylinder", o.colour, rgb, hx, hz)
        elif o.shape == "half cylinder":
            spec = ts.ObjectSpec(
                o.name, "halfcyl", o.colour, rgb, hx, hz, half_extents=(hx, hy, hz)
            )
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
