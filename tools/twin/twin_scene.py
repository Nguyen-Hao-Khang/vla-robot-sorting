# ruff: noqa: E501
"""Scene levels for the interactive digital twin: objects, containers and geometry helpers.

Pure layout data and geometry live at module level so they can be unit-tested without a GPU
or SAPIEN; the ``build_*`` functions import SAPIEN lazily and are only used inside the
RoboTwin conda environment.

Frame: RoboTwin world, metres. The robot base sits behind the table at negative y facing +y;
x<0 is the operator's left. Objects stand in one row at y=OBJECT_ROW_Y, 12 cm apart (at 10 cm
the arm brushes neighbours); containers stand in a row closer to the
robot (y=CONTAINER_ROW_Y) with about 7 cm of free table between the two rows, the shallow
tray on the left and the deep bin pushed out to the right so the arm does not pass over a
tall object in the bin on its way to the object row. Levels 3-7 have no staging pad
between the tray and the bin (objects placed there were squeezed against both containers);
temporary spots for swaps are the free slots of the object row (the "table" destination).
Levels 6-7 use textured household-like objects (twin_assets.py) in look-alike pairs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

CONTAINER_ROW_Y = -0.20
OBJECT_ROW_Y = -0.04
OBJECT_SLOTS_X = (-0.18, -0.06, 0.06, 0.18)
FLOOR_T = 0.004
WALL_T = 0.006
FINGER_CLEARANCE = 0.005
PLACE_DIS = 0.02
MIN_DROP = 0.005  # release tall objects only 5 mm above the floor (they tip when dropped)
PLACE_FRAC_MAX = 0.24  # highest grasp used to lower a tall object deeper (straight body only)
TRANSIT_STEP = 0.10  # fallback hop length when a straight constrained move is refused
MIN_LIFT = 0.07
LIFT_ABOVE_RIM = 0.05
TRANSPORT_MARGIN = 0.05  # carried object's bottom clears every rim and object top by this
MIN_LIFT_DZ = 0.05
RETREAT_DZ = 0.10  # straight up after release, before going home
GRIPPER_REACH = 0.15  # end-effector pose is ~12 cm above the fingertips (+3 cm margin)
GRASP_KEEPOUT = 0.02  # free gap the open fingers need around the grasped object
MOVED_TOL = 0.02  # an object that should not move but moved more than this = collision
TOPPLED_DEG = 35.0
TABLE_BOUNDS = ((-0.40, 0.40), (-0.45, 0.30))
INSIDE_MARGIN = 0.008
GRASP_TOP_MARGIN = 0.008  # keep the grasp point at least this far below the object top
DEEP_WALL = 0.03  # containers with walls at least this tall need a raised grasp point
FINGER_DEPTH = 0.03  # how far the fingertips may reach below a deep container's rim
ALIGN_TOL = 0.003  # closed-loop place: correct only errors larger than this
ALIGN_MAX = 0.04  # ... and never move more than this in one correction
SPAWN_JITTER = 0.01
ROUND_SEGMENTS = 16
# Rotates SAPIEN's x-axis cylinder so that its axis points up (+z).
UPRIGHT_Q = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    shape: str  # cube | box | cylinder | prism | sphere | tetrahedron | bottle
    color_name: str
    rgb: tuple[float, float, float]
    radius: (
        float  # half side (cube), radius (cylinder/sphere/bottle), circumradius (prism/tetra base)
    )
    half_height: float
    half_extents: tuple[float, float, float] | None = None  # box only
    aliases: tuple[str, ...] = ()
    kind: str = ""  # what the object is, e.g. "fresh milk carton" (shown to the planner)
    visual: tuple[str, str] | None = None  # (asset kind, design) for a textured mesh
    keywords: tuple[str, ...] = ()  # words a correct VLM label must contain (look-alike levels)
    mass: float = 0.05  # kg; light (10 g) objects were flung around by the arm

    @property
    def footprint(self) -> float:
        """Radius of the smallest circle around the object's footprint."""
        if self.shape == "box" and self.half_extents:
            return math.hypot(self.half_extents[0], self.half_extents[1])
        if self.shape == "cube":
            return self.radius * math.sqrt(2)
        return self.radius

    @property
    def grip_width(self) -> float:
        """Widest span the fingers may have to close across at the grasp height."""
        if self.shape == "box" and self.half_extents:
            return 2 * max(self.half_extents[0], self.half_extents[1])
        if self.shape == "prism":
            return 1.5 * self.radius
        if self.shape == "tetrahedron":
            return self.radius * math.sqrt(3) / 2  # edge of the mid-height cross-section
        return 2 * self.radius

    def describe(self) -> str:
        cm = 100
        height = 2 * self.half_height * cm
        width = 2 * self.radius * cm
        if self.kind:
            if self.shape == "box" and self.half_extents:
                hx, hy, _ = self.half_extents
                size = f"{2 * hx * cm:.1f} x {2 * hy * cm:.1f} x {height:.1f} cm"
            elif self.shape == "sphere":
                size = f"{width:.1f} cm diameter"
            else:
                size = f"{width:.1f} cm diameter, {height:.1f} cm tall"
            text = f"{self.kind} ({self.color_name}), {size}"
        elif self.shape == "cube":
            text = f"{self.color_name} cube, {width:.1f} cm wide"
        elif self.shape == "box":
            hx, hy, _ = self.half_extents or (self.radius, self.radius, self.half_height)
            text = f"{self.color_name} box, {2 * hx * cm:.1f} x {2 * hy * cm:.1f} cm, {height:.1f} cm tall"
        elif self.shape in ("cylinder", "bottle"):
            text = f"{self.color_name} upright {self.shape}, {width:.1f} cm diameter, {height:.1f} cm tall"
        elif self.shape == "sphere":
            text = f"{self.color_name} ball, {width:.1f} cm diameter"
        elif self.shape == "tetrahedron":
            edge = self.radius * math.sqrt(3) * cm
            text = f"{self.color_name} regular tetrahedron, {edge:.1f} cm edges"
        else:
            side = self.radius * math.sqrt(3) * cm
            text = f"{self.color_name} triangular prism, {side:.1f} cm sides, {height:.1f} cm tall"
        if self.aliases:
            text += f" (also called: {', '.join(self.aliases)})"
        return text


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    kind: str  # tray | bin | pad
    color_name: str
    rgb: tuple[float, float, float]
    xy: tuple[float, float]
    inner_half: tuple[float, float]
    wall_height: float = 0.0
    slots: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    aliases: tuple[str, ...] = ()
    outline: str = "rect"  # rect | round (round: inner_half[0] is the inner radius)

    @property
    def capacity(self) -> int:
        return len(self.slots)

    def describe(self) -> str:
        w, d = (2 * v * 100 for v in self.inner_half)
        if self.kind == "pad":
            text = f"{self.color_name} staging pad (temporary spot, e.g. for swaps), room for 1"
        elif self.kind == "table":
            text = f"free spots on the table in the object row, room for {self.capacity}"
        else:
            shape = "wide shallow tray" if self.kind == "tray" else "deep bin"
            text = (
                f"{self.color_name} {shape}, inside {w:.0f}x{d:.0f} cm, walls "
                f"{self.wall_height * 100:.1f} cm, room for {self.capacity}"
            )
        if self.aliases:
            text += f" (also called: {', '.join(self.aliases)})"
        return text


@dataclass(frozen=True)
class Level:
    number: int
    title: str
    objects: tuple[ObjectSpec, ...]
    containers: tuple[ContainerSpec, ...]
    examples: tuple[str, ...] = field(default_factory=tuple)
    spawns: tuple[tuple[str, float, float], ...] = ()  # explicit (name, x, y); default: slots

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objects)

    @property
    def destination_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.containers)

    def object(self, name: str) -> ObjectSpec:
        return next(o for o in self.objects if o.name == name)

    def container(self, name: str) -> ContainerSpec:
        return next(c for c in self.containers if c.name == name)

    def spawn_xy(self) -> dict[str, tuple[float, float]]:
        if self.spawns:
            return {name: (x, y) for name, x, y in self.spawns}
        return {o.name: (OBJECT_SLOTS_X[i], OBJECT_ROW_Y) for i, o in enumerate(self.objects)}


RED, GREEN, BLUE = (0.85, 0.1, 0.1), (0.1, 0.7, 0.2), (0.1, 0.3, 0.9)
ORANGE, YELLOW, PURPLE = (1.0, 0.5, 0.0), (1.0, 0.85, 0.1), (0.55, 0.2, 0.8)

TRAY_ALIASES = ("shallow tray", "khay nông", "khay rộng")
BIN_ALIASES = ("deep bin", "deep tray", "tall bin", "khay sâu", "hộp sâu", "thùng sâu", "thùng cao")
PAD_ALIASES = ("staging area", "temporary spot", "ô trung chuyển", "chỗ tạm")

# 24 x 12 cm shallow tray (2 cm walls) on the left, 18 x 12 cm deep bin (7 cm walls) on the right;
# 13.8 cm of free table between their outer walls.
WIDE_TRAY = ContainerSpec(
    "wide tray",
    "tray",
    "grey",
    (0.6, 0.6, 0.65),
    (-0.17, CONTAINER_ROW_Y),
    (0.12, 0.06),
    0.02,
    ((-0.06, 0.0), (0.06, 0.0)),
    TRAY_ALIASES,
)
DEEP_BIN = ContainerSpec(
    "deep bin",
    "bin",
    "dark green",
    (0.1, 0.4, 0.3),
    (0.19, CONTAINER_ROW_Y),
    (0.09, 0.06),
    0.07,
    ((-0.04, 0.0), (0.04, 0.0)),
    BIN_ALIASES[1:],
)
# Between the tray (outer edge x=-0.044) and the bin (outer edge x=+0.094).
STAGING_MID = ContainerSpec(
    "staging pad",
    "pad",
    "light grey",
    (0.85, 0.85, 0.85),
    (0.02, CONTAINER_ROW_Y),
    (0.022, 0.022),
    aliases=PAD_ALIASES,
)
STAGING_ROW = replace(STAGING_MID, xy=(OBJECT_SLOTS_X[3], OBJECT_ROW_Y))
TABLE_SPOT = ContainerSpec(
    "table",
    "table",
    "white",
    (1.0, 1.0, 1.0),
    (0.0, OBJECT_ROW_Y),
    (0.24, 0.03),
    0.0,
    tuple((x, 0.0) for x in OBJECT_SLOTS_X),
    ("free spot on the table", "mặt bàn", "bàn", "ra ngoài", "bên ngoài"),
)


def _small_tray(
    name: str, color: str, rgb: tuple[float, float, float], x: float, vi: str
) -> ContainerSpec:
    return ContainerSpec(
        name, "tray", color, rgb, (x, CONTAINER_ROW_Y), (0.04, 0.04), 0.015, aliases=(vi,)
    )


def _cube(name: str, color: str, rgb: tuple[float, float, float], half: float = 0.02) -> ObjectSpec:
    return ObjectSpec(name, "cube", color, rgb, half, half, aliases=("khối vuông",))


def _cyl(name: str, color: str, rgb: tuple[float, float, float], radius: float) -> ObjectSpec:
    return ObjectSpec(name, "cylinder", color, rgb, radius, 0.03, aliases=("khối trụ",))


def tetra_radius(edge: float) -> float:
    return edge / math.sqrt(3)


def tetra_half_height(edge: float) -> float:
    return edge * math.sqrt(2.0 / 3.0) / 2


CARTON_HALF = (0.0225, 0.019, 0.05)  # 180 ml drink carton: 4.5 x 3.8 x 10 cm
BOX_HALF = (0.025, 0.025, 0.025)  # small cardboard box: 5 cm cube
BALL_R = 0.024  # toy-size balls: a real tennis ball (6.7 cm) would not fit the gripper
BOTTLE_R, BOTTLE_HALF_H = 0.021, 0.065  # small PET bottle: 4.2 cm wide, 13 cm tall

LEVELS: dict[int, Level] = {
    1: Level(
        1,
        "one object, one tray",
        (_cube("red cube", "red", RED),),
        (replace(WIDE_TRAY, xy=(0.0, CONTAINER_ROW_Y)),),
        ("put the red cube in the tray", "gắp khối đỏ bỏ vào khay"),
    ),
    2: Level(
        2,
        "three coloured cubes, three coloured trays",
        (
            _cube("red cube", "red", RED),
            _cube("green cube", "green", GREEN),
            _cube("blue cube", "blue", BLUE),
        ),
        (
            _small_tray("yellow tray", "yellow", (1.0, 0.8, 0.0), -0.16, "khay vàng"),
            _small_tray("white tray", "white", (0.95, 0.95, 0.95), 0.0, "khay trắng"),
            _small_tray("purple tray", "purple", (0.6, 0.2, 0.8), 0.16, "khay tím"),
            STAGING_ROW,
            replace(
                TABLE_SPOT,
                xy=(-0.06, OBJECT_ROW_Y),
                inner_half=(0.18, 0.03),
                slots=((-0.12, 0.0), (0.0, 0.0), (0.12, 0.0)),
            ),
        ),
        (
            "put each cube in a different tray, red on the left",
            "bỏ khối xanh lá vào khay trắng, sau đó đổi chỗ khối đỏ và khối xanh dương",
        ),
    ),
    3: Level(
        3,
        "four shapes: cube, cylinder, triangular prism, tetrahedron; wide tray and deep bin",
        (
            _cube("cube", "orange", ORANGE, 0.025),
            _cyl("cylinder", "orange", ORANGE, 0.02),
            ObjectSpec(
                "triangular prism",
                "prism",
                "orange",
                ORANGE,
                0.026,
                0.03,
                aliases=("lăng trụ tam giác", "khối tam giác"),
            ),
            ObjectSpec(
                "tetrahedron",
                "tetrahedron",
                "orange",
                ORANGE,
                tetra_radius(0.06),
                tetra_half_height(0.06),
                aliases=("tứ diện", "khối tứ diện đều"),
            ),
        ),
        (WIDE_TRAY, DEEP_BIN, TABLE_SPOT),
        ("put the round object in the deep bin and the others in the tray",),
    ),
    4: Level(
        4,
        "same shape, three sizes",
        (
            _cyl("small cylinder", "blue", BLUE, 0.014),
            _cyl("medium cylinder", "blue", BLUE, 0.019),
            _cyl("large cylinder", "blue", BLUE, 0.025),
        ),
        (WIDE_TRAY, DEEP_BIN, TABLE_SPOT),
        ("bỏ trụ nhỏ nhất vào khay sâu", "put the two biggest cylinders in the tray"),
    ),
    5: Level(
        5,
        "same shape, two colours x two sizes",
        (
            _cyl("small red cylinder", "red", RED, 0.015),
            _cyl("large red cylinder", "red", RED, 0.024),
            _cyl("small green cylinder", "green", GREEN, 0.015),
            _cyl("large green cylinder", "green", GREEN, 0.024),
        ),
        (WIDE_TRAY, DEEP_BIN, TABLE_SPOT),
        ("put the small ones in the deep bin", "gắp trụ đỏ lớn bỏ vào khay rộng"),
    ),
    6: Level(
        6,
        "look-alike packages: fresh milk vs soy milk carton, box 1 vs box 2",
        (
            ObjectSpec(
                "fresh milk carton",
                "box",
                "white and blue",
                (0.96, 0.97, 0.98),
                CARTON_HALF[0],
                CARTON_HALF[2],
                CARTON_HALF,
                ("hộp sữa tươi", "sữa tươi"),
                "drink carton printed FRESH MILK with cow spots",
                ("carton", "fresh_milk"),
                ("fresh",),
            ),
            ObjectSpec(
                "soy milk carton",
                "box",
                "beige and green",
                (0.93, 0.87, 0.7),
                CARTON_HALF[0],
                CARTON_HALF[2],
                CARTON_HALF,
                ("hộp sữa đậu nành", "sữa đậu nành"),
                "drink carton printed SOY MILK with soybeans",
                ("carton", "soy_milk"),
                ("soy",),
            ),
            ObjectSpec(
                "box 1",
                "box",
                "brown",
                (0.77, 0.59, 0.38),
                BOX_HALF[0],
                BOX_HALF[2],
                BOX_HALF,
                ("hộp số 1", "thùng số 1", "carton box 1"),
                "cardboard box printed with the number 1",
                ("carton", "box_1"),
                ("1",),
            ),
            ObjectSpec(
                "box 2",
                "box",
                "brown",
                (0.77, 0.59, 0.38),
                BOX_HALF[0],
                BOX_HALF[2],
                BOX_HALF,
                ("hộp số 2", "thùng số 2", "carton box 2"),
                "cardboard box printed with the number 2",
                ("carton", "box_2"),
                ("2",),
            ),
        ),
        (WIDE_TRAY, DEEP_BIN, TABLE_SPOT),
        (
            "bỏ hộp sữa tươi vào khay sâu và hộp sữa đậu nành vào khay nông",
            "put box 2 in the deep bin",
        ),
    ),
    7: Level(
        7,
        "look-alike balls and bottles: tennis ball vs volleyball, water vs juice bottle",
        (
            ObjectSpec(
                "tennis ball",
                "sphere",
                "yellow-green",
                (0.8, 0.88, 0.24),
                BALL_R,
                BALL_R,
                aliases=("bóng tennis", "quả bóng tennis"),
                kind="toy-size tennis ball with a white seam",
                visual=("sphere", "tennis"),
                keywords=("tennis",),
            ),
            ObjectSpec(
                "volleyball",
                "sphere",
                "white, yellow and blue",
                (0.95, 0.9, 0.6),
                BALL_R,
                BALL_R,
                aliases=("bóng chuyền", "quả bóng chuyền"),
                kind="toy-size volleyball with coloured panels",
                visual=("sphere", "volleyball"),
                keywords=("volley",),
            ),
            ObjectSpec(
                "water bottle",
                "bottle",
                "clear blue",
                (0.75, 0.88, 0.96),
                BOTTLE_R,
                BOTTLE_HALF_H,
                aliases=("chai nước suối", "chai nước lọc"),  # "chai nước" = any bottle
                kind="small water bottle with a blue cap",
                visual=("bottle", "water"),
                keywords=("water",),
            ),
            ObjectSpec(
                "juice bottle",
                "bottle",
                "orange",
                (0.96, 0.59, 0.12),
                BOTTLE_R,
                BOTTLE_HALF_H,
                aliases=("chai nước cam", "chai nước ép"),
                kind="small orange juice bottle with a red cap",
                visual=("bottle", "juice"),
                keywords=("juice", "orange"),
            ),
        ),
        (WIDE_TRAY, DEEP_BIN, TABLE_SPOT),
        (
            "bỏ chai nước cam vào khay sâu, bóng tennis vào khay nông",
            "swap the two bottles' places after putting both in the tray",
        ),
    ),
}


# Realistic-ish masses (kg) for the household objects; everything else keeps ObjectSpec.mass.
OBJECT_MASS = {
    "fresh milk carton": 0.10,
    "soy milk carton": 0.10,
    "water bottle": 0.12,
    "juice bottle": 0.12,
    "tennis ball": 0.04,
    "volleyball": 0.04,
}
LEVELS = {
    n: replace(
        lv, objects=tuple(replace(o, mass=OBJECT_MASS.get(o.name, o.mass)) for o in lv.objects)
    )
    for n, lv in LEVELS.items()
}


def jittered_spawns(
    level: Level, rng: random.Random, jitter: float = SPAWN_JITTER
) -> dict[str, tuple[float, float, float]]:
    """Spawn (x, y, yaw) per object: slot position plus a small random offset (data collection)."""
    out = {}
    for name, (x, y) in level.spawn_xy().items():
        out[name] = (
            x + rng.uniform(-jitter, jitter),
            y + rng.uniform(-jitter, jitter),
            rng.uniform(-math.pi, math.pi),
        )
    return out


def floor_top(container: ContainerSpec, table_top: float) -> float:
    return table_top + (0.0 if container.kind in ("pad", "table") else FLOOR_T)


def grasp_height(obj: ObjectSpec, container: ContainerSpec | None) -> float:
    """Grasp point height above the container floor (or table) for a top-down grasp.

    Inside a deep container the fingertips may reach FINGER_DEPTH below the rim (the bin is
    wide enough around each slot for the open fingers), so the grasp point is raised to at
    least rim - FINGER_DEPTH and kept GRASP_TOP_MARGIN below the object top.
    """
    center = obj.half_height
    if container is None or container.wall_height < DEEP_WALL:
        return center
    lowest = container.wall_height - FINGER_DEPTH
    return min(2 * obj.half_height - GRASP_TOP_MARGIN, max(center, lowest))


def grasp_fraction(obj: ObjectSpec, container: ContainerSpec | None) -> float:
    """Grasp height above the object centre, in units of half_height (0 = centre)."""
    return max(0.0, (grasp_height(obj, container) - obj.half_height) / obj.half_height)


def graspable_in(
    obj: ObjectSpec, container: ContainerSpec, vertical_half: float | None = None
) -> bool:
    """True when a top-down grasp can reach the object inside this container.

    ``vertical_half``: half of the object's current vertical size (lying objects are lower).
    """
    if container.wall_height < DEEP_WALL:
        return True
    if obj.shape in ("sphere", "tetrahedron"):  # tapered tops give no grip high up
        return False
    half = obj.half_height if vertical_half is None else vertical_half
    return 2 * half - GRASP_TOP_MARGIN >= container.wall_height - FINGER_DEPTH


def release_gap(
    container: ContainerSpec,
    obj: ObjectSpec,
    grasp_frac: float = 0.0,
    grasp_above_bottom: float | None = None,
) -> float:
    """Height of the object's bottom above the floor at release.

    As low as possible (MIN_DROP) so tall objects do not tip when dropped, but high enough that
    the fingertips (at the grasp height) stay above the rim of the container.
    ``grasp_above_bottom`` overrides the upright formula (lying objects, tetrahedra).
    """
    rim = container.wall_height + FINGER_CLEARANCE
    fingertips_above_bottom = (
        obj.half_height * (1.0 + grasp_frac) if grasp_above_bottom is None else grasp_above_bottom
    )
    return max(MIN_DROP, rim - fingertips_above_bottom)


def release_z(
    container: ContainerSpec, obj: ObjectSpec, table_top: float, grasp_frac: float = 0.0
) -> float:
    """place_actor target z: it stops PLACE_DIS above this, so the bottom ends at release_gap."""
    return floor_top(container, table_top) + release_gap(container, obj, grasp_frac) - PLACE_DIS


def release_grasp_fraction(obj: ObjectSpec, container: ContainerSpec) -> float:
    """Grasp a bit higher on straight tall objects so they can be released lower in a deep bin.

    The fingertips must stay above the rim; grasping higher lets the object's bottom go down
    to MIN_DROP above the floor. Spheres and tapered shapes keep the centre grasp.
    """
    if obj.shape not in ("cube", "cylinder", "box", "bottle"):
        return 0.0
    need = (container.wall_height + FINGER_CLEARANCE - MIN_DROP) / obj.half_height - 1.0
    return min(PLACE_FRAC_MAX, max(0.0, need))


def quat_matrix(q) -> list[list[float]]:
    """Rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def vertical_half_extent(obj: ObjectSpec, q) -> float:
    """Half of the object's current vertical size (upright, tilted or lying on its side)."""
    row = quat_matrix(q)[2]  # world-z components of the object's local axes
    if obj.shape == "sphere":
        return obj.radius
    if obj.shape == "box" and obj.half_extents:
        hx, hy, hz = obj.half_extents
        return abs(row[0]) * hx + abs(row[1]) * hy + abs(row[2]) * hz
    if obj.shape in ("prism", "tetrahedron"):
        verts = (prism_vertices if obj.shape == "prism" else tetra_vertices)(
            obj.radius, obj.half_height
        )
        heights = [row[0] * x + row[1] * y + row[2] * z for x, y, z in verts]
        return (max(heights) - min(heights)) / 2
    c = min(1.0, abs(row[2]))
    return c * obj.half_height + math.sqrt(max(0.0, 1.0 - c * c)) * obj.radius


def descend_dz(
    obj: ObjectSpec,
    center_z: float,
    q,
    container: ContainerSpec,
    table_top: float,
    grasp_frac: float = 0.0,
    grasp_above_bottom: float | None = None,
) -> float:
    """Vertical move (<= 0) that brings the held object's bottom to the release height."""
    bottom = center_z - vertical_half_extent(obj, q)
    gap = release_gap(container, obj, grasp_frac, grasp_above_bottom)
    target = floor_top(container, table_top) + gap
    return min(0.0, target - bottom)


def safe_ee_z(table_top: float, clearance: float) -> float:
    """End-effector height at which the fingertips clear every rim and object top."""
    return table_top + clearance + GRIPPER_REACH


def transit_waypoints(start_xy, goal_xy, step: float = TRANSIT_STEP) -> list[tuple[float, float]]:
    """Horizontal displacements (dx, dy) of at most ``step`` metres from start to goal."""
    dx, dy = goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]
    n = max(1, math.ceil(math.hypot(dx, dy) / step))
    return [(dx / n, dy / n)] * n


def lift_height(level: Level) -> float:
    tallest = max((c.wall_height for c in level.containers), default=0.0)
    return max(MIN_LIFT, tallest + LIFT_ABOVE_RIM)


def on_table(xyz: list[float] | tuple[float, ...], table_top: float) -> bool:
    (x0, x1), (y0, y1) = TABLE_BOUNDS
    return (
        x0 <= xyz[0] <= x1 and y0 <= xyz[1] <= y1 and table_top - 0.02 <= xyz[2] <= table_top + 0.3
    )


def is_inside(
    container: ContainerSpec, xyz: list[float] | tuple[float, ...], table_top: float
) -> bool:
    dx = abs(xyz[0] - container.xy[0])
    dy = abs(xyz[1] - container.xy[1])
    hx, hy = container.inner_half
    resting = xyz[2] < table_top + container.wall_height + 0.1
    if container.outline == "round":
        return math.hypot(dx, dy) <= hx + INSIDE_MARGIN and resting
    return dx <= hx + INSIDE_MARGIN and dy <= hy + INSIDE_MARGIN and resting


def location_of(level: Level, xyz: list[float] | tuple[float, ...], table_top: float) -> str:
    """Container name, 'table' (free on the table) or 'off table' (fell off / unreachable)."""
    for container in level.containers:
        if container.kind != "table" and is_inside(container, xyz, table_top):
            return container.name
    return "table" if on_table(xyz, table_top) else "off table"


def in_destination(level: Level, dest: ContainerSpec, xyz, table_top: float) -> bool:
    if dest.kind == "table":
        return location_of(level, xyz, table_top) == "table"
    return is_inside(dest, xyz, table_top)


def occupants(
    level: Level, container: ContainerSpec, objects_xyz: dict[str, list[float]], table_top: float
) -> list[str]:
    return [n for n, xyz in objects_xyz.items() if in_destination(level, container, xyz, table_top)]


def free_room(
    level: Level, container: ContainerSpec, objects_xyz: dict[str, list[float]], table_top: float
) -> int:
    return max(0, container.capacity - len(occupants(level, container, objects_xyz, table_top)))


def transport_clearance(
    level: Level, objects_xyz: dict[str, list[float]], table_top: float, carried: str
) -> float:
    """Height above the table that the carried object's bottom must clear while moving."""
    tops = [c.wall_height + FLOOR_T for c in level.containers]
    for name, xyz in objects_xyz.items():
        if name != carried and on_table(xyz, table_top):
            tops.append(xyz[2] - table_top + level.object(name).half_height)
    return max(tops, default=0.0) + TRANSPORT_MARGIN


def lift_dz(obj: ObjectSpec, center_z: float, clearance: float, table_top: float) -> float:
    """Vertical lift so the object's bottom ends above the transport clearance."""
    bottom = center_z - obj.half_height
    return max(MIN_LIFT_DZ, table_top + clearance - bottom)


def tilt_deg(q: tuple[float, float, float, float] | list[float]) -> float:
    """Angle between the object's up axis and world up, from a (w, x, y, z) quaternion."""
    _w, x, y, _z = q
    up_z = 1 - 2 * (x * x + y * y)  # z-component of the rotated local z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def toppled(obj: ObjectSpec, q) -> bool:
    if obj.shape == "sphere":
        return False
    return tilt_deg(q) > TOPPLED_DEG


def footprint_radius(obj: ObjectSpec, tilted: bool = False) -> float:
    """Horizontal radius of the object's footprint (a fallen object lies on its side)."""
    if obj.shape == "box" and obj.half_extents:
        hx, hy = obj.half_extents[0], obj.half_extents[1]
        upright = math.hypot(hx, hy) if tilted else max(hx, hy)
    else:
        upright = obj.radius
    return max(upright, obj.half_height) if tilted and obj.shape != "sphere" else upright


def grasp_blocker(
    level: Level,
    obj_name: str,
    objects_xyz: dict[str, list[float]],
    tilts: dict[str, float] | None = None,
    keepout: float = GRASP_KEEPOUT,
) -> tuple[str, float] | None:
    """(neighbour, gap) when another object is too close for the open fingers, else None."""
    tilts = tilts or {}
    target = level.object(obj_name)
    tx, ty = objects_xyz[obj_name][:2]
    r_target = footprint_radius(target, tilts.get(obj_name, 0.0) > TOPPLED_DEG)
    worst = None
    for name, xyz in objects_xyz.items():
        if name == obj_name:
            continue
        other = level.object(name)
        r_other = footprint_radius(other, tilts.get(name, 0.0) > TOPPLED_DEG)
        gap = math.dist((tx, ty), xyz[:2]) - r_target - r_other
        if gap < keepout and (worst is None or gap < worst[1]):
            worst = (name, gap)
    return worst


def unexpected_moves(
    before: dict[str, list[float]],
    after: dict[str, list[float]],
    allowed: set[str],
    tol: float = MOVED_TOL,
) -> dict[str, float]:
    """Objects (not in ``allowed``) that moved more than ``tol`` metres."""
    moved = {}
    for name, start in before.items():
        if name in allowed or name not in after:
            continue
        distance = math.dist(start, after[name])
        if distance > tol:
            moved[name] = distance
    return moved


def choose_slot(
    container: ContainerSpec, others_xy: list[tuple[float, float]]
) -> tuple[float, float]:
    """Pick the free slot farthest from every other object (first slot on ties)."""
    best, best_gap = None, -1.0
    for ox, oy in container.slots:
        xy = (container.xy[0] + ox, container.xy[1] + oy)
        gap = min((math.dist(xy, other) for other in others_xy), default=math.inf)
        if gap > best_gap + 1e-9:
            best, best_gap = xy, gap
    return best


def align_correction(
    target_xy: tuple[float, float], actual_xy: tuple[float, float]
) -> tuple[float, float] | None:
    """World-frame (dx, dy) that moves a held object onto its target, or None if not needed/unsafe."""
    dx, dy = target_xy[0] - actual_xy[0], target_xy[1] - actual_xy[1]
    error = math.hypot(dx, dy)
    if error <= ALIGN_TOL or error > ALIGN_MAX:
        return None
    return dx, dy


def container_boxes(
    container: ContainerSpec,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Local (center, half_size) boxes of a tray/bin: floor plus walls, origin on the table.

    A round container's wall is approximated by ROUND_SEGMENTS boxes on its circumference
    (axis-aligned boxes, so the planner sees a slightly larger, conservative ring).
    """
    hx, hy = container.inner_half
    h, t = container.wall_height, WALL_T
    if container.outline == "round":
        r = hx + t / 2
        seg = math.pi * r / ROUND_SEGMENTS + t / 2
        boxes = [((0.0, 0.0, FLOOR_T / 2), (hx + t, hy + t, FLOOR_T / 2))]
        for i in range(ROUND_SEGMENTS):
            a = 2 * math.pi * i / ROUND_SEGMENTS
            cx, cy = r * math.cos(a), r * math.sin(a)
            ex = abs(math.cos(a)) * t / 2 + abs(math.sin(a)) * seg
            ey = abs(math.sin(a)) * t / 2 + abs(math.cos(a)) * seg
            boxes.append(((cx, cy, FLOOR_T + h / 2), (ex, ey, h / 2)))
        return boxes
    boxes = [((0.0, 0.0, FLOOR_T / 2), (hx + t, hy + t, FLOOR_T / 2))]
    zc = FLOOR_T + h / 2
    for sx in (-1, 1):
        boxes.append(((sx * (hx + t / 2), 0.0, zc), (t / 2, hy + t, h / 2)))
    for sy in (-1, 1):
        boxes.append(((0.0, sy * (hy + t / 2), zc), (hx, t / 2, h / 2)))
    return boxes


def prism_vertices(radius: float, half_height: float) -> list[tuple[float, float, float]]:
    """Equilateral triangle (one corner pointing +y, away from the robot) extruded along z."""
    corners = [
        (radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)))
        for a in (90, 210, 330)
    ]
    return [(x, y, -half_height) for x, y in corners] + [(x, y, half_height) for x, y in corners]


def tetra_vertices(radius: float, half_height: float) -> list[tuple[float, float, float]]:
    """Regular tetrahedron standing on a face; origin at mid-height on the vertical axis."""
    base = [
        (radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)), -half_height)
        for a in (90, 210, 330)
    ]
    return [*base, (0.0, 0.0, half_height)]


def convex_faces(
    vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """Orient each triangle so its normal points away from the vertex centroid (0-based)."""
    n = len(vertices)
    cx, cy, cz = (sum(v[i] for v in vertices) / n for i in range(3))
    out = []
    for a, b, c in faces:
        ax, ay, az = vertices[a]
        u = [vertices[b][i] - vertices[a][i] for i in range(3)]
        w = [vertices[c][i] - vertices[a][i] for i in range(3)]
        normal = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
        outward = normal[0] * (ax - cx) + normal[1] * (ay - cy) + normal[2] * (az - cz)
        out.append((a, b, c) if outward > 0 else (a, c, b))
    return out


def mesh_obj_text(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    material: str = "",
) -> str:
    lines = ["# convex mesh generated by twin_scene.py"]
    if material:
        lines += [f"mtllib {material}.mtl", f"usemtl {material}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in convex_faces(vertices, faces)]
    return "\n".join(lines) + "\n"


PRISM_FACES = [
    (0, 1, 2),
    (3, 4, 5),
    (0, 1, 4),
    (0, 4, 3),
    (1, 2, 5),
    (1, 5, 4),
    (2, 0, 3),
    (2, 3, 5),
]
TETRA_FACES = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)]


def prism_obj_text(radius: float, half_height: float, material: str = "") -> str:
    return mesh_obj_text(prism_vertices(radius, half_height), PRISM_FACES, material)


def contact_poses(frac: float = 0.0) -> list[list[list[float]]]:
    """Four top-down grasp frames at height ``frac`` (units of half_height) above the centre."""
    return [
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, frac], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, frac], [0, 0, 0, 1]],
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, frac], [0, 0, 0, 1]],
        [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, frac], [0, 0, 0, 1]],
    ]


def grasp_data(obj: ObjectSpec) -> dict:
    """RoboTwin Actor config: four top-down grasps at the centre, functional point at the bottom."""
    if obj.shape == "box" and obj.half_extents:
        scale = list(obj.half_extents)
    else:
        scale = [obj.radius, obj.radius, obj.half_height]
    return {
        "center": [0, 0, 0],
        "extents": scale,
        "scale": scale,
        "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
        "contact_points_pose": contact_poses(0.0),
        "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "functional_matrix": [
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, -1], [0.0, 0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, 1], [0.0, 0.0, 0.0, 1.0]],
        ],
        "contact_points_description": [],
        "contact_points_group": [[0, 1, 2, 3]],
        "contact_points_mask": [True],
        "target_point_description": ["The center point on the bottom of the object."],
    }


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


# ----------------------------------------------------------------- SAPIEN builders (GPU env)


def _physical_material(scene):
    import sapien

    try:
        return sapien.physx.PhysxMaterial(1.0, 1.0, 0.0)
    except Exception:  # fall back to the scene default
        return scene.default_physical_material


def write_mesh_files(obj: ObjectSpec, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{obj.shape}_{round(obj.radius * 1000)}_{round(obj.half_height * 1000)}"
    r, g, b = obj.rgb
    (cache_dir / f"{stem}.mtl").write_text(f"newmtl {stem}\nKd {r:.3f} {g:.3f} {b:.3f}\n")
    if obj.shape == "prism":
        text = mesh_obj_text(prism_vertices(obj.radius, obj.half_height), PRISM_FACES, stem)
    else:
        text = mesh_obj_text(tetra_vertices(obj.radius, obj.half_height), TETRA_FACES, stem)
    mesh = cache_dir / f"{stem}.obj"
    mesh.write_text(text)
    return mesh


def bottle_collision(radius: float, half_height: float) -> list[tuple[float, float, float]]:
    """(centre z, radius, half length) of the body, shoulder and neck+cap cylinders."""
    H = 2 * half_height
    body = (-half_height + 0.36 * H, radius, 0.36 * H)
    shoulder = (-half_height + 0.76 * H, radius * 0.75, 0.04 * H)
    neck = (-half_height + 0.90 * H, radius * 0.62, 0.10 * H)
    return [body, shoulder, neck]


def build_object(
    scene, obj: ObjectSpec, xyz: tuple[float, float, float], cache_dir: Path, yaw: float = 0.0
):
    """Create a dynamic, graspable object and wrap it as a RoboTwin Actor."""
    import sapien

    try:
        from envs.utils import Actor
    except ImportError:
        from envs.utils.actor_utils import Actor

    builder = scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    physical = _physical_material(scene)
    look = sapien.render.RenderMaterial(base_color=[*obj.rgb, 1])
    r, h = obj.radius, obj.half_height
    up = sapien.Pose(q=list(UPRIGHT_Q))
    textured = None
    if obj.visual is not None:
        from twin_assets import write_asset

        kind, design = obj.visual
        dims = {"carton": obj.half_extents, "sphere": (r,), "bottle": (r, h)}[kind]
        textured = write_asset(kind, design, tuple(dims), cache_dir)
    if obj.shape in ("cube", "box"):
        half = list(obj.half_extents) if obj.half_extents else [r, r, h]
        builder.add_box_collision(half_size=half, material=physical)
        if textured is None:
            builder.add_box_visual(half_size=half, material=look)
    elif obj.shape == "cylinder":
        builder.add_cylinder_collision(pose=up, radius=r, half_length=h, material=physical)
        builder.add_cylinder_visual(pose=up, radius=r, half_length=h, material=look)
    elif obj.shape == "sphere":
        builder.add_sphere_collision(radius=r, material=physical)
        if textured is None:
            builder.add_sphere_visual(radius=r, material=look)
    elif obj.shape == "bottle":
        for center_z, radius, half_len in bottle_collision(r, h):
            pose = sapien.Pose([0.0, 0.0, center_z], list(UPRIGHT_Q))
            builder.add_cylinder_collision(
                pose=pose, radius=radius, half_length=half_len, material=physical
            )
    else:
        mesh = write_mesh_files(obj, cache_dir)
        add_convex = getattr(builder, "add_convex_collision_from_file", None)
        if add_convex is None:
            builder.add_multiple_convex_collisions_from_file(filename=str(mesh), material=physical)
        else:
            add_convex(filename=str(mesh), material=physical)
        try:
            builder.add_visual_from_file(filename=str(mesh), material=look)
        except TypeError:
            builder.add_visual_from_file(filename=str(mesh))
    if textured is not None:
        builder.add_visual_from_file(filename=str(textured))
    builder.set_initial_pose(sapien.Pose(list(xyz), list(yaw_quat(yaw))))
    entity = builder.build(name=obj.name.replace(" ", "_"))
    body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if body is not None and obj.shape == "sphere":  # rolling resistance (felt / rubber surface)
        for name, value in (("angular_damping", 3.0), ("linear_damping", 0.5)):
            setter = getattr(body, f"set_{name}", None)
            if setter is not None:
                setter(value)
    return Actor(entity, grasp_data(obj), mass=obj.mass)


def build_container(scene, container: ContainerSpec, table_top: float):
    """Static tray/bin with real collision walls, or a visual-only staging pad (table: nothing)."""
    if container.kind == "table":  # free spots on the table itself: no geometry
        return None
    import sapien

    look = sapien.render.RenderMaterial(base_color=[*container.rgb, 1])
    name = container.name.replace(" ", "_")
    if container.kind == "pad":
        entity = sapien.Entity()
        entity.set_name(name)
        body = sapien.render.RenderBodyComponent()
        body.attach(sapien.render.RenderShapeBox([*container.inner_half, 0.001], look))
        entity.add_component(body)
        entity.set_pose(sapien.Pose([*container.xy, table_top + 0.001]))
        scene.add_entity(entity)
        return entity
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    physical = scene.default_physical_material
    if container.outline == "round":
        return _build_round_container(builder, container, table_top, look, physical, name)
    for center, half in container_boxes(container):
        pose = sapien.Pose(list(center))
        builder.add_box_collision(pose=pose, half_size=list(half), material=physical)
        builder.add_box_visual(pose=pose, half_size=list(half), material=look)
    builder.set_initial_pose(sapien.Pose([*container.xy, table_top]))
    return builder.build(name=name)


def _build_round_container(
    builder, container: ContainerSpec, table_top: float, look, physical, name
):
    """Cylindrical floor plus ROUND_SEGMENTS wall pieces rotated tangent to the circle."""
    import sapien

    radius, h, t = container.inner_half[0], container.wall_height, WALL_T
    up = sapien.Pose([0.0, 0.0, FLOOR_T / 2], list(UPRIGHT_Q))
    builder.add_cylinder_collision(
        pose=up, radius=radius + t, half_length=FLOOR_T / 2, material=physical
    )
    builder.add_cylinder_visual(pose=up, radius=radius + t, half_length=FLOOR_T / 2, material=look)
    r = radius + t / 2
    half = [t / 2, math.pi * r / ROUND_SEGMENTS + t / 2, h / 2]
    for i in range(ROUND_SEGMENTS):
        a = 2 * math.pi * i / ROUND_SEGMENTS
        pose = sapien.Pose([r * math.cos(a), r * math.sin(a), FLOOR_T + h / 2], list(yaw_quat(a)))
        builder.add_box_collision(pose=pose, half_size=half, material=physical)
        builder.add_box_visual(pose=pose, half_size=half, material=look)
    builder.set_initial_pose(sapien.Pose([*container.xy, table_top]))
    return builder.build(name=name)
