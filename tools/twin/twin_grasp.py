"""Orientation-aware top-down grasp planning for the digital twin (pure numpy, unit-tested).

RoboTwin builds the gripper pose from a contact frame given in the object's local frame:
gripper x (approach) = -column 1, gripper y = -column 2, gripper z = column 0 (see
``Base_Task.get_grasp_pose``: contact @ [[0,0,1],[-1,0,0],[0,-1,0]]). The fingers close along
gripper y or z depending on the embodiment; ``finger_axis_from_vectors`` reads it from the
simulated finger links at setup.

For each object, in its CURRENT pose (upright, lying on its side, a tetrahedron resting on any
face), this module lists the horizontal finger directions that can close on it, measures the
free room the open fingers have towards every neighbour and every deep wall in that
direction, and returns the contact frames for the best direction. Only when no direction has
room (or the object cannot be grasped from the top at all) is the grasp refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import twin_scene as ts

GRIPPER_MAX_OPEN = 0.058  # widest object closed on reliably (5 cm boxes work; 6 cm slipped)
FINGER_ROOM = 0.022  # free gap needed beside the object along the finger axis (open finger)
FINGER_HALF_WIDTH = 0.012  # half the finger width, across the finger axis
UPRIGHT_COS = 0.85  # |cos| between an axis and world up to count as vertical
FLAT_COS = 0.30  # |cos| below which an axis counts as horizontal
TETRA_GRASP_HEIGHT = 0.45  # grasp height on a tetrahedron (30% put the fingertips on the table)
TILT_CHANGE_DEG = 15.0  # an untouched object whose tilt changes this much was hit


def quat_to_mat(q) -> np.ndarray:
    return np.array(ts.quat_matrix(q), float)


@dataclass
class GraspChoice:
    ok: bool
    reason: str = ""
    frames: list = field(default_factory=list)  # local 4x4 contact frames (translation / scale)
    finger_world: tuple[float, float] | None = None  # horizontal finger direction (world)
    width: float = 0.0  # object width between the fingers
    room: float = math.inf  # smallest free gap beside the fingers
    lying: bool = False  # object is not in its upright pose
    grasp_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.0)  # grasp point - centre
    vertical_half: float = 0.0  # half of the current vertical size
    directions: int = 0  # grasp directions offered to RoboTwin
    blocked_directions: int = 0  # directions removed because a neighbour/wall is in the way
    frame_tilts: list = field(default_factory=list)  # approach tilt (deg) of each frame
    approach_world: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def describe(self) -> str:
        if not self.ok:
            return self.reason
        room = (
            "nothing in the finger path"
            if math.isinf(self.room)
            else f"{self.room * 100:.1f} cm finger room"
        )
        pose = "lying, " if self.lying else ""
        tilted = sum(t > TOPDOWN_TOL_DEG for t in self.frame_tilts)
        dirs = f"{len(self.frames)} grasp frame(s)" + (f", {tilted} tilted" if tilted else "")
        if self.blocked_directions:
            dirs += f", {self.blocked_directions} blocked by a neighbour/wall"
        return f"{pose}fingers across {self.width * 100:.1f} cm, {room}; {dirs}"


def finger_axis_from_vectors(finger_delta_world, ee_quat) -> str | None:
    """'y' or 'z': the gripper-frame axis along which the fingers open (None if unclear)."""
    d = np.asarray(finger_delta_world, float)
    if np.linalg.norm(d) < 1e-4:
        return None
    local = quat_to_mat(ee_quat).T @ (d / np.linalg.norm(d))
    if abs(local[1]) > 0.8:
        return "y"
    if abs(local[2]) > 0.8:
        return "z"
    return None


def finger_axis_from_displacements(displacements_world, ee_quat) -> str | None:
    """'y'/'z' from how the gripper links moved between open and closed (None if unclear).

    Works for parallel and linkage grippers: the finger links move mostly along the opening
    axis; the component along the approach axis (gripper x) is ignored.
    """
    rot = quat_to_mat(ee_quat)
    sy = sz = 0.0
    for d in displacements_world:
        local = rot.T @ np.asarray(d, float)
        sy += abs(local[1])
        sz += abs(local[2])
    if sy + sz < 1e-4:
        return None
    if sy >= 2 * sz:
        return "y"
    if sz >= 2 * sy:
        return "z"
    return None


def contact_axis(points) -> tuple[float, float] | None:
    """Horizontal direction through finger contact points on both sides (None if one-sided)."""
    pts = np.asarray([p[:2] for p in points], float)
    if len(pts) < 2:
        return None
    centred = pts - pts.mean(axis=0)
    _u, svals, vt = np.linalg.svd(centred, full_matrices=False)
    if svals[0] < 1e-9:
        return None
    axis = vt[0]
    spread = centred @ axis
    if spread.max() - spread.min() < 0.01:  # all contacts on one side or one point
        return None
    return float(axis[0]), float(axis[1])


def axis_angle_deg(a, b) -> float:
    """Angle between two undirected horizontal axes (0..90 deg)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(min(1.0, c)))


def scale_of(obj: ts.ObjectSpec) -> np.ndarray:
    if obj.shape == "box" and obj.half_extents:
        return np.asarray(obj.half_extents, float)
    return np.array([obj.radius, obj.radius, obj.half_height], float)


def support(obj: ts.ObjectSpec, q, direction) -> float:
    """Extent of the object from its centre along a world direction (support function)."""
    g = np.asarray(direction, float)
    g = g / (np.linalg.norm(g) or 1.0)
    rot = quat_to_mat(q)
    local = rot.T @ g
    if obj.shape == "sphere":
        return obj.radius
    if obj.shape in ("box", "cube"):
        half = scale_of(obj) if obj.shape == "box" else np.array([obj.radius] * 3)
        return float(np.sum(np.abs(local) * half))
    if obj.shape in ("cylinder", "bottle"):
        c = min(1.0, abs(local[2]))
        return float(c * obj.half_height + math.sqrt(max(0.0, 1 - c * c)) * obj.radius)
    if obj.shape == "prism":
        verts = np.asarray(ts.prism_vertices(obj.radius, obj.half_height))
    else:
        verts = np.asarray(ts.tetra_vertices(obj.radius, obj.half_height))
    return float(np.max(verts @ local))


def _frame(up_local, finger_local, finger_axis: str, offset_local, scale) -> list:
    c1 = up_local / np.linalg.norm(up_local)
    f = finger_local - np.dot(finger_local, c1) * c1
    f = f / np.linalg.norm(f)
    if finger_axis == "z":
        c0 = f
        c2 = np.cross(c0, c1)
    else:  # fingers along gripper y = -column 2
        c2 = -f
        c0 = np.cross(c1, c2)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = c0, c1, c2
    m[:3, 3] = np.asarray(offset_local, float) / scale
    return m.round(9).tolist()


@dataclass
class _Candidate:
    approach: np.ndarray  # local unit vector from the object towards the gripper
    finger_local: np.ndarray
    width: float
    offset: np.ndarray  # grasp point - centre (local, metres)
    tilt: float = 0.0  # angle between the approach and world up (deg)
    secondary: bool = False  # extra yaw, used only when no primary direction is free


WRIST_EASY_DEG = 90.0  # wrist turns up to this are tried first; up to 180 deg are allowed
TOPDOWN_TOL_DEG = 25.0  # approaches within this of vertical count as "from above"
MAX_TILT_DEG = 60.0  # steepest approach offered when the planner knows the obstacles
TILT_STEPS_DEG = (20.0, 35.0, 45.0, 55.0)  # tilted variants of a free top-down grasp
# (Panthera-HT cannot lift with a vertical gripper: the console ranks these with the arm's
# kinematic model and takes the least tilted one that reaches the whole pick)


def _tilt(approach: np.ndarray, up: np.ndarray) -> float:
    c = float(np.dot(approach, up) / np.linalg.norm(approach))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _perp_dirs(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors perpendicular to ``axis`` (local frame)."""
    a = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _round(approach, radius, offset, up) -> list[_Candidate]:
    """Round cross-section: two perpendicular yaws first, diagonals only as a fallback."""
    u, v = _perp_dirs(approach)
    t = _tilt(approach, up)
    out = [_Candidate(approach, d, 2 * radius, offset, t) for d in (u, v)]
    for d in ((u + v) / math.sqrt(2), (u - v) / math.sqrt(2)):
        out.append(_Candidate(approach, d, 2 * radius, offset, t, secondary=True))
    return out


def _candidates(obj: ts.ObjectSpec, rot: np.ndarray, upright_frac: float):
    """(up_local, lying, [candidates]) for every approach direction, or a refusal string."""
    up = rot.T @ np.array([0.0, 0.0, 1.0])
    ez = np.eye(3)[2]
    cos_max = math.cos(math.radians(MAX_TILT_DEG))
    if obj.shape == "sphere":
        return up, False, _round(up, obj.radius, np.zeros(3), up)
    if obj.shape in ("box", "cube"):
        half = scale_of(obj) if obj.shape == "box" else np.array([obj.radius] * 3)
        vertical = max(range(3), key=lambda i: abs(up[i]))
        lying = vertical != 2 or abs(up[2]) < UPRIGHT_COS
        cands = []
        for i in range(3):
            n = np.eye(3)[i] * (1.0 if up[i] >= 0 else -1.0)  # the face pointing most upwards
            if float(n @ up) < cos_max:
                continue
            offset = np.zeros(3)
            if i == 2 and not lying and upright_frac:
                offset = n * upright_frac * half[2]
            for j in range(3):
                if j != i:
                    cands.append(_Candidate(n, np.eye(3)[j], 2 * half[j], offset, _tilt(n, up)))
        if not cands:
            return "no face points up enough to grasp it"
        return up, lying, cands
    if obj.shape in ("cylinder", "bottle"):
        ca = float(up[2])
        lying = abs(ca) < UPRIGHT_COS
        cands = []
        if ca >= cos_max or (obj.shape == "cylinder" and -ca >= cos_max):
            n = ez if ca >= 0 else -ez  # end face up (a bottle is never held upside down)
            offset = n * upright_frac * obj.half_height if not lying else np.zeros(3)
            cands += _round(n, obj.radius, offset, up)
        side = up - ca * ez
        if np.linalg.norm(side) > 1e-6:
            side /= np.linalg.norm(side)
            if float(side @ up) >= cos_max:
                offset = ez * (-0.28 * obj.half_height) if obj.shape == "bottle" else np.zeros(3)
                across = np.cross(ez, side)
                cands.append(_Candidate(side, across, 2 * obj.radius, offset, _tilt(side, up)))
        if not cands:
            return "no side of it can be reached from above"
        return up, lying, cands
    if obj.shape == "prism":
        cands = []
        if up[2] >= cos_max:  # standing: jaws perpendicular to a side face
            for a in np.radians((30, 150, 270)):
                f = np.array([math.cos(a), math.sin(a), 0.0])
                cands.append(_Candidate(ez, f, 1.5 * obj.radius, np.zeros(3), _tilt(ez, up)))
        side = up - float(up[2]) * ez  # lying: come from above, perpendicular to the axis,
        if np.linalg.norm(side) > 1e-6:  # and close the jaws on the two triangular ends
            side /= np.linalg.norm(side)
            if float(side @ up) >= cos_max:
                cands.append(
                    _Candidate(side, ez, 2 * obj.half_height, np.zeros(3), _tilt(side, up))
                )
        if not cands:
            return "no face of it points up enough"
        return up, up[2] < UPRIGHT_COS, cands
    if obj.shape == "tetrahedron":
        verts = np.asarray(ts.tetra_vertices(obj.radius, obj.half_height))
        heights = verts @ up
        apex = int(np.argmax(heights))
        base = [i for i in range(4) if i != apex]
        if heights[apex] - np.max(heights[base]) < 0.8 * (2 * obj.half_height):
            return "it is not resting on a face"
        base_centre = verts[base].mean(axis=0)
        axis = verts[apex] - base_centre
        height = float(np.linalg.norm(axis))
        axis /= height
        point = base_centre + axis * (TETRA_GRASP_HEIGHT * height)
        shrink = 1.0 - TETRA_GRASP_HEIGHT
        cands = []
        for i in base:  # jaw axis from the base vertex through the opposite side face
            radial = verts[i] - base_centre
            radial -= np.dot(radial, axis) * axis
            cands.append(
                _Candidate(axis, radial, 1.5 * obj.radius * shrink, point, _tilt(axis, up))
            )
        return up, False, cands
    return f"no grasp model for shape {obj.shape!r}"


def _tilted_variants(cand: _Candidate, up: np.ndarray) -> list[_Candidate]:
    """Approaches tilted about the finger axis (the fingers keep closing on the same faces)."""
    side = np.cross(cand.finger_local, cand.approach)
    if np.linalg.norm(side) < 1e-6:
        return []
    side /= np.linalg.norm(side)
    out = []
    for t in TILT_STEPS_DEG:
        for sign in (1.0, -1.0):
            a = math.cos(math.radians(t)) * cand.approach + sign * math.sin(math.radians(t)) * side
            tilt = _tilt(a, up)
            if tilt <= MAX_TILT_DEG:
                out.append(_Candidate(a, cand.finger_local, cand.width, cand.offset, tilt))
    return out


def _wall_room(container: ts.ContainerSpec | None, centre_xy, finger_w, half_width) -> float:
    """Free gap between the open finger and the walls of a deep container along the axis."""
    if container is None or container.wall_height < ts.DEEP_WALL:
        return math.inf
    hx, hy = container.inner_half
    dx, dy = centre_xy[0] - container.xy[0], centre_xy[1] - container.xy[1]
    room = math.inf
    if container.outline == "round":
        norm = math.hypot(finger_w[0], finger_w[1]) or 1.0
        for sign in (1.0, -1.0):
            fx, fy = sign * finger_w[0] / norm, sign * finger_w[1] / norm
            along = dx * fx + dy * fy
            disc = along * along - (dx * dx + dy * dy) + hx * hx
            t = -along + math.sqrt(max(disc, 0.0))
            room = min(room, t - half_width)
        return room
    for sign in (1.0, -1.0):
        fx, fy = sign * finger_w[0], sign * finger_w[1]
        t = math.inf
        if abs(fx) > 1e-9:
            t = min(t, ((hx if fx > 0 else -hx) - dx) / fx)
        if abs(fy) > 1e-9:
            t = min(t, ((hy if fy > 0 else -hy) - dy) / fy)
        room = min(room, t - half_width)
    return room


def _finger_room(cand, rot, centre, neighbours, finger_axis, container) -> float:
    fw = rot @ cand.finger_local
    fw = np.array([fw[0], fw[1]])
    if np.linalg.norm(fw) < 1e-6:
        return math.inf  # fingers close vertically (side grasp of a lying object's ends)
    fw = fw / np.linalg.norm(fw)
    axes = [fw] if finger_axis else [fw, np.array([-fw[1], fw[0]])]
    room = math.inf
    half = cand.width / 2
    for axis in axes:
        perp = np.array([-axis[1], axis[0]])
        for n_obj, n_xyz, n_q in neighbours:
            d = np.asarray(n_xyz[:2], float) - centre[:2]
            if np.linalg.norm(d) > 0.25:
                continue
            along = abs(float(d @ axis))
            lateral = abs(float(d @ perp))
            r_along = support(n_obj, n_q, [*(-np.sign(d @ axis) * axis), 0.0])
            r_lat = support(n_obj, n_q, [*perp, 0.0])
            if lateral - r_lat >= FINGER_HALF_WIDTH:
                continue  # the neighbour is beside the finger, not in front of it
            room = min(room, along - r_along - half)
        room = min(room, _wall_room(container, centre[:2], axis, half))
    return room


def plan_grasp(
    obj: ts.ObjectSpec,
    xyz,
    q,
    neighbours: list[tuple[ts.ObjectSpec, list[float], list[float]]],
    finger_axis: str | None,
    container: ts.ContainerSpec | None = None,
    upright_frac: float = 0.0,
    allow_tilt: bool = False,
    ref_quat=None,
) -> GraspChoice:
    """Ordered grasp candidates for ``obj`` at (xyz, q), or a refusal with the reason.

    Candidates are top-down first (every free yaw), then, only with ``allow_tilt`` (the motion
    planner knows the obstacles), approaches tilted about the finger axis and grasps of leaning
    objects through their most upward face, up to MAX_TILT_DEG from vertical. Directions whose
    open fingers would hit a neighbour or a deep wall are removed. The executor tries the
    frames in this order and takes the first one the planner can reach.
    """
    rot = quat_to_mat(q)
    found = _candidates(obj, rot, 0.0 if obj.shape in ("tetrahedron", "sphere") else upright_frac)
    if isinstance(found, str):
        return GraspChoice(False, f"{obj.name} cannot be grasped from above: {found}")
    up, lying, cands = found
    centre = np.asarray(xyz, float)
    fits = [c for c in cands if c.width <= GRIPPER_MAX_OPEN]
    cands = fits if allow_tilt else [c for c in fits if c.tilt <= TOPDOWN_TOL_DEG]
    if not cands:
        why = (
            "it is resting on an edge / leaning; grasping it needs a tilted approach, which "
            "needs the obstacle-aware planner"
            if fits
            else "it is too wide for the gripper"
        )
        return GraspChoice(False, f"{obj.name} cannot be grasped from above: {why}")
    rated = [(c, _finger_room(c, rot, centre, neighbours, finger_axis, container)) for c in cands]
    free = [(c, r) for c, r in rated if r >= FINGER_ROOM and not c.secondary]
    if not any(c.tilt <= TOPDOWN_TOL_DEG for c, _ in free):
        free += [(c, r) for c, r in rated if r >= FINGER_ROOM and c.secondary]
    if allow_tilt:
        top = [(c, r) for c, r in free if c.tilt <= TOPDOWN_TOL_DEG]
        free += [(v, r) for c, r in top for v in _tilted_variants(c, up)]
    if not free:
        best_room = max(r for _, r in rated)
        blockers = _closest(obj, centre, neighbours)
        what = f" (closest: {blockers})" if blockers else ""
        return GraspChoice(
            False,
            f"{obj.name} has no room for the open fingers in any direction "
            f"(best {max(best_room, 0.0) * 100:.1f} cm < {FINGER_ROOM * 100:.1f} cm){what}",
            room=best_room,
        )
    free.sort(key=lambda cr: (round(cr[0].tilt / 10.0), cr[0].secondary))  # vertical first
    scale = scale_of(obj)
    axes = [finger_axis] if finger_axis else ["y", "z"]
    entries: list = []  # (sort key, frame, tilt)
    seen: list = []
    for order, (cand, _) in enumerate(free):
        for axis in axes:
            for sign in (1.0, -1.0):
                frame = _frame(cand.approach, sign * cand.finger_local, axis, cand.offset, scale)
                if frame in seen:  # 'y' along f and 'z' along f give the same yaws
                    continue
                seen.append(frame)
                turn = abs(wrist_turn_deg(frame, q, ref_quat)) if ref_quat is not None else 0.0
                # Both yaws of a symmetric pair are kept (wrist range 180 deg each way covers
                # every finger orientation); within one tilt group the yaw that turns the wrist
                # at most 90 deg comes first, the opposite one (90-180 deg) right after.
                hard = turn > WRIST_EASY_DEG
                key = (round(cand.tilt / 10.0), hard, cand.secondary, round(turn), order)
                entries.append((key, frame, cand.tilt))
    entries.sort(key=lambda e: e[0])
    frames = [f for _, f, _ in entries]
    tilts = [t for _, _, t in entries]
    first = free[0][0]
    fw = rot @ first.finger_local
    norm = math.hypot(fw[0], fw[1]) or 1.0
    approach_world = rot @ first.approach
    return GraspChoice(
        True,
        frames=frames,
        finger_world=(float(fw[0] / norm), float(fw[1] / norm)),
        width=first.width,
        room=max(r for _, r in free),
        lying=bool(lying),
        grasp_offset_world=tuple(float(v) for v in rot @ first.offset),
        vertical_half=support(obj, q, [0, 0, 1]),
        directions=len({(round(c.tilt), tuple(np.round(c.finger_local, 3))) for c, _ in free}),
        blocked_directions=len(rated) - len([c for c, r in rated if r >= FINGER_ROOM]),
        frame_tilts=tilts,
        approach_world=tuple(float(v) for v in approach_world),
    )


GRASP_TO_GRIPPER = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def gripper_rotation(frame_local, actor_quat) -> np.ndarray:
    """World rotation of the gripper (RoboTwin end-pose frame) for a contact frame."""
    return quat_to_mat(actor_quat) @ np.asarray(frame_local, float)[:3, :3] @ GRASP_TO_GRIPPER


def wrist_turn_deg(frame_local, actor_quat, ref_quat) -> float:
    """Signed rotation about the vertical between the current gripper and this grasp (deg)."""
    g = gripper_rotation(frame_local, actor_quat)
    r = quat_to_mat(ref_quat)
    # compare the gripper axis that lies most in the horizontal plane in the reference pose
    i = 1 if math.hypot(*r[:2, 1]) >= math.hypot(*r[:2, 2]) else 2
    a, b = r[:2, i], g[:2, i]
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
        return 0.0
    ang = math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
    return math.degrees(ang)


def _closest(obj, centre, neighbours) -> str:
    if not neighbours:
        return ""
    name, _xyz, _ = min(
        ((n.name, x, q) for n, x, q in neighbours),
        key=lambda item: math.dist(item[1][:2], centre[:2]),
    )
    return name


def finger_world_axis(frame_local, actor_quat, finger_axis: str) -> np.ndarray:
    """World direction of the fingers for a contact frame (for logs and tests)."""
    m = np.asarray(frame_local, float)[:3, :3]
    col = m[:, 0] if finger_axis == "z" else -m[:, 2]
    return quat_to_mat(actor_quat) @ col


def unexpected_tilts(
    before: dict[str, list[float]],
    after: dict[str, list[float]],
    allowed: set[str],
    tol_deg: float = TILT_CHANGE_DEG,
) -> dict[str, float]:
    """Objects (not in ``allowed``) whose orientation changed by more than ``tol_deg``."""
    out = {}
    for name, q0 in before.items():
        if name in allowed or name not in after:
            continue
        a, b = np.asarray(q0, float), np.asarray(after[name], float)
        dot = min(1.0, abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
        angle = math.degrees(2 * math.acos(dot))
        if angle > tol_deg:
            out[name] = angle
    return out


def hold_vector(ee_quat, motion: str) -> list[float] | None:
    """CuRobo partial-pose hold for a straight move: orientation + the right linear axes.

    ``motion`` 'horizontal' holds the gripper axis that points along world z (constant
    height); 'vertical' holds the two axes that span the horizontal plane (straight up/down).
    None when no gripper axis is close enough to vertical.
    """
    rot = quat_to_mat(ee_quat)
    iz = int(np.argmax(np.abs(rot[2, :])))
    if abs(rot[2, iz]) < 0.999:  # within ~2.5 deg: a 20 deg tilt must not count as vertical
        return None
    lin = [0.0, 0.0, 0.0]
    if motion == "horizontal":
        lin[iz] = 1.0
    else:
        lin = [1.0, 1.0, 1.0]
        lin[iz] = 0.0
    return [1.0, 1.0, 1.0, *lin]


def mat_to_quat(m) -> list[float]:
    """Rotation matrix -> quaternion [w, x, y, z] (w >= 0)."""
    m = np.asarray(m, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        k = 2.0 * math.sqrt(tr + 1.0)
        q = [0.25 * k, (m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        k = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / k, 0.25 * k, (m[0, 1] + m[1, 0]) / k, (m[0, 2] + m[2, 0]) / k]
    elif m[1, 1] > m[2, 2]:
        k = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / k, (m[0, 1] + m[1, 0]) / k, 0.25 * k, (m[1, 2] + m[2, 1]) / k]
    else:
        k = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / k, (m[0, 2] + m[2, 0]) / k, (m[1, 2] + m[2, 1]) / k, 0.25 * k]
    if q[0] < 0:
        q = [-v for v in q]
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / n for v in q]


EE_TO_TIP = 0.12  # RoboTwin end-effector pose sits this far behind the fingertip contact


def grasp_poses(actor_p, actor_q, frame_local, scale, pre_dis: float):
    """(pre-grasp pose, grasp pose) [x, y, z, qw, qx, qy, qz] for a contact frame.

    Same geometry as RoboTwin's ``get_grasp_pose`` but WITHOUT its ``choose_best_pose`` step,
    which re-rotates the pose about the object within the embodiment's ``rotate_lim`` and
    keeps the last reachable rotation (a tilted gripper even for a top-down frame). The twin
    moves to these exact poses itself, so a top-down frame gives a truly vertical gripper.
    """
    f = np.array(frame_local, float)
    f[:3, 3] = f[:3, 3] * np.asarray(scale, float)
    world = np.eye(4)
    world[:3, :3] = quat_to_mat(actor_q)
    world[:3, 3] = np.asarray(actor_p, float)
    contact = world @ f
    g = contact[:3, :3] @ GRASP_TO_GRIPPER
    x = g[:, 0]
    grasp_p = contact[:3, 3] - EE_TO_TIP * x
    pre_p = grasp_p - pre_dis * x
    q = mat_to_quat(g)
    return [*map(float, pre_p), *q], [*map(float, grasp_p), *q]


def approach_tilt_deg(pose) -> float:
    """Angle between the gripper approach (x axis) of a pose and straight down."""
    x = quat_to_mat(pose[3:7])[:, 0]
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(x[2])))))


def yaw_to_lean(quat, lean_xy, min_lean: float = 0.05) -> tuple[float, list[float]]:
    """Rotation about the world vertical (deg) that turns the horizontal part of the gripper
    approach towards ``lean_xy`` (e.g. away from the robot base), and the rotated quaternion.

    Turning about the vertical keeps a held object upright (it only spins in place), so a
    tilted grasp can be re-aimed for the place location where the arm reaches best.
    Returns (0, quat) for a (nearly) vertical approach.
    """
    r = quat_to_mat(quat)
    h = r[:2, 0]
    u = np.asarray(lean_xy, float)
    if np.linalg.norm(h) < min_lean or np.linalg.norm(u) < 1e-9:
        return 0.0, [float(v) for v in quat]
    ang = math.atan2(h[0] * u[1] - h[1] * u[0], float(h @ u))
    c, s = math.cos(ang), math.sin(ang)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return math.degrees(ang), mat_to_quat(rz @ r)
