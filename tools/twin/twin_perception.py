# ruff: noqa: E501
"""VLM object detection on the twin's workspace camera image, scored against simulator truth.

The workspace camera image is sent to a vision model through the model router (Gemma 4 first).
The model returns labelled 2D boxes (normalised 0-1000, [ymin, xmin, ymax, xmax]); each box
centre is projected onto a horizontal plane just above the table with the configured camera
intrinsics/extrinsics, giving an estimated (x, y) on the table. In simulation every estimate
is matched to the true object poses so that position error, label agreement, misses and false
detections are measured. On the real robot the same code runs without the truth step.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from twin_geometry import CameraSpec, as_model, pixel_to_plane, project, triangulate

MATCH_RADIUS_M = 0.06
PLANE_ABOVE_TABLE_M = 0.025  # typical object half height: box centres sit near this plane

DETECT_SYSTEM = (
    "You are the vision module of a tabletop robot. Detect every movable object on the table "
    "(ignore the robot arm, the table itself, trays, bins and flat pads). For each object give a "
    "short English label made of colour and shape/kind (for example 'red cube', 'small blue "
    "cylinder', 'yellow ball'); for everyday items name what the item is from its print and look "
    "(for example 'soy milk carton', 'fresh milk carton', 'box with number 2', 'tennis ball', 'volleyball', "
    "'orange juice bottle', 'water bottle'). Also give the colour, the shape, and a tight 2D "
    "bounding box as [ymin, xmin, ymax, xmax] normalised to 0-1000. Do not invent objects."
)
DETECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "color": {"type": "string"},
                    "shape": {"type": "string"},
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["label", "color", "shape", "box_2d"],
            },
        }
    },
    "required": ["objects"],
}
SHAPE_WORDS = {
    "cube": ("cube", "block", "box", "square"),
    "box": ("box", "carton", "block", "package", "pack"),
    "cylinder": ("cylinder", "can", "round", "tube"),
    "bottle": ("bottle",),
    "sphere": ("ball", "sphere"),
    "prism": ("prism", "triangle", "triangular"),
    "tetrahedron": ("tetrahedron", "pyramid", "triangle", "triangular"),
}


@dataclass
class Detection:
    label: str
    color: str
    shape: str
    box_px: tuple[float, float, float, float]  # u_min, v_min, u_max, v_max
    xy_est: tuple[float, float] | None = None
    view: str = ""
    ray: tuple | None = None  # (origin, unit direction) through the box centre
    z_est: float | None = None  # object centre height when triangulated from several views

    @property
    def area_px(self) -> float:
        u0, v0, u1, v1 = self.box_px
        return max(0.0, u1 - u0) * max(0.0, v1 - v0)

    @property
    def center_px(self) -> tuple[float, float]:
        u0, v0, u1, v1 = self.box_px
        return (u0 + u1) / 2, (v0 + v1) / 2


@dataclass
class Match:
    name: str
    detection: Detection
    error_m: float
    label_ok: bool


@dataclass
class PerceptionReport:
    matches: list[Match] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    extra: list[Detection] = field(default_factory=list)

    @property
    def mean_error_m(self) -> float | None:
        if not self.matches:
            return None
        return sum(m.error_m for m in self.matches) / len(self.matches)

    def summary(self) -> str:
        mean = self.mean_error_m
        labels = sum(m.label_ok for m in self.matches)
        return (
            f"matched {len(self.matches)}, missed {len(self.missed)}, extra {len(self.extra)}, "
            f"labels ok {labels}/{len(self.matches)}, mean xy error "
            + ("n/a" if mean is None else f"{mean * 100:.1f} cm")
        )


def validate_detections(payload: dict) -> str | None:
    objects = payload.get("objects")
    if not isinstance(objects, list):
        return "missing 'objects' list"
    for index, obj in enumerate(objects, start=1):
        box = obj.get("box_2d") if isinstance(obj, dict) else None
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(v, (int, float)) for v in box)
        ):
            return f"object {index}: box_2d must be 4 numbers"
    return None


def parse_detections(
    payload: dict, spec, table_top: float, view: str = "workspace"
) -> list[Detection]:
    """Boxes -> detections with a table-plane position estimate and a viewing ray."""
    model = as_model(spec)
    out = []
    for obj in payload.get("objects", []):
        ymin, xmin, ymax, xmax = (min(max(float(v), 0.0), 1000.0) for v in obj["box_2d"])
        if ymax <= ymin or xmax <= xmin:
            continue
        box = (
            xmin / 1000 * spec.width,
            ymin / 1000 * spec.height,
            xmax / 1000 * spec.width,
            ymax / 1000 * spec.height,
        )
        det = Detection(
            str(obj.get("label", "")).strip(),
            str(obj.get("color", "")).strip(),
            str(obj.get("shape", "")).strip(),
            box,
        )
        u, v = det.center_px
        det.xy_est = pixel_to_plane(model, u, v, table_top + PLANE_ABOVE_TABLE_M)
        det.view = view
        det.ray = model.ray(u, v)
        out.append(det)
    return out


def label_agrees(
    detection: Detection, color_name: str, shape: str, keywords: tuple[str, ...] = ()
) -> bool:
    """Colour + shape words for plain blocks; for look-alike items the kind keyword must appear."""
    text = f"{detection.label} {detection.color} {detection.shape}".casefold()
    if keywords:
        tokens = set(re.findall(r"[a-z0-9]+", text))
        return any(
            w.casefold() in tokens or (len(w) > 2 and w.casefold() in text) for w in keywords
        )
    color_ok = all(word in text for word in color_name.casefold().split())
    shape_ok = any(word in text for word in SHAPE_WORDS.get(shape, (shape,)))
    return color_ok and shape_ok


def match_to_truth(
    detections: list[Detection],
    truth: dict[str, tuple],
    radius: float = MATCH_RADIUS_M,
) -> PerceptionReport:
    """Greedy nearest matching. truth: name -> ((x, y), color_name, shape[, keywords])."""
    pairs = []
    for di, det in enumerate(detections):
        if det.xy_est is None:
            continue
        for name, entry in truth.items():
            xy = entry[0]
            pairs.append((math.dist(det.xy_est, xy), di, name))
    pairs.sort()
    used_d, used_t, report = set(), set(), PerceptionReport()
    for dist, di, name in pairs:
        if dist > radius or di in used_d or name in used_t:
            continue
        used_d.add(di)
        used_t.add(name)
        color, shape = truth[name][1], truth[name][2]
        keywords = truth[name][3] if len(truth[name]) > 3 else ()
        det = detections[di]
        report.matches.append(Match(name, det, dist, label_agrees(det, color, shape, keywords)))
    report.missed = [n for n in truth if n not in used_t]
    report.extra = [d for i, d in enumerate(detections) if i not in used_d]
    report.matches.sort(key=lambda m: m.name)
    return report


def report_lines(report: PerceptionReport) -> list[str]:
    rows = [f"PERCEPTION: {report.summary()}"]
    for m in report.matches:
        x, y = m.detection.xy_est
        rows.append(
            f"  {m.name}: seen as '{m.detection.label}' at x={x:+.3f} y={y:+.3f} "
            f"(error {m.error_m * 100:.1f} cm, label {'ok' if m.label_ok else 'MISMATCH'})"
        )
    rows += [f"  missed: {n}" for n in report.missed]
    rows += [f"  extra detection: '{d.label}'" for d in report.extra]
    return rows


def unique_labels(detections: list[Detection]) -> list[str]:
    """Scene names for planning from detections: 'red cube', 'red cube #2', ..."""
    seen: dict[str, int] = {}
    names = []
    for det in detections:
        base = det.label.casefold() or "object"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base} #{seen[base]}")
    return names


def annotate(image, detections: list[Detection], truth_px: dict[str, tuple[float, float]]):
    """Draw boxes (green) and true object centres (red crosses); returns a BGR copy (needs cv2)."""
    import cv2
    import numpy as np

    canvas = np.ascontiguousarray(np.asarray(image)[:, :, ::-1]).copy()
    for det in detections:
        u0, v0, u1, v1 = (round(v) for v in det.box_px)
        cv2.rectangle(canvas, (u0, v0), (u1, v1), (0, 200, 0), 2)
        cv2.putText(
            canvas, det.label, (u0, max(12, v0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1
        )
    for name, (u, v) in truth_px.items():
        cv2.drawMarker(canvas, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(
            canvas, name, (int(u) + 6, int(v) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )
    return canvas


def truth_pixels(
    spec: CameraSpec, truth_xyz: dict[str, list[float]]
) -> dict[str, tuple[float, float]]:
    out = {}
    for name, xyz in truth_xyz.items():
        px = project(spec, xyz)
        if px is not None:
            out[name] = px
    return out


# ----------------------------------------------------------------- multi-view fusion


@dataclass
class FusedObject:
    label: str
    color: str
    shape: str
    xy: tuple[float, float]
    z: float | None
    views: list[str]
    detections: list[Detection]


FUSE_RADIUS_M = 0.05


def fuse_views(
    detections: list[Detection], table_top: float, radius: float = FUSE_RADIUS_M
) -> list[FusedObject]:
    """Merge detections of the same object seen from several cameras/poses.

    Detections are clustered by their table-plane estimates (one detection per view per
    cluster). A cluster seen from two or more viewpoints is triangulated from its viewing rays,
    which also gives the object's height; otherwise the plane estimate is kept. The label comes
    from the largest (closest, most legible) box in the cluster.
    """
    clusters: list[list[Detection]] = []
    for det in sorted((d for d in detections if d.xy_est is not None), key=lambda d: -d.area_px):
        best, best_score = None, math.inf
        for cluster in clusters:
            if any(c.view == det.view for c in cluster):
                continue
            score = _association_score(cluster, det, table_top, radius)
            if score < best_score:
                best, best_score = cluster, score
        if best is None:
            clusters.append([det])
        else:
            best.append(det)
    fused = []
    for cluster in clusters:
        lead = cluster[0]  # largest box
        xy = (
            sum(c.xy_est[0] for c in cluster) / len(cluster),
            sum(c.xy_est[1] for c in cluster) / len(cluster),
        )
        z = None
        point = triangulate([c.ray for c in cluster if c.ray is not None])
        if point is not None and table_top - 0.01 <= point[2] <= table_top + 0.25:
            xy, z = (float(point[0]), float(point[1])), float(point[2])
        fused.append(
            FusedObject(
                lead.label, lead.color, lead.shape, xy, z, [c.view for c in cluster], cluster
            )
        )
    return fused


RAY_GAP_M = 0.03  # viewing rays of the same object pass within this distance


def ray_gap(a: tuple, b: tuple) -> tuple[float, np.ndarray] | None:
    """Closest distance between two rays and the midpoint of the closest approach."""
    (pa, da), (pb, db) = a, b
    w = pa - pb
    aa, bb, cc = float(da @ da), float(da @ db), float(db @ db)
    dd, ee = float(da @ w), float(db @ w)
    denom = aa * cc - bb * bb
    if abs(denom) < 1e-12:
        return None
    s = (bb * ee - cc * dd) / denom
    t = (aa * ee - bb * dd) / denom
    if s <= 0 or t <= 0:
        return None
    qa, qb = pa + s * da, pb + t * db
    return float(np.linalg.norm(qa - qb)), (qa + qb) / 2


def _association_score(
    cluster: list[Detection], det: Detection, table_top: float, radius: float
) -> float:
    """Lower is better; inf = not the same object. Rays meeting above the table, else plane distance."""
    lead = cluster[0]
    if lead.ray is not None and det.ray is not None:
        gap = ray_gap(lead.ray, det.ray)
        if gap is not None:
            distance, point = gap
            if distance < RAY_GAP_M and table_top - 0.01 <= point[2] <= table_top + 0.25:
                same_label = lead.label.casefold() == det.label.casefold()
                return distance - (0.01 if same_label else 0.0)
    cx = sum(c.xy_est[0] for c in cluster) / len(cluster)
    cy = sum(c.xy_est[1] for c in cluster) / len(cluster)
    dist = math.dist((cx, cy), det.xy_est)
    return dist + 1.0 if dist < radius else math.inf  # plane fallback ranks after ray matches


def fused_as_detections(fused: list[FusedObject]) -> list[Detection]:
    out = []
    for f in fused:
        lead = f.detections[0]
        out.append(
            Detection(f.label, f.color, f.shape, lead.box_px, f.xy, "+".join(f.views), None, f.z)
        )
    return out


# ----------------------------------------------------------------- change detection


CHANGE_PIXEL_DELTA = 30  # grey-level difference counted as a change
CHANGE_FRACTION = 0.002  # share of (unmasked) pixels that must change to trigger a rescan


CHANGE_BLUR = 2  # box-blur radius (px) before differencing: suppresses render noise
OBJECT_CHANGE = 0.15  # share of an object's image circle that must change to flag it


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img
    k = 2 * radius + 1
    pad = np.pad(img, radius + 1, mode="edge")
    c = pad.cumsum(axis=0).cumsum(axis=1)
    h, w = img.shape
    total = c[k : k + h, k : k + w] - c[0:h, k : k + w] - c[k : k + h, 0:w] + c[0:h, 0:w]
    return total / (k * k)


def _erode(mask: np.ndarray) -> np.ndarray:
    """3x3 erosion: isolated speckles (ray-tracing noise, aliasing) disappear."""
    pad = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.ones_like(mask)
    h, w = mask.shape
    for dv in range(3):
        for du in range(3):
            out &= pad[dv : dv + h, du : du + w]
    return out


def dilate(mask: np.ndarray, times: int = 1) -> np.ndarray:
    """3x3 dilation (``times`` passes): grow a pixel mask by one pixel per pass."""
    out = np.asarray(mask, bool)
    h, w = out.shape
    for _ in range(times):
        pad = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out)
        for dv in range(3):
            for du in range(3):
                grown |= pad[dv : dv + h, du : du + w]
        out = grown
    return out


def change_map(reference, current, delta: int = CHANGE_PIXEL_DELTA, blur: int = CHANGE_BLUR):
    """Boolean HxW map of pixels that really changed, or None if the image sizes differ."""
    ref = np.asarray(reference, np.float32).mean(axis=2)
    cur = np.asarray(current, np.float32).mean(axis=2)
    if ref.shape != cur.shape:
        return None
    return _erode(np.abs(_box_blur(cur, blur) - _box_blur(ref, blur)) > delta)


def changed_fraction(
    reference, current, mask=None, delta: int = CHANGE_PIXEL_DELTA, roi=None, changed=None
) -> float:
    """Share of watched pixels that changed (blurred, speckle-free grey-level difference).

    Only pixels inside ``roi`` (True = watched; default: whole image) and outside ``mask``
    (True = ignored, e.g. objects the plan moved) are counted. ``changed`` reuses a change_map.
    """
    if changed is None:
        changed = change_map(reference, current, delta)
    if changed is None:
        return 1.0
    keep = np.ones(changed.shape, bool) if roi is None else np.asarray(roi, bool).copy()
    if mask is not None:
        keep &= ~np.asarray(mask, bool)
    total = int(keep.sum())
    return float((changed & keep).sum()) / total if total else 0.0


def change_threshold(noise_floor: float | None) -> float:
    """Trigger level: CHANGE_FRACTION, raised above the measured frame-to-frame noise."""
    return max(CHANGE_FRACTION, 3.0 * (noise_floor or 0.0) + 0.0005)


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_mask(shape: tuple[int, int], polygon: list[tuple[float, float]]) -> np.ndarray:
    """Boolean HxW mask, True inside a convex counter-clockwise polygon (pixel coordinates)."""
    h, w = shape
    vv, uu = np.mgrid[0:h, 0:w]
    inside = np.ones((h, w), bool)
    n = len(polygon)
    if n < 3:
        return np.zeros((h, w), bool)
    for i in range(n):
        (x0, y0), (x1, y1) = polygon[i], polygon[(i + 1) % n]
        inside &= (x1 - x0) * (vv - y0) - (y1 - y0) * (uu - x0) >= 0
    return inside


WATCH_X = (-0.32, 0.32)
WATCH_Y = (-0.29, 0.06)  # objects + containers; the robot base (y=-0.45) stays outside
WATCH_HEIGHT = 0.16


def workspace_roi(camera, shape: tuple[int, int], table_top: float) -> np.ndarray:
    """Pixels covering the work area (table region up to WATCH_HEIGHT), excluding the robot."""
    model = as_model(camera)
    corners = []
    for x in WATCH_X:
        for y in WATCH_Y:
            for z in (table_top, table_top + WATCH_HEIGHT):
                px = model.project((x, y, z))
                if px is not None:
                    corners.append(px)
    return polygon_mask(shape, convex_hull(corners))


def region_changes(
    reference, current, circles: dict[str, tuple[float, float, float]], changed=None, mask=None
) -> dict[str, float]:
    """Changed share inside each named image circle (u, v, r): which object moved or vanished.

    ``mask`` (True = ignored) removes pixels of objects the plan moved on purpose.
    """
    if changed is None:
        changed = change_map(reference, current)
    if changed is None:
        return {name: 1.0 for name in circles}
    shape = changed.shape
    return {
        name: changed_fraction(None, None, mask, roi=circle_mask(shape, [c]), changed=changed)
        for name, c in circles.items()
    }


def circle_mask(shape: tuple[int, int], circles: list[tuple[float, float, float]]) -> np.ndarray:
    """Boolean HxW mask that is True inside each (u, v, radius_px) circle."""
    h, w = shape
    vv, uu = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), bool)
    for u, v, r in circles:
        mask |= (uu - u) ** 2 + (vv - v) ** 2 <= r * r
    return mask


def object_circles(
    camera, positions: list[tuple[float, float, float]], radius_m: float
) -> list[tuple[float, float, float]]:
    """Image circles covering spheres of ``radius_m`` around world points (for masking)."""
    model = as_model(camera)
    out = []
    for p in positions:
        centre = model.project(p)
        if centre is None:
            continue
        right = model.R[:, 0] * radius_m
        edge = model.project(np.asarray(p, float) + right)
        r = math.dist(centre, edge) if edge is not None else 40.0
        out.append((centre[0], centre[1], r))
    return out
