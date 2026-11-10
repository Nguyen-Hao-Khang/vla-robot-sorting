"""Camera geometry for the digital twin: configurable workspace camera, projection, table rays.

SAPIEN camera frame: x forward, y left, z up. Image pixels: u to the right, v down.
The workspace camera pose is read from configs/twin_cameras.json so that it can be set to the
measured pose of the real camera once it is mounted (sim and real images then share a view).
Pure numpy; unit-tested without SAPIEN.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CAMERA_CONFIG = PROJECT / "configs" / "twin_cameras.json"
DEFAULT_WORKSPACE = {
    "position": [0.0, 0.12, 1.25],
    "look_at": [0.0, -0.12, 0.74],
    "fovy_deg": 55.0,
    "width": 640,
    "height": 480,
}


@dataclass(frozen=True)
class CameraSpec:
    name: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fovy_deg: float
    width: int
    height: int

    @property
    def rotation(self) -> np.ndarray:
        return look_at_rotation(self.position, self.look_at)

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        return mat_to_quat(self.rotation)

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        return intrinsics(self.width, self.height, self.fovy_deg)


def load_camera_spec(path: Path = CAMERA_CONFIG, name: str = "workspace") -> CameraSpec:
    data = dict(DEFAULT_WORKSPACE)
    if path.is_file():
        data.update(json.loads(path.read_text(encoding="utf-8")).get(name, {}))
    return CameraSpec(
        name,
        tuple(float(v) for v in data["position"]),
        tuple(float(v) for v in data["look_at"]),
        float(data["fovy_deg"]),
        int(data["width"]),
        int(data["height"]),
    )


def look_at_rotation(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Columns: forward (x), left (y), up (z) of a SAPIEN camera looking at ``target``."""
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    left = np.cross(np.asarray(up, float), forward)
    if np.linalg.norm(left) < 1e-9:  # looking straight down/up: pick any left
        left = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    left /= np.linalg.norm(left)
    cam_up = np.cross(forward, left)
    return np.stack([forward, left, cam_up], axis=1)


def mat_to_quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    m = np.asarray(rot, float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    sign = 1.0 if w >= 0 else -1.0
    return tuple(sign * v / norm for v in (w, x, y, z))


def intrinsics(width: int, height: int, fovy_deg: float) -> tuple[float, float, float, float]:
    fy = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return fy, fy, width / 2, height / 2  # square pixels


# SAPIEN camera axes (x forward, y left, z up) -> OpenCV axes (x right, y down, z forward).
SAPIEN_TO_CV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
GL_TO_CV = np.diag([1.0, -1.0, -1.0])  # OpenGL camera (x right, y up, z back) -> OpenCV


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera in OpenCV convention: K (3x3) and camera-to-world pose (R, t).

    The same model serves the simulated cameras and, after calibration, the real ones:
    workspace camera = fixed extrinsic from a ChArUco calibration; wrist camera = robot
    forward kinematics x hand-eye calibration.
    """

    name: str
    K: np.ndarray
    R: np.ndarray  # columns: camera x/y/z axes in world
    t: np.ndarray  # camera centre in world
    width: int
    height: int

    @classmethod
    def from_spec(cls, spec: CameraSpec) -> CameraModel:
        fx, fy, cx, cy = spec.intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        R = spec.rotation @ SAPIEN_TO_CV
        return cls(spec.name, K, R, np.asarray(spec.position, float), spec.width, spec.height)

    @classmethod
    def from_sapien_pose(
        cls, name: str, position, quaternion, width: int, height: int, fovy_deg: float
    ) -> CameraModel:
        fx, fy, cx, cy = intrinsics(width, height, fovy_deg)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        R = quat_to_mat(quaternion) @ SAPIEN_TO_CV
        return cls(name, K, R, np.asarray(position, float), width, height)

    @classmethod
    def from_gl_matrices(cls, name: str, K, cam2world_gl, width: int, height: int) -> CameraModel:
        """From RoboTwin/SAPIEN camera config: intrinsic_cv and cam2world_gl (model matrix)."""
        m = np.asarray(cam2world_gl, float)
        return cls(name, np.asarray(K, float), m[:3, :3] @ GL_TO_CV, m[:3, 3].copy(), width, height)

    def project(self, point) -> tuple[float, float] | None:
        local = self.R.T @ (np.asarray(point, float) - self.t)
        if local[2] <= 1e-6:
            return None
        u = self.K[0, 0] * local[0] / local[2] + self.K[0, 2]
        v = self.K[1, 1] * local[1] / local[2] + self.K[1, 2]
        return float(u), float(v)

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        local = np.array(
            [(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], 1.0]
        )
        direction = self.R @ local
        return self.t.copy(), direction / np.linalg.norm(direction)

    def to_plane(self, u: float, v: float, plane_z: float) -> tuple[float, float] | None:
        origin, direction = self.ray(u, v)
        if abs(direction[2]) < 1e-9:
            return None
        t = (plane_z - origin[2]) / direction[2]
        if t <= 0:
            return None
        hit = origin + t * direction
        return float(hit[0]), float(hit[1])


def as_model(camera) -> CameraModel:
    return camera if isinstance(camera, CameraModel) else CameraModel.from_spec(camera)


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def project(spec, point) -> tuple[float, float] | None:
    """World point -> pixel (u, v); None if the point is behind the camera."""
    return as_model(spec).project(point)


def pixel_ray(spec, u: float, v: float) -> np.ndarray:
    return as_model(spec).ray(u, v)[1]


def pixel_to_plane(spec, u: float, v: float, plane_z: float) -> tuple[float, float] | None:
    """Intersect the pixel's viewing ray with the horizontal plane z = plane_z."""
    return as_model(spec).to_plane(u, v, plane_z)


def triangulate(
    rays: list[tuple[np.ndarray, np.ndarray]], min_angle_deg: float = 8.0
) -> np.ndarray | None:
    """Least-squares point closest to all rays (origin, unit direction); None if nearly parallel."""
    if len(rays) < 2:
        return None
    dirs = [d / np.linalg.norm(d) for _, d in rays]
    widest = max(
        math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))
        for i, a in enumerate(dirs)
        for b in dirs[i + 1 :]
    )
    if widest < min_angle_deg:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for (origin, _), d in zip(rays, dirs, strict=True):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ origin
    return np.linalg.solve(A, b)


def add_sapien_camera(scene, spec: CameraSpec):
    """Mount a render camera at the configured pose (RoboTwin env only)."""
    import sapien

    camera = scene.add_camera(
        spec.name, spec.width, spec.height, math.radians(spec.fovy_deg), 0.05, 10.0
    )
    camera.entity.set_pose(sapien.Pose(list(spec.position), list(spec.quaternion)))
    return camera


def read_rgb(camera) -> np.ndarray:
    """uint8 HxWx3 image from a SAPIEN render camera (after scene.update_render())."""
    camera.take_picture()
    rgba = np.asarray(camera.get_picture("Color"))
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    return rgba[:, :, :3]
