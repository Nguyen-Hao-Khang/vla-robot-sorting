"""Validated reader for recorded RoboTwin camera observations in HDF5.

Optional camera-replay dependencies (h5py, numpy, Pillow) are imported only when
this module is used. RGB frames are not object detections and this reader never
creates metric object poses, depth, or controller-ready commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any

CAMERA_IDS = ("head_camera", "wrist_camera")
DATASET_NAMES = ("rgb", "intrinsic_cv", "extrinsic_cv", "cam2world_gl")
CV_FROM_GL = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class NativeEpisodeError(ValueError):
    """Native episode data is missing, corrupt, or inconsistent."""


@dataclass(frozen=True, slots=True)
class EpisodeProvenance:
    episode_path: str
    episode_sha256: str
    episode_size_bytes: int
    schema_version: str | None
    robot_type: str | None
    collection_fingerprint: str | None
    source_classification: str
    export_classification: str
    frame_index: int
    frame_count: int
    camera_id: str
    rgb_sha256: str
    calibration_sha256: str
    timestamp_s: float | None
    timestamp_source: str | None
    scene_revision: str | None
    scene_revision_source: str | None
    depth_available: bool
    depth_dataset_shape: tuple[int, ...] | None
    object_poses_available: bool
    controller_eligible: bool = False


@dataclass(frozen=True, slots=True)
class RobotEpisodeFields:
    joint_action_arm: tuple[float, ...]
    joint_action_gripper: float
    endpose_arm: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NativeCameraFrame:
    camera_id: str
    frame_index: int
    frame_count: int
    rgb_encoding: str
    rgb_bytes: bytes
    rgb_image: Any
    intrinsic_matrix: tuple[tuple[float, ...], ...]
    extrinsic_world_to_camera_cv: tuple[tuple[float, ...], ...]
    camera_to_world_gl: tuple[tuple[float, ...], ...]
    camera_to_world_cv: tuple[tuple[float, ...], ...]
    dataset_inventory: tuple[tuple[str, tuple[int, ...], str], ...]
    cv_gl_max_abs_error: float
    robot_fields: RobotEpisodeFields
    provenance: EpisodeProvenance


@dataclass(frozen=True, slots=True)
class ConventionAudit:
    camera_count: int
    frame_count_per_camera: int
    checked_transform_count: int
    max_abs_error: float
    tolerance: float
    convention: str


def camera_to_world_cv_from_cam2world_gl(matrix: Any) -> tuple[tuple[float, ...], ...]:
    """Convert RoboTwin/SAPIEN OpenGL camera-to-world into CV camera-to-world.

    Camera-local OpenGL uses +Y up and -Z forward; CV uses +Y down and +Z
    forward. The two-axis sign conversion is right-handed and is post-multiplied
    because it changes camera-local coordinates, not world coordinates.
    """

    import numpy as np

    camera_to_world_gl = np.asarray(matrix, dtype=np.float64)
    if camera_to_world_gl.shape != (4, 4) or not np.isfinite(camera_to_world_gl).all():
        raise NativeEpisodeError("cam2world_gl must be a finite 4x4 matrix")
    if not np.allclose(camera_to_world_gl[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5):
        raise NativeEpisodeError("cam2world_gl must be a homogeneous transform")
    _validate_rotation(camera_to_world_gl[:3, :3], "cam2world_gl")
    converted = camera_to_world_gl @ np.asarray(CV_FROM_GL, dtype=np.float64)
    return _as_matrix_tuple(converted)


def _as_matrix_tuple(matrix: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _validate_rotation(rotation: Any, label: str) -> None:
    import numpy as np

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise NativeEpisodeError(f"{label} rotation must be finite and 3x3")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-4):
        raise NativeEpisodeError(f"{label} rotation must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 2e-4:
        raise NativeEpisodeError(f"{label} rotation must be right-handed")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class NativeEpisode:
    """Read and validate selected RGB/calibration frames from an immutable HDF5."""

    def __init__(self, episode_path: str | Path, *, expected_sha256: str) -> None:
        self.path = Path(episode_path)
        if not self.path.is_file():
            raise NativeEpisodeError(f"episode file does not exist: {self.path}")
        if (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise NativeEpisodeError("expected_sha256 must be a lowercase SHA256 digest")
        self.sha256 = _file_sha256(self.path)
        if self.sha256 != expected_sha256:
            raise NativeEpisodeError(
                f"episode SHA256 mismatch: expected {expected_sha256}, got {self.sha256}"
            )
        try:
            import h5py

            with h5py.File(self.path, "r") as episode:
                attrs = episode.attrs
                self.schema_version = _optional_text(attrs, "schema_version")
                self.robot_type = _optional_text(attrs, "robot_type")
                self.collection_fingerprint = _optional_text(attrs, "collection_fingerprint")
                self._root_attributes = {str(key): _plain(value) for key, value in attrs.items()}
        except Exception as exc:
            raise NativeEpisodeError(f"cannot open native HDF5 episode: {exc}") from exc

    def read_frame(self, camera_id: str, frame_index: int) -> NativeCameraFrame:
        if camera_id not in CAMERA_IDS:
            raise NativeEpisodeError(f"camera must be one of {', '.join(CAMERA_IDS)}")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise NativeEpisodeError("frame_index must be a non-negative integer")
        try:
            import h5py
            import numpy as np
            from PIL import Image, UnidentifiedImageError

            with h5py.File(self.path, "r") as episode:
                group_path = f"observation/{camera_id}"
                datasets = {}
                for name in DATASET_NAMES:
                    dataset_path = f"{group_path}/{name}"
                    if dataset_path not in episode:
                        raise NativeEpisodeError(f"required HDF5 dataset missing: {dataset_path}")
                    datasets[name] = episode[dataset_path]
                count = int(datasets["rgb"].shape[0])
                if frame_index >= count:
                    raise NativeEpisodeError(
                        f"frame_index {frame_index} is outside [0, {count - 1}]"
                    )
                for name, dataset in datasets.items():
                    if len(dataset.shape) == 0 or dataset.shape[0] != count:
                        raise NativeEpisodeError(
                            f"dataset {group_path}/{name} has inconsistent frame count"
                        )
                encoded = bytes(datasets["rgb"][frame_index])
                if encoded.startswith(b"\xff\xd8\xff"):
                    encoding = "jpeg"
                else:
                    raise NativeEpisodeError(
                        f"unsupported RGB encoding at frame {frame_index}; expected JPEG bytes"
                    )
                try:
                    with Image.open(BytesIO(encoded)) as source_image:
                        source_image.load()
                        rgb_image = source_image.convert("RGB").copy()
                except (UnidentifiedImageError, OSError) as exc:
                    raise NativeEpisodeError(
                        f"cannot decode RGB JPEG at frame {frame_index}: {exc}"
                    ) from exc

                raw_calibration = tuple(datasets[name][frame_index] for name in DATASET_NAMES[1:])
                calibration_digest = sha256()
                for matrix in raw_calibration:
                    calibration_digest.update(np.asarray(matrix).tobytes(order="C"))
                intrinsic = np.asarray(raw_calibration[0], dtype=np.float64)
                extrinsic = np.asarray(raw_calibration[1], dtype=np.float64)
                camera_to_world_gl = np.asarray(raw_calibration[2], dtype=np.float64)
                _validate_intrinsics(intrinsic)
                camera_to_world_cv = np.asarray(
                    camera_to_world_cv_from_cam2world_gl(camera_to_world_gl)
                )
                if extrinsic.shape != (3, 4) or not np.isfinite(extrinsic).all():
                    raise NativeEpisodeError("extrinsic_cv must be a finite 3x4 matrix")
                _validate_rotation(extrinsic[:, :3], "extrinsic_cv")
                camera_to_world_from_extrinsic = np.eye(4, dtype=np.float64)
                camera_to_world_from_extrinsic[:3, :] = np.linalg.inv(
                    np.vstack((extrinsic, (0.0, 0.0, 0.0, 1.0)))
                )[:3, :]
                error = float(np.max(np.abs(camera_to_world_from_extrinsic - camera_to_world_cv)))
                if error > 2e-4:
                    raise NativeEpisodeError(
                        "extrinsic_cv and cam2world_gl are inconsistent after CV/GL conversion "
                        f"(max abs error {error:.6g})"
                    )
                height, width = rgb_image.height, rgb_image.width
                if width <= 0 or height <= 0:
                    raise NativeEpisodeError("decoded RGB image has invalid dimensions")
                robot_fields = _read_robot_fields(episode, frame_index, count)
                timestamp_s, timestamp_source = _read_optional_scalar(
                    episode, frame_index, ("timestamp_s", "observation/timestamp_s")
                )
                scene_revision = _optional_text(episode.attrs, "scene_revision")
                depth_shape, depth_available = _depth_shape(
                    episode,
                    count,
                    camera_id=camera_id,
                    image_width=width,
                    image_height=height,
                )
                inventory_paths = [f"{group_path}/{name}" for name in DATASET_NAMES]
                inventory_paths.extend(
                    ("joint_action/arm", "joint_action/gripper", "endpose/arm", "pointcloud")
                )
                dataset_inventory = tuple(
                    (
                        dataset_path,
                        tuple(int(value) for value in episode[dataset_path].shape),
                        str(episode[dataset_path].dtype),
                    )
                    for dataset_path in inventory_paths
                    if dataset_path in episode
                )
                provenance = EpisodeProvenance(
                    episode_path=str(self.path.resolve()),
                    episode_sha256=self.sha256,
                    episode_size_bytes=self.path.stat().st_size,
                    schema_version=self.schema_version,
                    robot_type=self.robot_type,
                    collection_fingerprint=self.collection_fingerprint,
                    source_classification="recorded_sim_observation",
                    export_classification="not_assessed",
                    frame_index=frame_index,
                    frame_count=count,
                    camera_id=camera_id,
                    rgb_sha256=sha256(encoded).hexdigest(),
                    calibration_sha256=calibration_digest.hexdigest(),
                    timestamp_s=timestamp_s,
                    timestamp_source=timestamp_source,
                    scene_revision=scene_revision,
                    scene_revision_source="hdf5_root_attribute" if scene_revision else None,
                    depth_available=depth_available,
                    depth_dataset_shape=depth_shape,
                    object_poses_available=False,
                )
                return NativeCameraFrame(
                    camera_id=camera_id,
                    frame_index=frame_index,
                    frame_count=count,
                    rgb_encoding=encoding,
                    rgb_bytes=encoded,
                    rgb_image=rgb_image,
                    intrinsic_matrix=_as_matrix_tuple(intrinsic),
                    extrinsic_world_to_camera_cv=_as_matrix_tuple(extrinsic),
                    camera_to_world_gl=_as_matrix_tuple(camera_to_world_gl),
                    camera_to_world_cv=_as_matrix_tuple(camera_to_world_cv),
                    dataset_inventory=dataset_inventory,
                    cv_gl_max_abs_error=error,
                    robot_fields=robot_fields,
                    provenance=provenance,
                )
        except NativeEpisodeError:
            raise
        except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
            raise NativeEpisodeError(f"invalid native HDF5 frame: {exc}") from exc


    def audit_camera_conventions(self, *, tolerance: float = 2e-4) -> ConventionAudit:
        import h5py
        import numpy as np

        if not isfinite(tolerance) or tolerance <= 0:
            raise NativeEpisodeError("tolerance must be finite and positive")
        max_error = 0.0
        frame_count: int | None = None
        checked = 0
        try:
            with h5py.File(self.path, "r") as episode:
                for camera_id in CAMERA_IDS:
                    group_path = f"observation/{camera_id}"
                    ext_path = f"{group_path}/extrinsic_cv"
                    gl_path = f"{group_path}/cam2world_gl"
                    if ext_path not in episode or gl_path not in episode:
                        raise NativeEpisodeError(
                            f"required calibration dataset missing for {camera_id}"
                        )
                    extrinsics = np.asarray(episode[ext_path], dtype=np.float64)
                    gl_matrices = np.asarray(episode[gl_path], dtype=np.float64)
                    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
                        raise NativeEpisodeError(f"{ext_path} must have shape (frames, 3, 4)")
                    if gl_matrices.shape != (len(extrinsics), 4, 4):
                        raise NativeEpisodeError(
                            f"{gl_path} must have shape ({len(extrinsics)}, 4, 4)"
                        )
                    if frame_count is None:
                        frame_count = len(extrinsics)
                    elif frame_count != len(extrinsics):
                        raise NativeEpisodeError("camera frame counts are inconsistent")
                    for index, (extrinsic, cam2world_gl) in enumerate(
                        zip(extrinsics, gl_matrices, strict=True)
                    ):
                        if not np.isfinite(extrinsic).all():
                            raise NativeEpisodeError(f"{ext_path}[{index}] is non-finite")
                        _validate_rotation(extrinsic[:, :3], f"{ext_path}[{index}]")
                        camera_to_world_cv = np.asarray(
                            camera_to_world_cv_from_cam2world_gl(cam2world_gl)
                        )
                        extrinsic_h = np.vstack((extrinsic, (0.0, 0.0, 0.0, 1.0)))
                        converted = np.linalg.inv(extrinsic_h)
                        error = float(np.max(np.abs(converted - camera_to_world_cv)))
                        if error > tolerance:
                            raise NativeEpisodeError(
                                f"{camera_id} frame {index} calibration is inconsistent "
                                f"(max abs error {error:.6g}, tolerance {tolerance:.6g})"
                            )
                        max_error = max(max_error, error)
                        checked += 1
        except NativeEpisodeError:
            raise
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise NativeEpisodeError(f"cannot audit HDF5 camera conventions: {exc}") from exc
        if frame_count is None:
            raise NativeEpisodeError("episode contains no camera calibration frames")
        return ConventionAudit(
            camera_count=len(CAMERA_IDS),
            frame_count_per_camera=frame_count,
            checked_transform_count=checked,
            max_abs_error=max_error,
            tolerance=tolerance,
            convention=(
                "cam2world_cv = cam2world_gl @ diag(1,-1,-1,1); "
                "extrinsic_cv = inverse(cam2world_cv)[:3,:]"
            ),
        )


def _optional_text(attributes: Any, key: str) -> str | None:
    if key not in attributes:
        return None
    value = _plain(attributes[key])
    return str(value) if value is not None else None


def _validate_intrinsics(matrix: Any) -> None:
    import numpy as np

    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise NativeEpisodeError("intrinsic_cv must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise NativeEpisodeError("intrinsic_cv focal lengths must be positive")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-6):
        raise NativeEpisodeError("intrinsic_cv must have canonical last row")
    if abs(matrix[0, 1]) > 1e-8 or abs(matrix[1, 0]) > 1e-8:
        raise NativeEpisodeError("intrinsic_cv skew is unsupported by the pixel model")


def _read_robot_fields(episode: Any, index: int, count: int) -> RobotEpisodeFields:
    import numpy as np

    names = ("joint_action/arm", "joint_action/gripper", "endpose/arm")
    for name in names:
        if name not in episode:
            raise NativeEpisodeError(f"required proprioception dataset missing: {name}")
        if episode[name].shape[0] != count:
            raise NativeEpisodeError(f"proprioception dataset {name} has inconsistent frame count")
    arm = np.asarray(episode[names[0]][index], dtype=np.float64)
    gripper = float(episode[names[1]][index])
    endpose = np.asarray(episode[names[2]][index], dtype=np.float64)
    if arm.shape != (6,) or endpose.shape != (7,):
        raise NativeEpisodeError("arm joint vector must have 6 values and endpose must have 7")
    if not np.isfinite(arm).all() or not isfinite(gripper) or not np.isfinite(endpose).all():
        raise NativeEpisodeError("proprioception values must be finite")
    return RobotEpisodeFields(
        joint_action_arm=tuple(float(value) for value in arm),
        joint_action_gripper=gripper,
        endpose_arm=tuple(float(value) for value in endpose),
    )


def _read_optional_scalar(
    episode: Any, index: int, dataset_paths: tuple[str, ...]
) -> tuple[float | None, str | None]:
    for path in dataset_paths:
        if path in episode:
            values = episode[path]
            if values.shape[0] <= index:
                raise NativeEpisodeError(f"timestamp dataset {path} is shorter than camera data")
            value = float(values[index])
            if not isfinite(value) or value < 0:
                raise NativeEpisodeError(
                    f"timestamp dataset {path} must contain non-negative seconds"
                )
            return value, path
    return None, None


def _depth_shape(
    episode: Any,
    count: int,
    *,
    camera_id: str,
    image_width: int,
    image_height: int,
) -> tuple[tuple[int, ...] | None, bool]:
    """Distinguish camera-aligned depth from unorganized pointcloud data.

    A nonempty point cloud is never, by itself, a depth measurement for a
    particular RGB pixel. Even camera-aligned depth needs verified units before
    metric XYZ localization may be claimed by downstream consumers.
    """
    depth_path = f"observation/{camera_id}/depth"
    if depth_path in episode:
        dataset = episode[depth_path]
        shape = tuple(int(value) for value in dataset.shape)
        if shape != (count, image_height, image_width):
            raise NativeEpisodeError(
                f"camera depth {depth_path} must align with the selected RGB frame"
            )
        if dataset.dtype.kind not in ("f", "u", "i"):
            raise NativeEpisodeError(f"camera depth {depth_path} must be numeric")
        return shape, True
    if "pointcloud" in episode:
        shape = tuple(int(value) for value in episode["pointcloud"].shape)
        if not shape or shape[0] != count:
            raise NativeEpisodeError("pointcloud has inconsistent frame count")
        return shape, False
    return None, False
