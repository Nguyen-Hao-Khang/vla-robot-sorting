"""Same native task camera refresh contract (offline mocks, no GPU physics)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_robot_sorting.perception import native_task_pair_capture as module


class Camera:
    def __init__(self) -> None:
        self.calls = 0

    def take_picture(self) -> None:
        self.calls += 1


@dataclass
class FakeTask:
    cameras: object

    def get_obs(self):
        self.cameras.update_picture()
        return {"native": True}


def fake_task(*, repeat: bool = False) -> FakeTask:
    head, wrist = Camera(), Camera()
    cameras = SimpleNamespace(
        single_arm_mode=True,
        collect_head_camera=True,
        collect_wrist_camera=True,
        static_camera_name=["head_camera"],
        static_camera_list=[head],
        wrist_camera=wrist,
    )

    def update() -> None:
        head.take_picture()
        wrist.take_picture()

    cameras.update_picture = update
    task = FakeTask(cameras)
    if repeat:
        task.get_obs = lambda: (cameras.update_picture(), cameras.update_picture())
    return task


def test_same_native_task_captures_only_two_exact_views(monkeypatch, tmp_path: Path):
    task = fake_task()
    recorded = []

    def capture(camera, **kwargs):
        recorded.append((camera, kwargs))
        return tmp_path / f"{kwargs['camera_id']}_frame_000000_depth-manifest.json"

    monkeypatch.setattr(module, "capture_sapien_depth_frame", capture)
    monkeypatch.setattr(
        module,
        "audit_native_metric_quality",
        lambda root: {
            "camera_count": 2,
            "controller_eligible": False,
            "status": "PASS_NON_CONTROL",
        },
    )
    # Capture function is responsible for actual manifest; mock pathlib only
    # in this unit test by substituting the JSON parse instead of pretending real data.
    original_read = Path.read_text
    def read_text(self, *args, **kwargs):
        if self.name.endswith("_depth-manifest.json"):
            return '{"status":"capture_written"}'
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    result = module.capture_task_native_pair(task, tmp_path / "new")
    assert result["camera_count"] == 2
    assert result["visual_object_identity_verified"] is False
    assert [row[1]["camera_id"] for row in recorded] == ["head_camera", "wrist_camera"]
    assert all(row[1]["scene_revision"] is None for row in recorded)
    assert all(row[1]["frame_index_verified"] is True for row in recorded)
    assert task.cameras.wrist_camera.calls == 1
    assert task.cameras.static_camera_list[0].calls == 1


def test_refuse_repeated_refresh_before_evidence(monkeypatch, tmp_path: Path):
    task = fake_task(repeat=True)
    with pytest.raises(ValueError, match="exactly once"):
        module.capture_task_native_pair(task, tmp_path / "new")


def test_refuse_non_single_arm(tmp_path: Path):
    task = fake_task()
    task.cameras.single_arm_mode = False
    with pytest.raises(ValueError, match="single arm"):
        module.capture_task_native_pair(task, tmp_path / "new")


def test_refuse_missing_camera(tmp_path: Path):
    task = fake_task()
    task.cameras.wrist_camera = None
    with pytest.raises(ValueError, match="exactly one"):
        module.capture_task_native_pair(task, tmp_path / "new")


def test_refuse_overwrite(tmp_path: Path):
    task = fake_task()
    with pytest.raises(ValueError, match="must not exist"):
        module.capture_task_native_pair(task, tmp_path)


def test_get_obs_exception_restores_original_camera_method(tmp_path: Path):
    task = fake_task()
    original = task.cameras.update_picture
    def broken():
        raise RuntimeError("device")
    task.get_obs = broken
    with pytest.raises(RuntimeError, match="device"):
        module.capture_task_native_pair(task, tmp_path / "new")
    assert task.cameras.update_picture is original
