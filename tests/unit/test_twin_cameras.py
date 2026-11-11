"""Digital twin camera window: frame ordering, tiling, rate limit, fail-safe (no GPU/GUI)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "twin_cameras", ROOT / "tools/twin/twin_cameras.py"
)
cams = importlib.util.module_from_spec(spec)
sys.modules["twin_cameras"] = cams
spec.loader.exec_module(cams)


def test_only_wrist_and_fixed_workspace_camera_are_shown() -> None:
    img = np.zeros((4, 4, 3), np.uint8)
    frames = {
        "head_camera": img,
        "observer": img,
        "left_camera": img,
        "right_camera": img,
        "workspace (config) camera": img,
    }
    labels = [label for label, _ in cams.order_frames(frames)]
    assert labels == ["wrist camera", "workspace camera"]


def test_tile_resizes_to_common_height_and_converts_floats() -> None:
    a = np.ones((240, 320, 3), np.float32) * 0.5
    b = np.zeros((480, 640, 4), np.uint8)
    canvas, offsets = cams.tile([("a", a), ("b", b)], 120)
    assert canvas.shape == (120, 160 + 160, 3) and canvas.dtype == np.uint8
    assert offsets == [0, 160] and canvas[0, 0, 0] == 127


class FakeCameras:
    def __init__(self):
        self.pictures = 0

    def update_picture(self):
        self.pictures += 1

    def get_rgb(self):
        return {"left_camera": {"rgb": np.zeros((8, 8, 3), np.uint8)}, "head_camera": {}}

    def get_observer_rgb(self):
        raise RuntimeError("no observer")


class FakeTask:
    def __init__(self):
        self.cameras = FakeCameras()
        self.steps = 0

    def _update_render(self):
        pass

    def _step_scene(self):
        self.steps += 1


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self):
        self.shown = []

    def putText(self, *a):
        pass

    def imshow(self, name, img):
        self.shown.append(img.shape)

    def waitKey(self, _):
        return -1


def test_step_hook_streams_at_limited_rate() -> None:
    task = FakeTask()
    stream = cams.CameraStream(None, hz=0.001, echo=lambda _: None)
    stream.cv2, stream.enabled = FakeCv2(), True
    stream.attach(task)
    for _ in range(5):
        task._step_scene()
    assert task.steps == 5 and task.cameras.pictures == 1
    stream.tick(force=True)
    assert len(stream.cv2.shown) == 2


def test_errors_disable_window_without_raising() -> None:
    task = FakeTask()
    messages = []
    stream = cams.CameraStream(task, hz=5, echo=messages.append)
    stream.cv2, stream.enabled = FakeCv2(), True
    task.cameras.get_rgb = lambda: (_ for _ in ()).throw(RuntimeError("render lost"))
    stream.tick(force=True)
    assert not stream.enabled and "render lost" in messages[0]
    stream.tick(force=True)
    assert len(messages) == 1


def test_zero_hz_disables_stream() -> None:
    assert not cams.CameraStream(None, hz=0).enabled
