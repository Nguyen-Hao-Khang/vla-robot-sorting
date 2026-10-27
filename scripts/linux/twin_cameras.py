"""Live camera window for the digital twin: wrist (gripper) camera plus workspace cameras.

Renders RoboTwin's own task cameras (``task.cameras``) into one OpenCV window, rate limited
(default 5 Hz) both while idle and during motion (via a ``_step_scene`` hook). Display only:
any rendering/GUI error disables the window once with a message and never stops the robot.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

import numpy as np

WINDOW = "Twin cameras (wrist | workspace)"  # the RoboTwin head camera is not shown


def label_for(name: str) -> str:
    low = name.lower()
    if "left" in low or "wrist" in low or "hand" in low:
        return "wrist camera"
    if "config" in low:
        return name
    if "head" in low:
        return "workspace (head) camera"
    if "observer" in low or "third" in low:
        return "workspace (observer) camera"
    return f"{name}"


def order_frames(frames: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """Wrist first, then head, then observer/others; drop the right wrist in single-arm mode."""

    def rank(name: str) -> tuple[int, str]:
        label = label_for(name)
        if label == "wrist camera":
            return (0, name)
        if "config" in name.lower():
            return (1, name)
        if label.startswith("workspace (head)"):
            return (2, name)
        if label.startswith("workspace (observer)"):
            return (2, name)
        return (3, name)

    names = [n for n in frames if "right" not in n.lower()]
    # Two cameras only, as on the real robot: the wrist camera and the fixed workspace camera.
    names = [n for n in names if label_for(n) == "wrist camera" or "config" in n.lower()]
    return [(label_for(n).replace(" (config)", ""), frames[n]) for n in sorted(names, key=rank)]


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(np.nan_to_num(array) * scale, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return array[:, :, :3]


def resize_nearest(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    width = max(1, round(w * height / h))
    rows = (np.arange(height) * h / height).astype(int)
    cols = (np.arange(width) * w / width).astype(int)
    return image[rows][:, cols]


def tile(frames: list[tuple[str, np.ndarray]], height: int) -> tuple[np.ndarray, list[int]]:
    """Side-by-side RGB canvas of equal-height frames; returns canvas and each frame's x offset."""
    parts = [resize_nearest(to_uint8_rgb(img), height) for _, img in frames]
    offsets, x = [], 0
    for part in parts:
        offsets.append(x)
        x += part.shape[1]
    return np.concatenate(parts, axis=1), offsets


class CameraStream:
    def __init__(
        self,
        task: object,
        hz: float = 5.0,
        height: int = 240,
        echo: Callable[[str], None] = print,
    ) -> None:
        self.task, self.height, self.echo = task, height, echo
        self.period = 1.0 / hz if hz > 0 else 0.0
        self.last = float("-inf")
        self.enabled = hz > 0
        self.cv2 = None
        self._busy = False
        self.extra_sources: dict[str, Callable[[], np.ndarray | None]] = {}
        if self.enabled:
            try:
                import cv2

                self.cv2 = cv2
            except ImportError as exc:
                self._disable(f"OpenCV is not available ({exc})")

    def _disable(self, why: str) -> None:
        if self.enabled:
            self.echo(f"CAMERA_WINDOW_OFF: {why}. The twin keeps running without it.")
        self.enabled = False

    def attach(self, task: object) -> None:
        """Point the stream at a (new) task and hook its physics step for updates during motion."""
        self.task = task
        original = task._step_scene

        def stepped() -> None:
            original()
            self.tick()

        task._step_scene = stepped

    def grab(self) -> dict[str, np.ndarray]:
        task = self.task
        task._update_render()
        task.cameras.update_picture()
        frames = {
            name: data["rgb"] for name, data in task.cameras.get_rgb().items() if "rgb" in data
        }
        try:
            for name, data in task.cameras.get_observer_rgb().items():
                image = data.get("rgb") if isinstance(data, dict) else data
                if image is not None:
                    frames[name] = image
        except Exception:  # observer camera is optional
            pass
        for name, source in self.extra_sources.items():
            image = source()
            if image is not None:
                frames[name] = image
        return frames

    def tick(self, force: bool = False) -> None:
        if not self.enabled or self._busy or self.task is None:
            return
        now = time.perf_counter()
        if not force and now - self.last < self.period:
            return
        self.last = now
        self._busy = True
        try:
            frames = order_frames(self.grab())
            if not frames:
                self._disable("the task has no RGB cameras")
                return
            canvas, offsets = tile(frames, self.height)
            bgr = np.ascontiguousarray(canvas[:, :, ::-1])
            cv2 = self.cv2
            for (label, _), x in zip(frames, offsets, strict=True):
                cv2.putText(bgr, label, (x + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(
                    bgr, label, (x + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
            cv2.imshow(WINDOW, bgr)
            cv2.waitKey(1)
        except Exception as exc:  # display only; never break the motion
            self._disable(f"{type(exc).__name__}: {exc}")
        finally:
            self._busy = False

    def close(self) -> None:
        if self.cv2 is not None:
            with contextlib.suppress(Exception):
                self.cv2.destroyAllWindows()
