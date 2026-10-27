"""Render profiles for resource-bounded RoboTwin validation."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

FAST_PREVIEW_RENDER_PROFILE: dict[str, object] = {
    "render_spp": 1,
    "render_path_depth": 2,
    "render_denoiser": "none",
}


def apply_fast_preview_render_profile(task_args: MutableMapping[str, Any]) -> None:
    """Apply RoboTwin's upstream fast-preview ray-tracing settings in place."""
    task_args.update(FAST_PREVIEW_RENDER_PROFILE)
