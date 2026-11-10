from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/linux/probe_panthera_grasp_recovery.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "probe_panthera_grasp_recovery_aperture_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panthera_symmetric_joint_travel_maps_to_relative_aperture() -> None:
    module = _module()

    minimum, maximum = (
        module.symmetric_parallel_jaw_aperture_from_joint_scale([0.0, 0.04])
    )

    assert minimum == pytest.approx(0.0)
    assert maximum == pytest.approx(0.08)


def test_load_gripper_proxy_uses_full_relative_aperture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setitem(
        sys.modules,
        "yaml",
        SimpleNamespace(
            safe_load=lambda _handle: {
                "collision_spheres": {
                    "gripper_L_link": [{"radius": 0.018}],
                    "gripper_R_link": [{"radius": 0.018}],
                }
            }
        ),
    )

    sphere_path = (
        tmp_path
        / "assets"
        / "embodiments"
        / "panthera-6dof"
        / "collision_panthera.yml"
    )
    sphere_path.parent.mkdir(parents=True)
    sphere_path.write_text(
        """
collision_spheres:
  gripper_L_link:
    - center: [0.0, 0.0, 0.0]
      radius: 0.018
  gripper_R_link:
    - center: [0.0, 0.0, 0.0]
      radius: 0.018
""".strip()
        + "\n",
        encoding="utf-8",
    )

    gripper, provenance = module.load_gripper_proxy(
        tmp_path,
        {
            "left_embodiment_config": {
                "gripper_scale": [0.0, 0.04],
            }
        },
    )

    assert gripper.min_width_m == pytest.approx(0.0)
    assert gripper.max_width_m == pytest.approx(0.08)
    assert gripper.contact_center_offset_m == pytest.approx(0.10)
    assert gripper.pregrasp_standoff_m == pytest.approx(0.08)
    assert provenance["task_pre_grasp_dis_m"] == pytest.approx(0.06)
    assert provenance["pregrasp_standoff_m"] == pytest.approx(0.08)
    assert provenance["contact_center_offset_m"] == pytest.approx(0.10)
    assert provenance["task_grasp_frame_to_contact_m"] == pytest.approx(0.12)
    assert provenance["panthera_default_grasp_dis_m"] == pytest.approx(-0.02)
    assert provenance["per_finger_joint_scale_m"] == [0.0, 0.04]
    assert provenance["relative_aperture_multiplier"] == pytest.approx(2.0)
