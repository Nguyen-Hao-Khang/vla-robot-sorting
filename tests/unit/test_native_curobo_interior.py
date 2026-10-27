"""Prove in-memory CuRobo clip reaches both vendor generators, without CUDA."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vla_robot_sorting.planning import native_curobo_interior as native


def _install_fake_vendor(monkeypatch, original_path: Path):
    curobo = ModuleType("curobo")
    wrap = ModuleType("curobo.wrap")
    reacher = ModuleType("curobo.wrap.reacher")
    motion_gen = ModuleType("curobo.wrap.reacher.motion_gen")
    original_config = {
        "robot_cfg": {
            "kinematics": {
                "cspace": {
                    "joint_names": [f"joint{i}" for i in range(1, 7)],
                    "position_limit_clip": 0.0,
                    "retract_config": [0.0] * 6,
                }
            }
        }
    }
    other = {"arbitrary": "unmodified"}
    observed = []

    def loader(path):
        return copy.deepcopy(original_config if Path(path).resolve() == original_path else other)

    def factory(config, *args, **kwargs):
        observed.append((copy.deepcopy(motion_gen.load_yaml(config)), dict(kwargs)))
        return SimpleNamespace()

    motion_gen.load_yaml = loader

    class FakeMotionGenConfig:
        load_from_robot_config = staticmethod(factory)

    motion_gen.MotionGenConfig = FakeMotionGenConfig
    curobo.wrap = wrap
    wrap.reacher = reacher
    reacher.motion_gen = motion_gen
    for name, mod in (
        ("curobo", curobo),
        ("curobo.wrap", wrap),
        ("curobo.wrap.reacher", reacher),
        ("curobo.wrap.reacher.motion_gen", motion_gen),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return motion_gen, observed, loader, factory


def test_native_curobo_exact_yml_hash_and_lf_normalization(tmp_path: Path):
    source = Path(
        "D:/DAMH-Bridge/audit-current/Panthera-HT_RoboTwin-pinned/"
        "RoboTwin/assets/embodiments/panthera-6dof/curobo.yml"
    )
    if not source.exists():
        pytest.skip("pinned audit source unavailable")
    original = source.read_bytes()
    normalized = original.replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized).hexdigest() == native.PINNED_CUROBO_YAML_LF_SHA256
    fixture = tmp_path / "curobo.yml"
    fixture.write_bytes(normalized)
    native.verify_pinned_curobo_yaml(fixture)
    fixture.write_bytes(normalized + b"\n# unpinned change\n")
    with pytest.raises(RuntimeError, match="sha256"):
        native.verify_pinned_curobo_yaml(fixture)


def test_scoped_loader_and_factory_enforce_clip_and_restore(monkeypatch, tmp_path):
    path = tmp_path / "curobo.yml"
    path.write_text("test", encoding="utf8")
    vendor, observed, loader, factory = _install_fake_vendor(monkeypatch, path)
    monkeypatch.setattr(native, "verify_pinned_curobo_yaml", lambda _: None)

    with native.native_curobo_interior_setup(path) as report:
        vendor.MotionGenConfig.load_from_robot_config(str(path), {"table": 1}, num_trajopt_seeds=1)
        vendor.MotionGenConfig.load_from_robot_config(
            str(path), {"table": 1}, num_trajopt_seeds=1, num_graph_seeds=1
        )
    assert report["config_intercepts"] == 2
    assert report["factory_calls"] == 2
    assert all(
        row[0]["robot_cfg"]["kinematics"]["cspace"]["position_limit_clip"]
        == native.CUROBO_POSITION_LIMIT_CLIP_RAD
        for row in observed
    )
    assert observed[0][1]["num_ik_seeds"] == native.PRIMARY_IK_SEEDS
    assert observed[0][1]["num_trajopt_seeds"] == native.PRIMARY_TRAJOPT_SEEDS
    assert observed[1][1]["num_trajopt_seeds"] == 1
    assert vendor.load_yaml is loader
    assert vendor.MotionGenConfig.load_from_robot_config is factory
    assert vendor.load_yaml(path)["robot_cfg"]["kinematics"]["cspace"]["position_limit_clip"] == 0.0


def test_scoped_interior_loader_exception_restores_original(monkeypatch, tmp_path):
    path = tmp_path / "curobo.yml"
    path.write_text("test", encoding="utf8")
    vendor, _, loader, factory = _install_fake_vendor(monkeypatch, path)
    monkeypatch.setattr(native, "verify_pinned_curobo_yaml", lambda _: None)
    with (
        pytest.raises(ValueError, match="test interrupt"),
        native.native_curobo_interior_setup(path),
    ):
        vendor.MotionGenConfig.load_from_robot_config(str(path), num_trajopt_seeds=1)
        raise ValueError("test interrupt")
    assert vendor.load_yaml is loader
    assert vendor.MotionGenConfig.load_from_robot_config is factory


def test_premature_setup_without_generator_fails_closed(monkeypatch, tmp_path):
    path = tmp_path / "curobo.yml"
    path.write_text("test", encoding="utf8")
    vendor, _, _, _ = _install_fake_vendor(monkeypatch, path)
    monkeypatch.setattr(native, "verify_pinned_curobo_yaml", lambda _: None)
    with (
        pytest.raises(RuntimeError, match="did not reach"),
        native.native_curobo_interior_setup(path),
    ):
        vendor.MotionGenConfig.load_from_robot_config(str(path), num_trajopt_seeds=1)


def test_carried_motiongen_uses_same_clip_without_modifying_input(monkeypatch):
    from vla_robot_sorting.planning.carried_object import PantheraRoboTwinCarriedObjectPlanner

    curobo = ModuleType("curobo")
    wrap = ModuleType("curobo.wrap")
    reacher = ModuleType("curobo.wrap.reacher")
    motion_gen = ModuleType("curobo.wrap.reacher.motion_gen")
    captured = []

    class Config:
        @staticmethod
        def load_from_robot_config(config, world, **kwargs):
            captured.append((copy.deepcopy(config), dict(world), dict(kwargs)))
            return "config-ok"

    class MotionGen:
        def __init__(self, cfg):
            assert cfg == "config-ok"

        def warmup(self):
            captured.append("warmup")

    motion_gen.MotionGenConfig = Config
    motion_gen.MotionGen = MotionGen
    curobo.wrap = wrap
    wrap.reacher = reacher
    reacher.motion_gen = motion_gen
    for name, mod in (
        ("curobo", curobo),
        ("curobo.wrap", wrap),
        ("curobo.wrap.reacher", reacher),
        ("curobo.wrap.reacher.motion_gen", motion_gen),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    original = {
        "robot_cfg": {"kinematics": {"cspace": {"joint_names": [f"joint{i}" for i in range(1, 7)]}}}
    }
    planner = object.__new__(PantheraRoboTwinCarriedObjectPlanner)
    returned = planner._native_motion_gen_factory(original, {"table": "pinned"})
    assert isinstance(returned, MotionGen)
    assert captured[0][0]["robot_cfg"]["kinematics"]["cspace"]["position_limit_clip"] == (
        native.CUROBO_POSITION_LIMIT_CLIP_RAD
    )
    assert captured[0][2]["num_ik_seeds"] == native.PRIMARY_IK_SEEDS
    assert captured[0][2]["num_trajopt_seeds"] == 1
    assert captured[1] == "warmup"
    assert "position_limit_clip" not in original["robot_cfg"]["kinematics"]["cspace"]


def test_scoped_vendor_search_config_widens_attempts_without_leaking(monkeypatch):
    mod = ModuleType("envs.robot.planner")
    calls = []

    def original_config(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs

    mod.MotionGenPlanConfig = original_config
    monkeypatch.setitem(sys.modules, "envs.robot.planner", mod)
    vendor_cls = type("CuroboPlanner", (), {"__module__": "envs.robot.planner"})
    robot = SimpleNamespace(left_planner=vendor_cls())
    with native.scoped_native_curobo_search(robot) as policy:
        configured = mod.MotionGenPlanConfig(max_attempts=10)
        assert configured["max_attempts"] == 24
    assert policy["factory_invocations"] == 1
    assert policy["vendor_original_restored"] is True
    assert mod.MotionGenPlanConfig is original_config

    with (
        pytest.raises(RuntimeError, match="unexpected pinned"),
        native.scoped_native_curobo_search(robot),
    ):
        mod.MotionGenPlanConfig(max_attempts=15)
    assert mod.MotionGenPlanConfig is original_config


def test_pinned_curobo_implementation_clips_both_joint_position_boundaries():
    source = Path(
        "D:/DAMH-Bridge/audit-current/curobo-pinned/"
        "src/curobo/cuda_robot_model/cuda_robot_generator.py"
    )
    if not source.is_file():
        pytest.skip("pinned CuRobo source checkout unavailable")
    text = source.read_text(encoding="utf8")
    assert 'joint_limits["position"][0] += self.cspace.position_limit_clip' in text
    assert 'joint_limits["position"][1] -= self.cspace.position_limit_clip' in text
    assert 0.005 < native.CUROBO_POSITION_LIMIT_CLIP_RAD < 0.02
    assert native.INTERIOR_RETRACT_CONFIG_RAD[1:3] == (0.05, 0.0625)
