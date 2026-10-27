from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/linux/probe_panthera_grasp_recovery.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "probe_panthera_grasp_recovery_helpers_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePlanner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def plan(self, start_state, goal, context, *, scene):
        self.calls.append((start_state, goal, context, scene))
        return self.result


def _plan(planner_id: str, *, succeeded: bool, planning_time_s: float):
    return SimpleNamespace(
        planner_id=planner_id,
        succeeded=succeeded,
        planning_time_s=planning_time_s,
        failure=None,
    )


def test_primary_success_skips_fallback() -> None:
    module = _module()
    primary_result = _plan(
        "robotwin-curobo",
        succeeded=True,
        planning_time_s=0.04,
    )
    fallback_result = _plan(
        "robotwin-mplib-rrt",
        succeeded=True,
        planning_time_s=0.08,
    )
    primary = FakePlanner(primary_result)
    fallback = FakePlanner(fallback_result)

    selected, provenance = module.plan_primary_fallback(
        primary,
        fallback,
        "start",
        "goal",
        "context",
        scene={"scene": 1},
    )

    assert selected is primary_result
    assert len(primary.calls) == 1
    assert fallback.calls == []
    assert provenance == {
        "primary_planner_id": "robotwin-curobo",
        "primary_success": True,
        "fallback_attempted": False,
        "fallback_planner_id": None,
        "fallback_success": None,
        "selected_planner_id": "robotwin-curobo",
        "plan_success": True,
        "planning_time_s": 0.04,
    }


def test_primary_failure_uses_mplib_fallback() -> None:
    module = _module()
    primary_result = _plan(
        "robotwin-curobo",
        succeeded=False,
        planning_time_s=0.03,
    )
    fallback_result = _plan(
        "robotwin-mplib-rrt",
        succeeded=True,
        planning_time_s=0.07,
    )
    primary = FakePlanner(primary_result)
    fallback = FakePlanner(fallback_result)

    selected, provenance = module.plan_primary_fallback(
        primary,
        fallback,
        "start",
        "goal",
        "context",
        scene={"scene": 2},
    )

    assert selected is fallback_result
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    assert provenance["primary_planner_id"] == "robotwin-curobo"
    assert provenance["primary_success"] is False
    assert provenance["fallback_attempted"] is True
    assert provenance["fallback_planner_id"] == "robotwin-mplib-rrt"
    assert provenance["fallback_success"] is True
    assert provenance["selected_planner_id"] == "robotwin-mplib-rrt"
    assert provenance["plan_success"] is True
    assert provenance["planning_time_s"] == 0.10


def test_collision_hit_payload_preserves_pair_identity_fields() -> None:
    module = _module()
    hit = SimpleNamespace(
        link_name1="gripper_L_link",
        body_name2="pillbottle_actor",
        object_name2="pillbottle",
    )

    payload = module.collision_hit_payload(hit)

    assert payload["attributes"]["link_name1"] == "gripper_L_link"
    assert payload["attributes"]["body_name2"] == "pillbottle_actor"
    assert payload["attributes"]["object_name2"] == "pillbottle"


def test_collision_bounds_proxy_fits_tall_near_round_mesh_as_cylinder() -> None:
    module = _module()

    geometry, provenance = module.collision_bounds_geometry_proxy(
        (0.074118492603302, 0.15479133799672126, 0.0732840633392334),
        mesh_max_radial_distance_m=0.039,
        mesh_long_axis_local_index=1,
    )

    assert type(geometry).__name__ == "CylinderGeometry"
    assert geometry.axis_local == (0.0, 1.0, 0.0)
    assert geometry.radius_m == pytest.approx(0.039)
    assert geometry.height_m == pytest.approx(0.15479133799672126)
    assert provenance["primitive"] == "cylinder"
    assert provenance["long_axis_local_index"] == 1
    assert provenance["radial_aspect_ratio"] < 1.02


def test_collision_bounds_proxy_keeps_non_cylindrical_mesh_as_box() -> None:
    module = _module()

    geometry, provenance = module.collision_bounds_geometry_proxy((0.04, 0.08, 0.06))

    assert type(geometry).__name__ == "BoxGeometry"
    assert geometry.size_xyz_m == pytest.approx((0.04, 0.08, 0.06))
    assert provenance["primitive"] == "box"


def test_near_round_aabb_without_mesh_radius_fails_safe_to_box() -> None:
    module = _module()

    geometry, provenance = module.collision_bounds_geometry_proxy(
        (0.074118492603302, 0.15479133799672126, 0.0732840633392334)
    )

    assert type(geometry).__name__ == "BoxGeometry"
    assert provenance["primitive"] == "box"
    assert provenance["mesh_max_radial_distance_m"] is None


def test_mesh_vertex_radius_uses_farthest_radial_vertex() -> None:
    module = _module()

    radius = module.max_radial_distance_from_vertices(
        [
            (0.03, -0.08, 0.04),
            (-0.03, 0.08, -0.04),
            (0.0, 0.0, 0.02),
        ],
        center_xyz_m=(0.0, 0.0, 0.0),
        long_axis_index=1,
    )

    assert radius == pytest.approx(0.05)


def _install_fake_mplib_target_modules(monkeypatch):
    class FakeArray:
        def __init__(self, values):
            self.values = [
                list(item) if isinstance(item, (list, tuple)) else item
                for item in values
            ]

        def __iter__(self):
            return iter(self.values)

        def __getitem__(self, index):
            return self.values[index]

        def __mul__(self, other):
            if isinstance(other, FakeArray):
                scale = other.values
                return FakeArray(
                    [
                        [float(value) * float(scale[index]) for index, value in enumerate(row)]
                        for row in self.values
                    ]
                )
            return FakeArray([float(value) * float(other) for value in self.values])

    numpy_module = ModuleType("numpy")
    numpy_module.asarray = lambda values, dtype=None: FakeArray(values)
    numpy_module.allclose = lambda values, expected: all(
        abs(float(value) - float(expected)) <= 1e-9 for value in values
    )
    numpy_module.pi = 3.141592653589793

    class FakePose:
        def __init__(self, value=None, q=None):
            self.value = value
            self.q = q

        def __imul__(self, other):
            return self

    class FakeGeometry:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeCollisionObject:
        def __init__(self, geometry):
            self.geometry = geometry

    class FakeFCLObject:
        def __init__(self, name, pose, shapes, shape_poses):
            self.name = name
            self.pose = pose
            self.shapes = shapes
            self.shape_poses = shape_poses

    class FakeBVHModel:
        def begin_model(self):
            pass

        def add_sub_model(self, **kwargs):
            self.kwargs = kwargs

        def end_model(self):
            pass

    class DummyShape:
        pass

    class FakeConvexMesh:
        def __init__(self):
            self.local_pose = "local-pose"
            self.scale = (0.08, 0.08, 0.08)
            self.vertices = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0))
            self.triangles = ((0, 1, 2),)

    class FakeRigidBase:
        pass

    mplib_module = ModuleType("mplib")
    mplib_module.Pose = FakePose
    collision_pkg = ModuleType("mplib.collision_detection")
    fcl = ModuleType("mplib.collision_detection.fcl")
    fcl.BVHModel = FakeBVHModel
    fcl.Box = lambda **kwargs: FakeGeometry(**kwargs)
    fcl.Capsule = lambda **kwargs: FakeGeometry(**kwargs)
    fcl.CollisionObject = FakeCollisionObject
    fcl.Convex = lambda **kwargs: FakeGeometry(**kwargs)
    fcl.Cylinder = lambda **kwargs: FakeGeometry(**kwargs)
    fcl.FCLObject = FakeFCLObject
    fcl.Sphere = lambda **kwargs: FakeGeometry(**kwargs)

    sapien_utils = ModuleType("mplib.sapien_utils")
    conversion = ModuleType("mplib.sapien_utils.conversion")
    conversion.convert_object_name = lambda actor: "080_pillbottle_23"

    sapien = ModuleType("sapien")
    physx = ModuleType("sapien.physx")
    physx.PhysxCollisionShapeBox = DummyShape
    physx.PhysxCollisionShapeCapsule = DummyShape
    physx.PhysxCollisionShapeConvexMesh = FakeConvexMesh
    physx.PhysxCollisionShapeCylinder = DummyShape
    physx.PhysxCollisionShapeSphere = DummyShape
    physx.PhysxCollisionShapeTriangleMesh = DummyShape
    physx.PhysxRigidBaseComponent = FakeRigidBase

    transforms3d = ModuleType("transforms3d")
    euler = ModuleType("transforms3d.euler")
    euler.euler2quat = lambda *args: (1.0, 0.0, 0.0, 0.0)

    for name, value in {
        "numpy": numpy_module,
        "mplib": mplib_module,
        "mplib.collision_detection": collision_pkg,
        "mplib.collision_detection.fcl": fcl,
        "mplib.sapien_utils": sapien_utils,
        "mplib.sapien_utils.conversion": conversion,
        "sapien": sapien,
        "sapien.physx": physx,
        "transforms3d": transforms3d,
        "transforms3d.euler": euler,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)

    return FakeConvexMesh(), FakeRigidBase


def test_scaled_target_sync_reuses_planner_and_applies_mesh_scale(monkeypatch) -> None:
    module = _module()
    convex_shape, rigid_type = _install_fake_mplib_target_modules(monkeypatch)

    class FakeWorld:
        def __init__(self):
            self.objects = {}

        def has_object(self, name):
            return name in self.objects

        def remove_object(self, name):
            self.objects.pop(name, None)

        def add_object(self, obj):
            self.objects[obj.name] = obj

    world = FakeWorld()
    planner = SimpleNamespace(planning_world=world)
    component = SimpleNamespace(collision_shapes=[convex_shape])
    actor = SimpleNamespace(
        pose="actor-pose",
        find_component_by_type=lambda typ: component if typ is rigid_type else None,
    )
    wrapper = SimpleNamespace(planner=planner)
    task = SimpleNamespace(
        pillbottle=SimpleNamespace(actor=actor),
        robot=SimpleNamespace(
            left_mplib_planner=wrapper,
            left_move_group="tool_link",
        ),
    )

    provenance = module.sync_scaled_target_into_mplib_world(task)

    assert wrapper.planner is planner
    target = world.objects["080_pillbottle_23"]
    scaled_vertices = target.shapes[0].geometry.vertices
    assert abs(scaled_vertices[0][0] - 0.08) < 1e-9
    assert abs(scaled_vertices[1][1] - 0.16) < 1e-9
    assert abs(scaled_vertices[2][2] - 0.24) < 1e-9
    assert provenance["strategy"] == "scaled_target_fcl_injection"
    assert provenance["target_present"] is True
    assert provenance["collision_shape_count"] == 1
    assert provenance["scaled_mesh_shape_count"] == 1
    assert provenance["non_unit_scale_shape_count"] == 1
    assert provenance["scale_vectors"] == [[0.08, 0.08, 0.08]]
    assert provenance["planner_reused"] is True
    assert provenance["planning_world_reused"] is True


def test_scaled_target_sync_fails_closed_if_world_drops_object(monkeypatch) -> None:
    module = _module()
    convex_shape, rigid_type = _install_fake_mplib_target_modules(monkeypatch)

    class DroppingWorld:
        def has_object(self, name):
            return False

        def remove_object(self, name):
            pass

        def add_object(self, obj):
            pass

    component = SimpleNamespace(collision_shapes=[convex_shape])
    actor = SimpleNamespace(
        pose="actor-pose",
        find_component_by_type=lambda typ: component if typ is rigid_type else None,
    )
    task = SimpleNamespace(
        pillbottle=SimpleNamespace(actor=actor),
        robot=SimpleNamespace(
            left_mplib_planner=SimpleNamespace(
                planner=SimpleNamespace(planning_world=DroppingWorld())
            ),
            left_move_group="tool_link",
        ),
    )

    with pytest.raises(RuntimeError, match="failed to register target"):
        module.sync_scaled_target_into_mplib_world(task)
