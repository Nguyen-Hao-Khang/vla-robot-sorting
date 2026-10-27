from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_robot_sorting.contracts import JointState, MotionGoal, PlanningContext, Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    CylinderGeometry,
    GeometrySource,
    SceneObject,
    SceneSnapshot,
)
from vla_robot_sorting.planning import (
    PantheraRoboTwinCarriedObjectPlanner,
    carried_support_contact_diagnostic,
    carried_world_from_scene,
    prepare_carried_robot_config,
)
from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES


class FakeRobot:
    def __init__(self) -> None:
        self.left_entity_origion_pose = SimpleNamespace(
            p=(0.0, -0.45, 0.75),
            q=(0.7071067812, 0.0, 0.0, 0.7071067812),
        )
        self.measured = [0.0] * 6

    def get_left_arm_real_jointState(self):
        return [*self.measured, 1.0]


class FakeWorldChecker:
    def __init__(self) -> None:
        self.enabled = {"table": True}
        self.calls: list[tuple[str, bool]] = []

    def get_obstacle_names(self) -> list[str]:
        return list(self.enabled)

    def enable_obstacle(self, name: str, *, enable: bool = True) -> None:
        if name not in self.enabled:
            raise ValueError(f"unknown obstacle {name}")
        self.enabled[name] = enable
        self.calls.append((name, enable))


class FakeMotionGen:
    def __init__(self) -> None:
        self.detached = False
        self.world_coll_checker = FakeWorldChecker()

    def check_start_state(self, native_state):
        del native_state
        if self.world_coll_checker.enabled["table"]:
            return False, "Start state is colliding with world"
        return True, None

    def detach_object_from_robot(self, *, link_name: str) -> None:
        assert link_name == "attached_object"
        self.detached = True


class FakeCarriedPlanner(PantheraRoboTwinCarriedObjectPlanner):
    def __init__(self, robot: FakeRobot, config_path: Path) -> None:
        self.raw_config = {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": "robot.urdf",
                    "collision_spheres": "spheres.yml",
                    "extra_collision_spheres": None,
                }
            }
        }
        self.last_config = None
        self.last_world = None
        super().__init__(
            robot,
            config_path=config_path,
            object_id="pillbottle",
            attachment_parent_link="tool_link",
            motion_gen_factory=self._factory,
        )

    def _load_yaml(self):
        return deepcopy(self.raw_config)

    def _factory(self, config, world):
        self.last_config = config
        self.last_world = world
        return FakeMotionGen()

    def _native_joint_state(self, start_state):
        return tuple(start_state.positions)

    def _attach_native(self, motion_gen, start_state):
        del motion_gen, start_state
        return True

    def _plan_native(self, motion_gen, start_state, target_gripper_pose):
        del motion_gen, target_gripper_pose
        return {
            "status": "Success",
            "position": [
                start_state.positions,
                (0.1, -0.1, 0.2, -0.2, 0.1, 0.0),
            ],
            "velocity": [
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.1, -0.1, 0.1, -0.1, 0.05, 0.0),
            ],
        }


def _scene() -> SceneSnapshot:
    target = SceneObject(
        object_id="pillbottle",
        pose=Pose6D(
            position_xyz=(0.10, 0.05, 0.80),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        geometry=BoxGeometry((0.06, 0.04, 0.12)),
        geometry_source=GeometrySource.SIM_GROUND_TRUTH,
        semantic_labels=("pill bottle",),
        identity_confidence=1.0,
    )
    return SceneSnapshot(
        scene_revision="scene-1",
        transform_revision="tf-1",
        frame_id="world",
        objects=(target,),
        table_height_m=0.735,
        metadata={
            "curobo_table_collision": {
                "frame_id": "world",
                "center_xyz_m": (0.0, 0.0, 0.715),
                "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
                "dims_xyz_m": (0.7, 2.0, 0.04),
                "source": "unit-test",
            }
        },
    )


def _state() -> JointState:
    return JointState(PANTHERA_JOINT_NAMES, (0.0,) * 6)


def _goal() -> MotionGoal:
    return MotionGoal(
        Pose6D(
            position_xyz=(0.1, 0.0, 0.9),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        )
    )


def test_prepare_carried_config_is_copy_and_reserves_spheres(tmp_path: Path) -> None:
    raw = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": "robot.urdf",
                "collision_spheres": "spheres.yml",
                "extra_collision_spheres": None,
            }
        }
    }
    original = deepcopy(raw)
    config_path = tmp_path / "curobo.yml"

    prepared = prepare_carried_robot_config(
        raw,
        config_path=config_path,
        attachment_parent_link="tool_link",
        sphere_count=64,
    )

    assert raw == original
    kin = prepared["robot_cfg"]["kinematics"]
    assert Path(kin["urdf_path"]).is_absolute()
    assert Path(kin["collision_spheres"]).is_absolute()
    assert kin["extra_collision_spheres"]["attached_object"] == 64
    extra_link = kin["extra_links"]["attached_object"]
    assert extra_link["parent_link_name"] == "tool_link"
    assert extra_link["link_name"] == "attached_object"
    assert extra_link["joint_type"] == "FIXED"
    assert extra_link["joint_name"] == "attached_object_joint"
    assert extra_link["fixed_transform"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert "attached_object" in kin["self_collision_ignore"]["tool_link"]
    assert "attached_object" in kin["self_collision_ignore"]["gripper_L_link"]
    assert "attached_object" in kin["self_collision_ignore"]["gripper_R_link"]
    assert kin["self_collision_buffer"]["attached_object"] == 0.0


def test_carried_world_transforms_table_and_object_to_panthera_base() -> None:
    world, provenance = carried_world_from_scene(
        FakeRobot(),
        _scene(),
        object_id="pillbottle",
    )

    table = world["cuboid"]["table"]
    target = world["cuboid"]["pillbottle"]
    assert table["dims"] == pytest.approx([0.7, 2.0, 0.04])
    assert target["dims"] == pytest.approx([0.06, 0.04, 0.12])
    # +90 degree world yaw means world +y maps to base +x.
    assert table["pose"][0] == pytest.approx(0.45, abs=1e-6)
    assert table["pose"][2] == pytest.approx(-0.035, abs=1e-6)
    assert target["pose"][0] == pytest.approx(0.50, abs=1e-6)
    assert target["pose"][1] == pytest.approx(-0.10, abs=1e-6)
    assert target["pose"][2] == pytest.approx(0.05, abs=1e-6)
    assert provenance["object_id"] == "pillbottle"


def test_carried_planner_fails_closed_before_attachment(tmp_path: Path) -> None:
    planner = FakeCarriedPlanner(FakeRobot(), tmp_path / "curobo.yml")

    result = planner.plan(
        _state(),
        _goal(),
        PlanningContext("scene-1", "tf-1"),
        _scene(),
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code.value == "backend_unavailable"


def test_carried_planner_initializes_attachment_and_plans(tmp_path: Path) -> None:
    planner = FakeCarriedPlanner(FakeRobot(), tmp_path / "curobo.yml")

    provenance = planner.initialize_attachment(_state(), _scene())
    result = planner.plan(
        _state(),
        _goal(),
        PlanningContext("scene-1", "tf-1"),
        _scene(),
    )

    assert provenance["reserved_sphere_count"] == 100
    assert provenance["pinned_config_mutated_on_disk"] is False
    assert planner.last_config["robot_cfg"]["kinematics"]["extra_collision_spheres"][
        "attached_object"
    ] == 100
    assert "pillbottle" in planner.last_world["cuboid"]
    assert result.succeeded
    assert result.path is not None
    assert result.path.source == "robotwin-curobo-carried-object"
    assert result.path.metadata["carried_object"]["object_id"] == "pillbottle"
    assert result.path.metadata["vendor_sample_count"] == 2
    assert result.path.metadata["vendor_interpolation_dt_s"] == pytest.approx(0.004)
    assert result.path.metadata["vendor_velocity_available"] is True
    assert result.path.metadata["vendor_velocity_samples_rad_s"] == (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.1, -0.1, 0.1, -0.1, 0.05, 0.0),
    )

    motion_gen = planner._motion_gen
    planner.detach()
    assert isinstance(motion_gen, FakeMotionGen)
    assert motion_gen.detached


def test_carried_planner_rejects_stale_start_state(tmp_path: Path) -> None:
    robot = FakeRobot()
    robot.measured[0] = 0.2
    planner = FakeCarriedPlanner(robot, tmp_path / "curobo.yml")

    with pytest.raises(ValueError, match="stale"):
        planner.initialize_attachment(_state(), _scene())


def test_carried_world_wraps_principal_axis_cylinder_in_conservative_cuboid() -> None:
    scene = _scene()
    original = scene.objects[0]
    cylinder = SceneObject(
        object_id=original.object_id,
        pose=original.pose,
        geometry=CylinderGeometry(
            radius_m=0.037,
            height_m=0.155,
            axis_local=(0.0, 1.0, 0.0),
        ),
        geometry_source=original.geometry_source,
        semantic_labels=original.semantic_labels,
        identity_confidence=original.identity_confidence,
    )
    cylinder_scene = SceneSnapshot(
        scene_revision=scene.scene_revision,
        transform_revision=scene.transform_revision,
        frame_id=scene.frame_id,
        objects=(cylinder,),
        table_height_m=scene.table_height_m,
        metadata=scene.metadata,
    )

    world, provenance = carried_world_from_scene(
        FakeRobot(),
        cylinder_scene,
        object_id="pillbottle",
    )

    target = world["cuboid"]["pillbottle"]
    assert target["dims"] == pytest.approx([0.074, 0.155, 0.074])
    assert provenance["object_geometry"] == "cylinder"
    geometry = provenance["object_geometry_provenance"]
    assert geometry["representation"] == "conservative_cuboid_envelope"
    assert geometry["cylinder_axis_local"] == (0.0, 1.0, 0.0)


def test_prepare_carried_config_explicit_asset_overrides_win(tmp_path: Path) -> None:
    raw = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": "../../../../wrong/robot.urdf",
                "collision_spheres": "../../../../wrong/spheres.yml",
                "extra_collision_spheres": None,
            }
        }
    }
    config_path = tmp_path / "assets" / "embodiments" / "panthera-6dof" / "curobo.yml"
    config_path.parent.mkdir(parents=True)
    urdf = config_path.parent / "panthera_6dof.urdf"
    spheres = config_path.parent / "collision_panthera.yml"
    urdf.write_text("<robot/>", encoding="utf-8")
    spheres.write_text("collision_spheres: {}", encoding="utf-8")

    prepared = prepare_carried_robot_config(
        raw,
        config_path=config_path,
        attachment_parent_link="tool_link",
        asset_path_overrides={
            "urdf_path": urdf,
            "collision_spheres": spheres,
        },
    )

    kin = prepared["robot_cfg"]["kinematics"]
    assert kin["urdf_path"] == str(urdf.resolve())
    assert kin["collision_spheres"] == str(spheres.resolve())


def test_prepare_carried_config_rejects_missing_asset_override(tmp_path: Path) -> None:
    raw = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": "robot.urdf",
                "collision_spheres": "spheres.yml",
                "extra_collision_spheres": None,
            }
        }
    }

    with pytest.raises(FileNotFoundError, match="collision_spheres"):
        prepare_carried_robot_config(
            raw,
            config_path=tmp_path / "curobo.yml",
            attachment_parent_link="tool_link",
            asset_path_overrides={
                "collision_spheres": tmp_path / "missing.yml",
            },
        )


def test_prepare_carried_config_registers_virtual_link_in_collision_list(
    tmp_path: Path,
) -> None:
    raw = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": "robot.urdf",
                "collision_spheres": "spheres.yml",
                "collision_link_names": ["tool_link", "gripper_L_link", "gripper_R_link"],
                "self_collision_ignore": {
                    "tool_link": ["gripper_L_link", "gripper_R_link"],
                },
                "extra_collision_spheres": None,
            }
        }
    }
    original = deepcopy(raw)

    prepared = prepare_carried_robot_config(
        raw,
        config_path=tmp_path / "curobo.yml",
        attachment_parent_link="tool_link",
        sphere_count=32,
    )

    assert raw == original
    kin = prepared["robot_cfg"]["kinematics"]
    assert kin["collision_link_names"].count("attached_object") == 1
    assert "attached_object" in kin["self_collision_ignore"]["tool_link"]
    assert "attached_object" in kin["self_collision_ignore"]["gripper_L_link"]
    assert "attached_object" in kin["self_collision_ignore"]["gripper_R_link"]


def test_prepare_carried_config_rejects_conflicting_virtual_link(tmp_path: Path) -> None:
    raw = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": "robot.urdf",
                "collision_spheres": "spheres.yml",
                "extra_links": {
                    "attached_object": {
                        "parent_link_name": "wrong_link",
                    }
                },
                "extra_collision_spheres": None,
            }
        }
    }

    with pytest.raises(ValueError, match="conflicts"):
        prepare_carried_robot_config(
            raw,
            config_path=tmp_path / "curobo.yml",
            attachment_parent_link="tool_link",
        )


def test_carried_planner_provenance_records_virtual_attachment_contract(
    tmp_path: Path,
) -> None:
    planner = FakeCarriedPlanner(FakeRobot(), tmp_path / "curobo.yml")

    provenance = planner.initialize_attachment(_state(), _scene())

    assert provenance["attached_object_link"] == "attached_object"
    assert provenance["attachment_parent_link"] == "tool_link"
    assert provenance["attachment_joint_name"] == "attached_object_joint"
    assert provenance["attachment_contact_links"] == [
        "gripper_L_link",
        "gripper_R_link",
    ]
    assert provenance["world_obstacle_removed_on_attach"] is True


def test_carried_start_diagnostic_reports_world_collision(tmp_path: Path) -> None:
    planner = FakeCarriedPlanner(FakeRobot(), tmp_path / "curobo.yml")
    planner.initialize_attachment(_state(), _scene())

    diagnostic = planner.diagnose_start_state(_state())

    assert diagnostic == {
        "available": True,
        "valid": False,
        "status": "Start state is colliding with world",
    }


def test_table_disabled_diagnostic_is_planning_only_and_reenables_table(
    tmp_path: Path,
) -> None:
    planner = FakeCarriedPlanner(FakeRobot(), tmp_path / "curobo.yml")
    planner.initialize_attachment(_state(), _scene())

    diagnostic = planner.diagnose_plan_with_obstacle_disabled(
        _state(),
        _goal(),
        obstacle_name="table",
    )

    assert diagnostic["planning_only"] is True
    assert diagnostic["executed"] is False
    assert diagnostic["obstacle_disabled"] is True
    assert diagnostic["obstacle_reenabled"] is True
    assert diagnostic["start_state"]["valid"] is True
    assert diagnostic["plan_success"] is True
    assert diagnostic["path_samples"] == 2
    assert diagnostic["world_obstacle_names"] == ["table"]
    motion_gen = planner._motion_gen
    assert isinstance(motion_gen, FakeMotionGen)
    assert motion_gen.world_coll_checker.enabled["table"] is True
    assert motion_gen.world_coll_checker.calls == [
        ("table", False),
        ("table", True),
    ]


def test_table_disabled_diagnostic_reenables_table_after_planner_exception(
    tmp_path: Path,
) -> None:
    class FailingPlanner(FakeCarriedPlanner):
        def _plan_native(self, motion_gen, start_state, target_gripper_pose):
            del motion_gen, start_state, target_gripper_pose
            raise RuntimeError("diagnostic planning failed")

    planner = FailingPlanner(FakeRobot(), tmp_path / "curobo.yml")
    planner.initialize_attachment(_state(), _scene())

    diagnostic = planner.diagnose_plan_with_obstacle_disabled(
        _state(),
        _goal(),
        obstacle_name="table",
    )

    assert diagnostic["plan_success"] is False
    assert diagnostic["executed"] is False
    assert diagnostic["obstacle_reenabled"] is True
    assert diagnostic["error"] == "RuntimeError: diagnostic planning failed"
    motion_gen = planner._motion_gen
    assert isinstance(motion_gen, FakeMotionGen)
    assert motion_gen.world_coll_checker.enabled["table"] is True


def test_carried_failure_preserves_native_motiongen_status(tmp_path: Path) -> None:
    class NativeFailPlanner(FakeCarriedPlanner):
        def _plan_native(self, motion_gen, start_state, target_gripper_pose):
            del motion_gen, start_state, target_gripper_pose
            return {
                "status": "Fail",
                "native_status": "IK Fail",
                "native_diagnostics": {
                    "valid_query": True,
                    "status": "IK Fail",
                    "attempts": 10,
                    "trajopt_attempts": 0,
                    "used_graph": False,
                    "start_state": {
                        "available": True,
                        "valid": True,
                        "status": None,
                    },
                },
            }

    planner = NativeFailPlanner(FakeRobot(), tmp_path / "curobo.yml")
    planner.initialize_attachment(_state(), _scene())

    result = planner.plan(
        _state(),
        _goal(),
        PlanningContext("scene-1", "tf-1"),
        _scene(),
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code.value == "optimization_failed"
    assert result.failure.message == "carried-object CuRobo status=IK Fail"
    assert result.failure.details["native_status"] == "IK Fail"
    assert result.failure.details["native_diagnostics"]["valid_query"] is True


def test_support_contact_geometry_detects_native_seed0_overlap() -> None:
    diagnostic = carried_support_contact_diagnostic(
        {
            "object_base_pose_wxyz": (
                0.5049254913640708,
                0.054398029100811277,
                0.06746007804752319,
                0.7085140990798312,
                0.7055226844806688,
                -0.008467996947033096,
                -0.013183554755485272,
            ),
            "object_dims_xyz_m": (
                0.07423684120324518,
                0.15479133799672126,
                0.07423684120324518,
            ),
            "table_base_pose_wxyz": (
                0.44999998807907093,
                9.992006956928613e-17,
                -0.03499999046325686,
                0.7071067811865476,
                0.0,
                0.0,
                -0.7071067811865476,
            ),
            "table_dims_xyz_m": (1.2, 0.7, 0.05),
        },
        surface_sphere_radius_m=0.003,
    )

    assert diagnostic["object_bottom_z_base_m"] == pytest.approx(
        -0.010339077420,
        abs=1e-9,
    )
    assert diagnostic["table_top_z_base_m"] == pytest.approx(
        -0.009999990463,
        abs=1e-9,
    )
    assert diagnostic["signed_gap_m"] == pytest.approx(-0.000339086957, abs=1e-9)
    assert diagnostic["required_lift_to_positive_gap_m"] == pytest.approx(
        0.000339086957,
        abs=1e-9,
    )
    assert diagnostic["required_lift_to_sphere_clearance_m"] == pytest.approx(
        0.003339086957,
        abs=1e-9,
    )
    assert diagnostic["conservative_overlap"] is True
    assert diagnostic["within_surface_sphere_radius"] is True
    assert diagnostic["diagnostic_only"] is True


def test_attachment_provenance_includes_support_contact_geometry(
    tmp_path: Path,
) -> None:
    planner = FakeCarriedPlanner(FakeRobot(), tmp_path / "curobo.yml")

    provenance = planner.initialize_attachment(_state(), _scene())

    support = provenance["support_contact_geometry"]
    assert support["diagnostic_only"] is True
    assert support["signed_gap_m"] == pytest.approx(0.005, abs=1e-9)
    assert support["conservative_overlap"] is False
    assert support["within_surface_sphere_radius"] is False


def test_support_escape_plan_is_bounded_and_reenables_table(tmp_path: Path) -> None:
    planner = FakeCarriedPlanner(FakeRobot(), tmp_path / "curobo.yml")
    planner.initialize_attachment(_state(), _scene())

    result, diagnostic = planner.plan_support_escape(
        _state(),
        _goal(),
        PlanningContext("scene-1", "tf-1"),
        _scene(),
        obstacle_name="table",
        requested_escape_m=0.006,
        max_escape_m=0.01,
    )

    assert result.succeeded
    assert result.path is not None
    assert diagnostic["bounded"] is True
    assert diagnostic["requested_escape_m"] == pytest.approx(0.006)
    assert diagnostic["obstacle_disabled"] is True
    assert diagnostic["obstacle_reenabled"] is True
    assert diagnostic["start_state"]["valid"] is True
    assert diagnostic["plan_success"] is True
    assert result.path.metadata["support_escape_planning"]["obstacle_reenabled"] is True
    motion_gen = planner._motion_gen
    assert isinstance(motion_gen, FakeMotionGen)
    assert motion_gen.world_coll_checker.enabled["table"] is True
    assert motion_gen.world_coll_checker.calls[-2:] == [
        ("table", False),
        ("table", True),
    ]


def test_support_escape_plan_rejects_distance_above_bound(tmp_path: Path) -> None:
    planner = FakeCarriedPlanner(FakeRobot(), tmp_path / "curobo.yml")
    planner.initialize_attachment(_state(), _scene())

    result, diagnostic = planner.plan_support_escape(
        _state(),
        _goal(),
        PlanningContext("scene-1", "tf-1"),
        _scene(),
        obstacle_name="table",
        requested_escape_m=0.02,
        max_escape_m=0.01,
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code.value == "invalid_goal"
    assert diagnostic["bounded"] is False
    assert diagnostic["error"] == "escape_distance_out_of_bounds"
    motion_gen = planner._motion_gen
    assert isinstance(motion_gen, FakeMotionGen)
    assert motion_gen.world_coll_checker.calls == []


def test_carried_planner_fails_closed_without_aligned_vendor_velocity(
    tmp_path: Path,
) -> None:
    class MissingVelocityPlanner(FakeCarriedPlanner):
        def _plan_native(self, motion_gen, start_state, target_gripper_pose):
            del motion_gen, target_gripper_pose
            return {
                "status": "Success",
                "native_status": "Success",
                "native_diagnostics": {"valid_query": True},
                "position": [
                    start_state.positions,
                    (0.1, -0.1, 0.2, -0.2, 0.1, 0.0),
                ],
            }

    planner = MissingVelocityPlanner(FakeRobot(), tmp_path / "curobo.yml")
    planner.initialize_attachment(_state(), _scene())

    result = planner.plan(
        _state(),
        _goal(),
        PlanningContext("scene-1", "tf-1"),
        _scene(),
    )

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code.value == "internal_error"
    assert "missing aligned vendor velocity samples" in result.failure.message
    assert result.failure.details["native_status"] == "Success"
    assert result.failure.details["position_samples"] == 2


def test_carried_planner_prepends_zero_velocity_with_prepended_start(
    tmp_path: Path,
) -> None:
    class PrependedStartPlanner(FakeCarriedPlanner):
        def _plan_native(self, motion_gen, start_state, target_gripper_pose):
            del motion_gen, start_state, target_gripper_pose
            return {
                "status": "Success",
                "position": [
                    (0.05, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
                ],
                "velocity": [
                    (0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
                ],
            }

    planner = PrependedStartPlanner(FakeRobot(), tmp_path / "curobo.yml")
    planner.initialize_attachment(_state(), _scene())

    result = planner.plan(
        _state(),
        _goal(),
        PlanningContext("scene-1", "tf-1"),
        _scene(),
    )

    assert result.succeeded
    assert result.path is not None
    assert len(result.path.positions) == 3
    velocities = result.path.metadata["vendor_velocity_samples_rad_s"]
    assert velocities[0] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert velocities[1:] == (
        (0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def test_native_carried_planner_extracts_vendor_velocity_contract() -> None:
    import inspect

    native_source = inspect.getsource(
        PantheraRoboTwinCarriedObjectPlanner._plan_native
    )
    plan_source = inspect.getsource(PantheraRoboTwinCarriedObjectPlanner.plan)

    assert "native_result.interpolated_plan.position" in native_source
    assert "native_result.interpolated_plan.velocity" in native_source
    assert '"vendor_velocity_available": True' in plan_source
    assert '"vendor_velocity_samples_rad_s": velocities' in plan_source
