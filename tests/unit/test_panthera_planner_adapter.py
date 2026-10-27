import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace

import pytest

import vla_robot_sorting.planning.panthera_robotwin as panthera_robotwin
from vla_robot_sorting.contracts import (
    JointState,
    MotionFailureCode,
    MotionGoal,
    PlanningContext,
    Pose6D,
)
from vla_robot_sorting.planning import (
    PantheraRoboTwinMotionPlanner,
    PantheraRoboTwinMPlibAcquisitionPlanner,
    PantheraRoboTwinMPlibPlanner,
    PantheraRoboTwinMPlibScrewPlanner,
    pose6d_to_robotwin_target,
)
from vla_robot_sorting.planning.panthera_robotwin import (
    curobo_table_world_from_scene,
)
from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES


def _goal(frame_id: str = "world") -> MotionGoal:
    return MotionGoal(
        pose=Pose6D(
            position_xyz=(0.4, -0.1, 0.3),
            orientation_xyzw=(0.1, 0.2, 0.3, 0.92736185),
            frame_id=frame_id,
        )
    )


def _context() -> PlanningContext:
    return PlanningContext(scene_revision="scene-3", transform_revision="tf-9")


def _state(value: float = 0.0) -> JointState:
    return JointState(
        joint_names=PANTHERA_JOINT_NAMES,
        positions=(value,) * 6,
    )


class FakeRobot:
    def __init__(self) -> None:
        self.real_state = [0.0] * 6 + [0.5]
        self.targets: list[tuple[float, ...]] = []
        self.result: object = {
            "status": "Success",
            "position": [[0.0] * 6, [0.05] * 6],
            "velocity": [[0.0] * 6, [0.1] * 6],
        }
        self.raise_error = False

    def get_left_arm_real_jointState(self) -> list[float]:
        return self.real_state

    def left_plan_path(self, target: tuple[float, ...]) -> object:
        self.targets.append(target)
        if self.raise_error:
            raise RuntimeError("planner unavailable")
        return self.result


def _clock(values: list[float]):
    iterator: Iterator[float] = iter(values)
    return lambda: next(iterator)


def test_pose_conversion_reorders_xyzw_to_robotwin_wxyz() -> None:
    target = pose6d_to_robotwin_target(_goal())

    assert target[:3] == pytest.approx((0.4, -0.1, 0.3))
    assert target[3:] == pytest.approx((0.92736185, 0.1, 0.2, 0.3))


def test_pose_conversion_rejects_non_world_frame() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        pose6d_to_robotwin_target(_goal("robot_base"))


def test_planner_success_preserves_revisions_and_measured_start() -> None:
    robot = FakeRobot()
    planner = PantheraRoboTwinMotionPlanner(
        robot,
        clock=_clock([10.0, 10.02]),
    )

    result = planner.plan(_state(), _goal(), _context(), scene={})

    assert result.succeeded
    assert result.path is not None
    assert result.planning_time_s == pytest.approx(0.02)
    assert result.path.positions[0] == (0.0,) * 6
    assert result.path.positions[-1] == (0.05,) * 6
    assert result.path.metadata["scene_revision"] == "scene-3"
    assert result.path.metadata["transform_revision"] == "tf-9"
    assert result.path.metadata["vendor_velocity_available"] is True
    assert result.path.metadata["vendor_interpolation_dt_s"] == pytest.approx(1.0 / 250.0)
    assert result.path.metadata["vendor_velocity_samples_rad_s"] == (
        (0.0,) * 6,
        (0.1,) * 6,
    )
    assert robot.targets[0][3:] == pytest.approx((0.92736185, 0.1, 0.2, 0.3))


def test_planner_rejects_stale_start_before_vendor_call() -> None:
    robot = FakeRobot()
    planner = PantheraRoboTwinMotionPlanner(
        robot,
        start_state_tolerance_rad=0.01,
        clock=_clock([1.0, 1.01]),
    )

    result = planner.plan(_state(0.2), _goal(), _context(), scene={})

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is MotionFailureCode.EXECUTION_STALE
    assert robot.targets == []


def test_planner_maps_vendor_failure_to_typed_failure() -> None:
    robot = FakeRobot()
    robot.result = {"status": "Fail"}
    planner = PantheraRoboTwinMotionPlanner(robot, clock=_clock([1.0, 1.01]))

    result = planner.plan(_state(), _goal(), _context(), scene={})

    assert result.failure is not None
    assert result.failure.code is MotionFailureCode.OPTIMIZATION_FAILED


def test_planner_maps_vendor_exception_to_backend_failure() -> None:
    robot = FakeRobot()
    robot.raise_error = True
    planner = PantheraRoboTwinMotionPlanner(robot, clock=_clock([1.0, 1.01]))

    result = planner.plan(_state(), _goal(), _context(), scene={})

    assert result.failure is not None
    assert result.failure.code is MotionFailureCode.BACKEND_UNAVAILABLE


class FakeJoint:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class FakeEntity:
    def __init__(self) -> None:
        self._names = (*PANTHERA_JOINT_NAMES, "gripper_L_joint", "gripper_R_joint")

    def get_active_joints(self) -> list[FakeJoint]:
        return [FakeJoint(name) for name in self._names]

    def get_qpos(self) -> list[float]:
        return [0.0] * len(self._names)


class FakeMPlibBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.screw_calls: list[dict[str, object]] = []
        self.screw_result: object = {
            "status": "Success",
            "position": [
                [0.0] * 8,
                [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.01, 0.02],
            ],
            "velocity": [
                [0.0] * 8,
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0],
            ],
        }

    def plan_pose(self, qpos, target_pose, **kwargs):
        self.calls.append({"qpos": qpos, "target_pose": target_pose, **kwargs})
        return {
            "status": "Success",
            "position": [
                [0.0] * 8,
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01, 0.02],
            ],
            "velocity": [
                [0.0] * 8,
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.0],
            ],
        }

    def plan_screw(self, qpos, target_pose, **kwargs):
        self.screw_calls.append({"qpos": qpos, "target_pose": target_pose, **kwargs})
        return self.screw_result


class FakeMPlibRobot(FakeRobot):
    def __init__(self) -> None:
        super().__init__()
        self.left_entity = FakeEntity()
        self.left_mplib_planner = FakeMPlibBackend()
        self.transformed_targets: list[tuple[float, ...]] = []

    def _trans_from_gripper_to_endlink(
        self,
        target: tuple[float, ...],
        *,
        arm_tag: str,
    ) -> tuple[str, tuple[float, ...]]:
        assert arm_tag == "left"
        self.transformed_targets.append(target)
        return ("end-link-pose", target)


def test_mplib_fallback_extracts_only_panthera_arm_joints() -> None:
    robot = FakeMPlibRobot()
    planner = PantheraRoboTwinMPlibPlanner(
        robot,
        clock=_clock([2.0, 2.03]),
    )

    result = planner.plan(_state(), _goal(), _context(), scene={})

    assert result.succeeded
    assert result.path is not None
    assert result.path.positions[-1] == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    assert result.path.metadata["fallback_backend"] is True
    assert result.path.metadata["vendor_velocity_samples_rad_s"][-1] == pytest.approx(
        (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    )
    assert robot.left_mplib_planner.calls[0]["try_times"] == 10
    assert robot.left_mplib_planner.calls[0]["log"] is False


def test_curobo_table_world_sync_transforms_runtime_table_to_base_frame() -> None:
    half_sqrt = 2.0**-0.5
    robot = SimpleNamespace(
        left_entity_origion_pose=SimpleNamespace(
            p=(0.0, -0.45, 0.75),
            q=(half_sqrt, 0.0, 0.0, half_sqrt),
        )
    )
    scene = SimpleNamespace(
        metadata={
            "curobo_table_collision": {
                "source": "pinned RoboTwin create_table_and_wall tabletop collision",
                "frame_id": "world",
                "center_xyz_m": (0.0, 0.0, 0.715),
                "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
                "dims_xyz_m": (1.2, 0.7, 0.05),
            }
        }
    )

    prepared = curobo_table_world_from_scene(robot, scene)

    assert prepared is not None
    world, provenance = prepared
    table = world["cuboid"]["table"]
    assert table["dims"] == pytest.approx([1.2, 0.7, 0.05])
    assert table["pose"][:3] == pytest.approx([0.45, 0.0, -0.035], abs=1e-9)
    assert table["pose"][3:] == pytest.approx(
        [half_sqrt, 0.0, 0.0, -half_sqrt],
        abs=1e-9,
    )
    assert provenance["table_base_pose_wxyz"] == pytest.approx(table["pose"])


def test_curobo_table_world_sync_rejects_non_world_metadata() -> None:
    robot = SimpleNamespace(
        left_entity_origion_pose=SimpleNamespace(
            p=(0.0, -0.45, 0.75),
            q=(1.0, 0.0, 0.0, 0.0),
        )
    )
    scene = SimpleNamespace(
        metadata={
            "curobo_table_collision": {
                "source": "synthetic",
                "frame_id": "robot_base",
                "center_xyz_m": (0.0, 0.0, 0.715),
                "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
                "dims_xyz_m": (1.2, 0.7, 0.05),
            }
        }
    )

    with pytest.raises(ValueError, match="world frame"):
        curobo_table_world_from_scene(robot, scene)


class FakeMotionGen:
    def __init__(self) -> None:
        self.worlds: list[object] = []

    def update_world(self, world: object) -> None:
        self.worlds.append(world)


def test_curobo_world_sync_updates_both_motion_generators_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    half_sqrt = 2.0**-0.5

    class FakeWorldConfig:
        @staticmethod
        def from_dict(value):
            return {"parsed": value}

    curobo_module = ModuleType("curobo")
    geom_module = ModuleType("curobo.geom")
    types_module = ModuleType("curobo.geom.types")
    types_module.WorldConfig = FakeWorldConfig
    monkeypatch.setitem(sys.modules, "curobo", curobo_module)
    monkeypatch.setitem(sys.modules, "curobo.geom", geom_module)
    monkeypatch.setitem(sys.modules, "curobo.geom.types", types_module)

    motion_gen = FakeMotionGen()
    motion_gen_batch = FakeMotionGen()
    planner = SimpleNamespace(
        motion_gen=motion_gen,
        motion_gen_batch=motion_gen_batch,
    )
    robot = SimpleNamespace(
        left_planner=planner,
        left_entity_origion_pose=SimpleNamespace(
            p=(0.0, -0.45, 0.75),
            q=(half_sqrt, 0.0, 0.0, half_sqrt),
        ),
    )
    scene = SimpleNamespace(
        metadata={
            "curobo_table_collision": {
                "source": "runtime tabletop",
                "frame_id": "world",
                "center_xyz_m": (0.0, 0.0, 0.715),
                "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
                "dims_xyz_m": (1.2, 0.7, 0.05),
            }
        }
    )

    first = panthera_robotwin._sync_curobo_world(robot, scene)
    second = panthera_robotwin._sync_curobo_world(robot, scene)

    assert first is not None
    assert first["synchronized"] is True
    assert first["cache_hit"] is False
    assert first["updated_motion_generators"] == ("motion_gen", "motion_gen_batch")
    assert len(motion_gen.worlds) == 1
    assert len(motion_gen_batch.worlds) == 1
    parsed = motion_gen.worlds[0]
    assert isinstance(parsed, dict)
    pose = parsed["parsed"]["cuboid"]["table"]["pose"]
    assert pose[:3] == pytest.approx([0.45, 0.0, -0.035], abs=1e-9)

    assert second is not None
    assert second["synchronized"] is True
    assert second["cache_hit"] is True
    assert second["updated_motion_generators"] == ()
    assert len(motion_gen.worlds) == 1
    assert len(motion_gen_batch.worlds) == 1


def test_mplib_screw_acquisition_uses_target_aware_wrapper() -> None:
    robot = FakeMPlibRobot()
    planner = PantheraRoboTwinMPlibScrewPlanner(
        robot,
        clock=_clock([3.0, 3.02]),
    )

    result = planner.plan(_state(), _goal(), _context(), scene={})

    assert result.succeeded
    assert result.path is not None
    assert result.planner_id == "robotwin-mplib-screw"
    assert result.path.positions[-1] == pytest.approx(
        (0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
    )
    assert result.path.metadata["acquisition_strategy"] == "mplib_screw_acquisition"
    assert result.path.metadata["target_aware_planning_world_required"] is True
    assert result.path.metadata["independent_collision_recheck_required"] is True
    assert planner.last_provenance == {
        "strategy": "mplib_screw_acquisition",
        "wrapper_method": "left_mplib_planner.plan_screw",
        "target_aware_planning_world_required": True,
        "plan_success": True,
        "vendor_status": "Success",
        "selected_planner_id": "robotwin-mplib-screw",
        "straight_screw_motion": True,
        "independent_collision_recheck_required": True,
    }
    call = robot.left_mplib_planner.screw_calls[0]
    assert call["arms_tag"] == "left"
    assert call["log"] is False
    assert call["target_pose"][0] == "end-link-pose"


def test_mplib_screw_acquisition_maps_collision_failure_to_optimization_failure() -> None:
    robot = FakeMPlibRobot()
    robot.left_mplib_planner.screw_result = {"status": "screw plan failed"}
    planner = PantheraRoboTwinMPlibScrewPlanner(
        robot,
        clock=_clock([4.0, 4.01]),
    )

    result = planner.plan(_state(), _goal(), _context(), scene={})

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is MotionFailureCode.OPTIMIZATION_FAILED
    assert planner.last_provenance is not None
    assert planner.last_provenance["vendor_status"] == "screw plan failed"
    assert planner.last_provenance["plan_success"] is False


def test_mplib_screw_acquisition_rejects_stale_start_before_vendor_call() -> None:
    robot = FakeMPlibRobot()
    planner = PantheraRoboTwinMPlibScrewPlanner(
        robot,
        start_state_tolerance_rad=0.01,
        clock=_clock([5.0, 5.01]),
    )

    result = planner.plan(_state(0.2), _goal(), _context(), scene={})

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is MotionFailureCode.EXECUTION_STALE
    assert robot.left_mplib_planner.screw_calls == []


def test_mplib_acquisition_uses_screw_without_fallback_when_primary_succeeds() -> None:
    robot = FakeMPlibRobot()
    planner = PantheraRoboTwinMPlibAcquisitionPlanner(robot)

    result = planner.plan(_state(), _goal(), _context(), scene={})

    assert result.succeeded
    assert result.planner_id == "robotwin-mplib-screw"
    assert len(robot.left_mplib_planner.screw_calls) == 1
    assert robot.left_mplib_planner.calls == []
    assert planner.last_provenance is not None
    assert planner.last_provenance["strategy"] == "mplib_target_aware_acquisition"
    assert planner.last_provenance["primary_success"] is True
    assert planner.last_provenance["fallback_attempted"] is False
    assert planner.last_provenance["selected_planner_id"] == "robotwin-mplib-screw"


def test_mplib_acquisition_falls_back_to_target_aware_rrt_on_screw_optimization_failure() -> None:
    robot = FakeMPlibRobot()
    robot.left_mplib_planner.screw_result = {
        "status": "screw plan failed",
    }
    planner = PantheraRoboTwinMPlibAcquisitionPlanner(robot)

    result = planner.plan(_state(), _goal(), _context(), scene={})

    assert result.succeeded
    assert result.planner_id == "robotwin-mplib-rrt"
    assert len(robot.left_mplib_planner.screw_calls) == 1
    assert len(robot.left_mplib_planner.calls) == 1
    assert planner.last_provenance is not None
    assert planner.last_provenance["primary_success"] is False
    assert planner.last_provenance["primary_vendor_status"] == "screw plan failed"
    assert planner.last_provenance["fallback_allowed"] is True
    assert planner.last_provenance["fallback_attempted"] is True
    assert planner.last_provenance["fallback_success"] is True
    assert planner.last_provenance["selected_planner_id"] == "robotwin-mplib-rrt"
    assert planner.last_provenance["plan_success"] is True


def test_mplib_acquisition_does_not_fallback_on_stale_start_state() -> None:
    robot = FakeMPlibRobot()
    planner = PantheraRoboTwinMPlibAcquisitionPlanner(robot)

    result = planner.plan(_state(0.2), _goal(), _context(), scene={})

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code is MotionFailureCode.EXECUTION_STALE
    assert robot.left_mplib_planner.screw_calls == []
    assert robot.left_mplib_planner.calls == []
    assert planner.last_provenance is not None
    assert planner.last_provenance["fallback_allowed"] is False
    assert planner.last_provenance["fallback_attempted"] is False
