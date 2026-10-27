"""Source-shaped FCL boundary tests; these are not native collision evidence."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vla_robot_sorting.planning.manipulation_recovery import RecoveryContractError
from vla_robot_sorting.planning.native_plan_diagnostics import (
    MPLIB_PLANNER_SHA256,
    audit_native_plan_trace,
    native_distance_request,
    single_witness_distance_record,
)


class FakeFCL:
    GJKSolverType = SimpleNamespace(GST_INDEP=object())

    @staticmethod
    def DistanceRequest(**kwargs):
        return kwargs


def _result(distance, point):
    return SimpleNamespace(
        min_distance=distance,
        nearest_points=np.asarray(point, dtype=np.float64),
    )


def test_native_request_uses_independent_solver_and_retains_signed_distance():
    request = native_distance_request(FakeFCL)
    assert request == {
        "enable_nearest_points": True,
        "enable_signed_distance": True,
        "distance_tolerance": 1e-6,
        "gjk_solver_type": FakeFCL.GJKSolverType.GST_INDEP,
    }


def test_one_fcl_witness_is_not_fabricated_into_a_point_pair():
    record = single_witness_distance_record(
        "link2",
        _result(
            0.41922242596036524,
            [-0.005214438386777541, -0.3936048476687993, 0.8701161597532359],
        ),
    )
    assert record["signed_distance_m"] == 0.41922242596036524
    assert record["nearest_point_world_m"] == [
        -0.005214438386777541,
        -0.3936048476687993,
        0.8701161597532359,
    ]
    assert record["witness_contract"] == "one_solver_point_on_first_object"
    assert "nearest_points_world_m" not in record


@pytest.mark.parametrize(
    "distance,point",
    [
        (float("nan"), [0, 0, 0]),
        (float("inf"), [0, 0, 0]),
        (101.0, [0, 0, 0]),
        (0.1, [0, 0]),
        (0.1, [[0, 0, 0], [0, 0, 0]]),
        (0.1, [0, float("nan"), 0]),
    ],
)
def test_bad_native_distance_or_witness_fails_closed(distance, point):
    with pytest.raises(RecoveryContractError):
        single_witness_distance_record("link2", _result(distance, point))


def _trace():
    scope = {"target": "080_pillbottle_23"}
    scope_sha = hashlib.sha256(
        json.dumps(scope, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    sample = {
        "sample_index": 0,
        "time_from_start_s": 0.0,
        "qpos_rad": [0.0] * 6,
        "full_qpos": [0.0] * 8,
        "fk_link_poses_world_wxyz": {"tool_link": [0, 0, 0, 1, 0, 0, 0]},
        "collision_pairs": [],
        "target_link_distances": [
            {
                "robot_collision_object": "link2",
                "signed_distance_m": 0.1,
                "nearest_point_world_m": [0, 0, 0],
                "witness_contract": "one_solver_point_on_first_object",
            }
        ],
        "target_signed_distance_m": 0.1,
        "scope_sha256": scope_sha,
    }
    return {
        "schema": "s1m.plan-samples.v2",
        "source_kind": "native_planner_geometry",
        "samples": [sample],
        "waypoint_count": 1,
        "physics_steps_during_collection": 0,
        "execution_status": "not_executed",
        "controller_eligible": False,
        "native_acceptance": False,
        "provenance": {
            "versions": {"mplib": "0.2.1", "sapien": "3.0.0b1"},
            "mplib_planner_sha256": MPLIB_PLANNER_SHA256,
            "scope": scope,
            "scope_sha256": scope_sha,
            "distance_selftest": {
                "separated_m": 0.1,
                "overlapping_m": -0.05,
            },
            "fk_selftest": {"passed": True},
            "distance_query": {
                "solver": "GST_INDEP",
                "distance_tolerance_m": 1e-6,
                "signed_distance": True,
                "witness_contract": "one_solver_point_on_first_object",
            },
        },
    }


def test_auditor_accepts_structural_native_v2_without_motion_authorization():
    result = audit_native_plan_trace(_trace())
    assert result["passed"] is True
    assert result["native_acceptance"] is False
    assert result["controller_eligible"] is False


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["samples"][0]["target_link_distances"][0].update(
            nearest_points_world_m=[[0, 0, 0], [0, 0, 0]]
        ),
        lambda d: d["samples"][0]["target_link_distances"][0].update(
            witness_contract="two_points_claimed"
        ),
        lambda d: d["provenance"]["distance_query"].update(solver="GST_LIBCCD"),
        lambda d: d["samples"][0]["target_link_distances"][0].update(
            nearest_point_world_m=[float("nan"), 0, 0]
        ),
        lambda d: d.update(schema="s1m.plan-samples.v1"),
    ],
)
def test_auditor_rejects_ambiguous_or_unverified_witness_contract(change):
    record = copy.deepcopy(_trace())
    change(record)
    assert audit_native_plan_trace(record)["passed"] is False
