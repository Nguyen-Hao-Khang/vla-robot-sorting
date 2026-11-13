# ADR 0004: A mandatory safety layer between the policy and the real arm

- Status: Accepted
- Date: 2026-10-27

## Context

A VLA can output any action, including joint targets beyond the limits, positions off the table, or fast motions. The Panthera-HT carries 3.5 kg at up to 860 mm, enough to damage equipment or injure people.

## Decision

Every action sent to the real arm passes the safety layer in `src/vla_robot_sorting/safety/`. The layer:

1. rejects joint positions outside the manufacturer limits;
2. rejects gripper positions outside a configured workspace box;
3. limits joint velocity, acceleration and jerk;
4. stops the arm when the policy stays silent beyond a timeout;
5. leaves a hardware emergency stop in reach at all times.

The same layer runs in simulation, so rejection counts are comparable across both settings.

Implementation: `action_filter.py` checks every action at run time (items 1-4) and `joint_limits.py` validates planned trajectories (items 1 and 3). Thresholds live in [`configs/safety.json`](../../configs/safety.json) with two profiles:

- `simulation` admits the scripted expert's motion (velocity up to 1.41 rad/s, acceleration 99th percentile up to 2.3 rad/s², tool point inside the box), measured on the first 80 simulated episodes, so the filter does not change demonstrations that the policy imitates;
- `real` starts slower (0.5 rad/s, 2 rad/s²), with a smaller workspace box, a 0.5 s watchdog and a stop after 3 rejected actions in a row. It is revisited after the first low-speed tests on the arm.

Per action the filter rejects malformed or out-of-limit targets and jumps larger than `reject_step`, clamps targets just past the limit band and the gripper command, limits joint velocity and acceleration, and rejects targets whose tool point leaves the workspace box or whose joints go below `min_link_z` (forward kinematics from the URDF, `panthera.py`). A rejected action becomes "hold position"; repeated rejections, the watchdog or `emergency_stop` latch a stop that only `reset` clears.

## Consequences

- Nobody relaxes a threshold to make a run pass ([contributing](../../CONTRIBUTING.md#3-code)).
- Every change to the layer comes with a test for the rejected case.
- Safety interventions are an evaluation metric ([evaluation protocol](../evaluation-protocol.md)).
