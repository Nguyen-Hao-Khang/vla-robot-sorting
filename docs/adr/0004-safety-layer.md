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

Implemented today: items 1 and 3 (`joint_limits.py`). Items 2 and 4 are work item 3.2 in the [project plan](../project-plan.md).

## Consequences

- Nobody relaxes a threshold to make a run pass ([contributing](../../CONTRIBUTING.md#3-code)).
- Every change to the layer comes with a test for the rejected case.
- Safety interventions are an evaluation metric ([evaluation protocol](../evaluation-protocol.md)).
