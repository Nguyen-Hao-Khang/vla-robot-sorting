# ADR 0001: RoboTwin with the Panthera-HT model as the simulator

- Status: Accepted
- Date: 2026-10-05

## Context

The project needs hundreds of pick-and-place demonstrations and hundreds of evaluation episodes before the real arm arrives. The simulator must include an accurate Panthera-HT model, render RGB cameras, and run on a laptop with a GTX 1660 Ti.

## Options

| Option | Advantages | Disadvantages |
|---|---|---|
| RoboTwin (SAPIEN, PhysX), Panthera-HT fork by HighTorque Robotics | Vendor-provided robot model; CuRobo planner integrated; used by the community to generate VLA data | Linux with an NVIDIA GPU only; sparse documentation |
| Isaac Sim / Isaac Lab | High-quality rendering and physics | Needs an RTX GPU with large VRAM; the Panthera-HT model would have to be built |
| MuJoCo | Light and fast | The robot model would have to be converted; less realistic rendering |

## Decision

Use Panthera-HT RoboTwin at commit `2aee31ad`, CuRobo `v0.7.8` and SAPIEN `3.0.0b1` ([`configs/repos.lock.yaml`](../../configs/repos.lock.yaml)).

## Consequences

- Simulation runs only on Ubuntu with an NVIDIA GPU. Data processing, training and writing happen on Windows or cloud GPUs.
- Project code reaches RoboTwin through `tools/twin/` and does not modify RoboTwin sources.
- Reach and collision limits of the simulated setup are listed in [architecture, section 5](../architecture.md#5-known-simulator-limits).
