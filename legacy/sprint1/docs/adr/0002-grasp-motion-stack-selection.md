# ADR 0002: Select a project-owned full-6-DoF grasp and motion stack

**Status:** Accepted (2026-10-27) — all acceptance spikes passed natively (B1–B4 on 2026-10-06, B5/B7 on 2026-10-07). Timing layer as built: CuRobo interpolation + physics-aligned time dilation, not TOPPRA. **Scope narrowed by V2** (`docs/PROJECT_STATE.md`): this stack is the scripted demonstration expert and safety baseline, not the main execution path.

**Original status:** Proposed — Pending Validation

**Date:** 2026-10-02

**Deciders:** Sprint 1 robotics simulation/infrastructure team
**Related:** ADR 0001, W-A compatibility audit, W-B grasp-to-motion study

## Context

Sprint 1 requires object-specific full 6-DoF grasp poses for Panthera-HT in RoboTwin, smooth continuous collision-aware motion, and receding-horizon execution. The target Linux gate is an Ubuntu 22.04 laptop with a GTX 1660 Ti Max-Q and 6 GB VRAM. Semantic/VLM reasoning is deliberately slower and remote; local geometric/control decisions must remain available on the simulation host.

The project must avoid coupling its interfaces to simulator or planner implementation classes, preserve object/frame/transform provenance, and retain exact experiment traceability. The companion W-B study distinguishes verified upstream facts from target-machine inferences.

## Proposed decision

1. Make a project-owned geometry-first parallel-jaw GraspGenerator and GraspRanker the initial canonical path. It returns object-specific SE(3) candidates, pre-grasp poses, approach vectors, jaw width, clearance and provenance.
2. Permit explicitly tagged simulator ground-truth object geometry only as a reproducible simulation-control-oracle condition. It is not an end-to-end perception claim.
3. Use a CuRobo v0.7.8 adapter as the preferred IK and motion planning implementation only after spikes B1 and B2 of W-B pass on the target Ubuntu machine.
4. Use MPlib 0.2.1 as the executable fallback adapter if CuRobo fails the pinned-environment, memory, collision, or latency gates. Recheck selected paths in the current RoboTwin/SAPIEN collision scene.
5. Use TOPPRA 0.6.9 for velocity- and acceleration-constrained time parameterization, then apply a project-owned jerk measurement gate and duration dilation/C2 local refit when necessary.
6. Run accepted trajectories as short ExecutionSegments. Validate robot state, scene version, transform version and tracking error at each checkpoint; reject stale segments and replan rather than executing long open-loop trajectories.
7. Keep GraspNet-baseline, AnyGrasp and Contact-GraspNet outside the canonical first runtime. They can enter only as separately recorded comparators after license, compatibility and memory evidence.

## Consequences

### Positive

- The first stack preserves full 6-DoF grasping without depending on unavailable learned-model weights or unmeasured GPU headroom.
- A failure of CuRobo does not change the canonical robot or simulator, and does not discard measurable collision-aware planning.
- Planner and generator implementations can change behind stable project contracts.
- Ground-truth simulation controls are useful and honest because their privileged source is explicit in the manifest.

### Costs and risks

- Geometry-first grasp quality must be benchmarked rather than assumed.
- CuRobo v0.7.8 needs empirical validation against the target driver, CUDA/PyTorch environment, renderer and 6 GB memory ceiling.
- TOPPRA alone does not guarantee jerk bounds; the wrapper must measure and gate them.
- MPlib 0.2.1 packaging/license evidence needs recording before a dependency lock or redistribution.

## Validation required before acceptance

ADR status changes only after these W-B spikes have attached artifacts:

- B1 CuRobo import, fixed-pose IK/plan and warm-up evidence;
- B2 renderer, two-camera and CuRobo coexistence telemetry;
- B3 MPlib fallback fixed-pose collision-cleared timed plan;
- B4 timing, jerk and segment-splice results;
- B5 full-6-DoF geometry-first candidate and collision/IK result; and
- B7 stale-segment rejection and replanning recovery.

The exact evidence must include locked integration commit, vendored RoboTwin tree hash, configuration hash, package environment export, target GPU telemetry, seeds and raw diagnostics.

## Alternatives considered

**AnyGrasp as primary:** rejected pending technical license/feature-ID/weights access and 6 GB coexistence evidence.

**GraspNet-baseline as primary:** deferred because its upstream setup is an older PyTorch 1.6/custom-extension stack with no target-laptop memory assurance.

**Contact-GraspNet as primary:** rejected for the target laptop because its upstream README states a one-NVIDIA-GPU inference expectation of at least 8 GB.

**Another simulator:** rejected. It would violate the canonical Panthera-HT plus RoboTwin scope instead of resolving an adapter/resource issue.

## References

- [W-B 6-DoF grasp-to-motion stack study](../research/W-B-6dof-grasp-motion-stack-study.md)
- [S1.1 repository lock](../../configs/s1.1/repos.lock.yaml)
- [ADR 0001](0001-project-owned-interface-boundaries.md)
