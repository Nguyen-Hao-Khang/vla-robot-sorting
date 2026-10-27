# ADR-0001 — Project-owned interface boundaries

**Status:** Accepted
**Date:** 2026-10-02

## Context

Sprint 1 integrates rapidly changing third-party components: Panthera/RoboTwin, VLM providers, grounding/grasp stacks, planners and trajectory tools. The project must remain testable on Windows even when canonical simulation is Linux-only.

## Decision

All third-party implementations sit behind project-owned contracts. The initial contract package is `src/vla_robot_sorting/contracts`.

The semantic layer may emit validated task intent but may not emit low-level motor commands. Full 6-DoF geometric targets, motion planning, safety validation and execution remain separate interfaces.

## Consequences

- Windows unit tests can validate orchestration without RoboTwin/CUDA.
- Third-party implementations can be replaced without rewriting the full stack.
- Adapter code must translate explicit coordinate-frame and action conventions.
- Contract changes require deliberate review because they affect multiple sub-sprints.

## Evidence

See `docs/DECISIONS.md` (AD-02 through AD-06).
