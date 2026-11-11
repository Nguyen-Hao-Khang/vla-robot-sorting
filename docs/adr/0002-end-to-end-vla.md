# ADR 0002: An end-to-end VLA as the main system

- Status: Accepted
- Date: 2026-10-27

## Context

The project goal is to sort objects from language commands with a VLA model on a robot arm. The first project phase built a modular pipeline (perception, command parsing, grasp selection, motion planning) that works in simulation but contains no VLA.

## Options

| Option | Description |
|---|---|
| A. End-to-end VLA | One model maps images, command and joint state directly to joint targets |
| B. Vision-language planner + VLA | A vision-language model splits the command into sub-steps; a VLA executes each step |
| C. Modular pipeline | Hand-built perception, parsing and motion planning |

## Decision

Option A. Options B and C are out of scope for this project period.

- A meets the project goal directly and runs locally, without an online service at run time.
- B adds a large model and an interface that needs its own evaluation, which the schedule cannot absorb.
- C uses no VLA. Its motion planning and grasp selection live on as the scripted expert that generates demonstrations (ADR 0003). The full modular pipeline is preserved at the tag `sprint1-archive`.

## Consequences

- System quality depends mostly on the training data, so scene randomisation and command diversity are requirements.
- The model gives no safety guarantee, so the real arm needs a separate safety layer (ADR 0004).
