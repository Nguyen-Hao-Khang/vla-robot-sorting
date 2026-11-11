# ADR 0003: Training data from a simulated expert and a language model

- Status: Accepted
- Date: 2026-10-27

## Context

Fine-tuning a VLA needs hundreds of demonstrations with images, joint states, actions and commands. The real arm becomes available at the end of November, and a teleoperated demonstration takes 1–2 minutes.

## Options

| Option | Advantages | Disadvantages |
|---|---|---|
| Real demonstrations only | No simulation-to-reality gap | Not enough time for hundreds of episodes |
| Simulated expert + language-model commands, plus a few real demonstrations | Large volume, automatic labels, reproducible from seeds | Simulation-to-reality gap; the expert uses simulator ground truth |
| Public datasets (Open X-Embodiment, LIBERO) | Available now | Different robots and tasks; no Vietnamese |

## Decision

The second option:

- A scripted expert (CuRobo motion planning, object poses from the simulator) performs the task; `twin_record.py` records it.
- A large language model writes several Vietnamese and English phrasings per task.
- 30–50 real demonstrations are added once the arm arrives.

## Consequences

- Ground-truth object poses serve only to generate demonstrations. The VLA receives images, the command and the joint state.
- People write the test commands, separately from the training commands, to measure understanding of unseen phrasings.
- Procedure: [data and training](../data-and-training.md).
