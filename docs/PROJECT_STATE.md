# Project State (V2) — single source of truth

**Status:** provisionally approved by the product owner on 2026-10-27. This file wins over every other document in the repository.
**Supersedes:** the Sprint 1 scope and research framing (see [`HISTORY.md`](HISTORY.md) for what Sprint 1 and the digital twin delivered).
**Unchanged:** the Sprint 1 evidence manifests in `experiments/accepted/` remain valid for what they claim.
**Decision ledger:** [`docs/DECISIONS.md`](DECISIONS.md).

## 1. Why V2 exists

The project topic requires the team to *deploy and fine-tune a VLA model* (requirement 3) and to build a sorting application that integrates a VLA with the arm (requirement 4).

Through Sprint 1 and Twin v1–v15 the project drifted. It built a modular pipeline instead: a cloud VLM, geometric grasping, CuRobo and an executor. **No VLA model was trained.** The week-1 thesis treated an end-to-end VLA only as a baseline.

The teaching assistant's feedback (2026-10-22) made three points:

- VLAs must be retrained for each environment.
- RoboTwin can generate datasets with AI.
- The novelty of the project was not yet visible.

V2 returns to the topic. It keeps the parts of the old plan that still fit.

## 2. Working thesis

> A small VLA, fine-tuned on data produced by an MLLM together with a scripted simulation expert, can sort objects from Vietnamese and English commands at both the attribute level and the semantic-category level. It can then be transferred to the physical Panthera-HT arm with few real demonstrations, within a low compute budget (6 GB GPU at run time). Placing a VLM in front of the same VLA lets new categories be added without retraining, at the cost of depending on the VLM at run time.

The constraint is **compute**, not robot price. Panthera-HT is not a low-cost arm.

## 3. System

| Variant | Path | Role |
|---|---|---|
| **Flat end-to-end VLA (main)** | camera + command → fine-tuned VLA → joint actions → safety layer → robot | Final product; runs locally, no API at run time |
| VLM + VLA | command → VLM rewrites to an atomic instruction → same VLA → safety layer | Tests C1/C2; fallback for hard semantic categories |
| Modular pipeline / Twin v15 | Gemini plan + geometric grasp + CuRobo | **Demonstration expert** (data generation), **safety layer** reference, **upper-bound** reference. Not the main execution path |

## 4. Research questions and retained claims

| ID | Question | Claim |
|---|---|---|
| RQ1 | Success of the fine-tuned small VLA on attribute and category sorting, with human-written VI/EN/code-switched commands | — |
| RQ2 | Which instruction-data strategy (fixed template, machine translation, MLLM paraphrase, code-switch) makes the VLA most robust to unseen Vietnamese commands | — |
| RQ3 | Flat VLA vs VLM + VLA as the number of demos per category N grows | C1 (revised) |
| RQ4 | Cost (demos, GPU hours, changed lines) and success when adding an unseen category | C2 |
| RQ5 | Performance retained from RoboTwin to the real Panthera-HT with 0 and 30–50 real demos | C3 (revised) |

- C4 (error attribution) is **no longer a claim**. The failing layer is still logged, and VI/EN/code-switch results are still reported separately.
- There are **no hard % thresholds**. Report each result with ≥ 30 trials per condition and a Wilson 95% confidence interval.

## 5. Experiments

| ID | Experiment | Answers |
|---|---|---|
| E1 | Reproduce the chosen VLA on LIBERO | Training-pipeline sanity only |
| E2 | Flat VLA in RoboTwin, attribute/category × VI/EN/code-switch, human-written held-out commands | RQ1 |
| E3 | Same VLA, four instruction-data strategies | RQ2 |
| E4 | Flat vs VLM + VLA at three N levels | RQ3 |
| E5 | Held-out category: flat (0 and +k demos) vs VLM + VLA (prompt edit) | RQ4 |
| E6 | Sim-only / real-only (30–50 demos) / sim + real on the physical arm | RQ5 |
| E7 | (optional) camera pose, lighting and distractor shifts in simulation | OOD robustness |

## 6. Data

The data sources are combined:

- Twin v15 as the scripted expert. It is reachability-aware (see `HISTORY.md`).
- RoboTwin's AI code-generation pipeline, where it supports single-arm Panthera.
- An MLLM (Gemini, called online during development) generating VI/EN/code-switched instructions.

The datasets are stored in LeRobot format. Target size: 500–1,000 simulated demos and 30–50 real teleoperated demos.

## 7. Milestones (deadline end of December 2026)

The team plans by milestone. Each task is a GitHub issue attached to one milestone. The dates are targets.

| ID | Milestone | Target | Exit criteria |
|---|---|---|---|
| M0 | Project setup and model choice | 13/11 | Members onboarded; CI green; VLA chosen by bake-off (V2-14); task spec merged; 20-demo pilot runs through recording, LeRobot conversion, fine-tuning and evaluation in the twin |
| M1 | Simulation data and VLA, attribute level | 27/11 | 500 to 1,000 simulated demos; fine-tuned VLA and ACT baseline; E2 attribute-level success rates with Wilson 95% intervals |
| M2 | Hardware bring-up and robot go/no-go | 30/11 | Robot runs through its SDK; safety layer blocks out-of-workspace commands; cameras calibrated; teleoperation records demos |
| M3 | VLA on the real robot | 11/12 | 30 to 50 real demos; sim + real fine-tune; the policy completes sorting runs on the physical arm through the safety layer |
| M4 | Experiments complete | 20/12 | E6 on the real robot with at least 30 trials per condition; E3 and the human-written command set if in scope; results frozen and traceable |
| M5 | Report submitted | 31/12 | Report in `report/` complete and cross-checked against evidence; release `v1.0` tagged |

**Scope priority.**

- **Must:** a fine-tuned VLA sorting by attribute (colour, shape) from Vietnamese and English commands, in simulation and on the real arm, with at least 30 trials per condition, the safety layer, the hardware chapter and the full report. This covers all five topic requirements.
- **Should:** semantic-category sorting, the human-written command set, the ACT baseline, and E3.
- **Could:** the VLM + VLA variant (E4, E5) and OOD tests (E7).

**If M2 fails** (robot not ready on 30/11), inform the advisor immediately. The team then moves effort to Should and Could items in simulation, reduces E6 to a qualitative demo, and records the change as a limitation.

## 8. Report

The report lives in `report/` (LaTeX, Vietnamese) and follows the structure of a research paper:

1. Introduction
2. Background and related work. The literature survey is redone from scratch, and the team checks every citation against the original.
3. Hardware: Panthera-HT, control path, cameras and calibration, safety
4. Method
5. Experiments
6. Discussion and conclusion

This semester's work is a course project that continues into the graduation thesis. No conference submission is planned.

## 9. Open items

- **VLA model choice.** SmolVLA vs Evo-1 bake-off, decided in M0 (V2-14). See `docs/research/v2-vla-model-survey.md`.
- **TinyVLA language encoder.** The TA described it as ModernBERT, but the paper (arXiv 2409.12514) describes Pythia-based VLMs. Confirm with the TA.
