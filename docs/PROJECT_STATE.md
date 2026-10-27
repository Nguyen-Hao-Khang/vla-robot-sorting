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

## 7. Schedule (deadline end of December 2026)

| Week | Dates | Engineering | Writing |
|---|---|---|---|
| T1+T2 | 27/10–2/11 | V2 cleanup; VLA bake-off and decision (31/10); E1 smoke on LIBERO; task spec; human-written command form; 20-demo pilot through the LeRobot pipeline; check RoboTwin code-gen with Panthera | 6-chapter skeleton; Ch.2 reading split |
| (spare) | 3–9/11 | Buffer week gained by the acceleration: start T3 early | Ch.2 |
| T3 | 10–16/11 | Generate demos + instructions; LeRobot dataset; collect human-written commands; fine-tune v0 | Ch.2, Ch.4 data |
| T4 | 17–23/11 | E2, E3; robot arrives: install, SDK, safety layer, cameras | Ch.3 |
| T5 | 24–30/11 | E4, E5; calibration, teleop. **Go/no-go 30/11** | Ch.4 |
| T6 | 1–7/12 | 30–50 real demos, sim + real fine-tune | Ch.1 |
| T7 | 8–14/12 | E6 on the real robot | Ch.5 sim |
| T8 | 15–21/12 | E7 if time allows; freeze results | Ch.5, Ch.6 |
| T9 | 22–31/12 | No new experiments | Final report |

**Go/no-go on 30/11.** All four conditions must hold:

- the robot runs through its SDK;
- the safety layer blocks out-of-workspace commands;
- the cameras are calibrated;
- teleoperation records demos.

If any condition fails, inform the advisor immediately, reduce E6 to a qualitative demo, and record the change as a limitation.

## 8. Report

The report has six chapters:

1. Introduction
2. Background and related work — the literature survey is redone from scratch, and every citation is checked against the original.
3. Hardware: Panthera-HT, control path, cameras and calibration, safety
4. Method
5. Experiments
6. Discussion and conclusion

It is a course project that continues into the graduation thesis, written to the standard of a research paper. No conference submission is planned.

## 9. Open items

- **VLA model choice.** SmolVLA vs Evo-1 bake-off, decision due 2026-10-31 (V2-14). See `docs/research/v2-vla-model-survey.md`.
- **TinyVLA language encoder.** The TA described it as ModernBERT, but the paper (arXiv 2409.12514) describes Pythia-based VLMs. Confirm with the TA.
