# Project plan

Period: 10 November to 31 December 2026. Each work item maps to one GitHub issue created from the **Task** form. Assignments live on the issues.

## 1. Scope

| Level | Content |
|---|---|
| Required | A fine-tuned small VLA that sorts objects by colour and shape from Vietnamese and English commands, in RoboTwin and on the real Panthera-HT behind the safety layer. Quantitative evaluation with at least 30 episodes per condition. Course report. |
| Extension | Sorting by printed identity and semantic category (household items), in the simulated dataset from version 1; on the real arm if time permits. |
| Out of scope | A planner model in front of the VLA; deformable or stacked objects; multi-step commands. |

## 2. Phases

| Phase | Dates | Exit state |
|---|---|---|
| 1. Simulation data and model selection | 10 Nov – 23 Nov | Randomised scene generator, bilingual command generator and evaluation script exist; base VLA chosen; report chapters 1–2 drafted |
| 2. Training and simulated evaluation | 24 Nov – 7 Dec | Dataset of 500–1,000 simulated episodes; main model and ACT baseline trained; E1 and E2 results |
| 3. Real-robot integration | 24 Nov – 7 Dec | Arm controlled from the PC through the safety layer; cameras calibrated; 30–50 real demonstrations |
| 4. Real-robot experiments | 8 Dec – 20 Dec | E3 results, failure analysis, demo video; results frozen on 20 Dec |
| 5. Write-up and release | 14 Dec – 31 Dec | Report complete, presentation ready, release `v1.0` |

Phases 2 and 3 run in parallel: phase 2 uses the training and simulation machines, phase 3 uses the real arm.

## 3. Checkpoints

At each checkpoint the team checks the condition. If it fails, the fallback applies instead of moving the deadline.

| Date | Checkpoint | Condition | Fallback |
|---|---|---|---|
| 16 Nov | Chapters 1–2 to advisor | Drafts compile | Send the current draft with the missing parts listed |
| 23 Nov | Model and data generator fixed | ADR 0005 accepted; 50 demonstrations with bilingual commands | Default to SmolVLA and keep comparing in parallel |
| 30 Nov | Real arm operational | Arm moves through the safety layer; cameras mounted | Inform the advisor; reduce E3 to two conditions (simulation only, combined) |
| 7 Dec | Simulation results | E1 and E2 tables exist | Drop extension 4.5; give all GPU time to the main model |
| 20 Dec | Results frozen | Every table and figure regenerates from logs | No new experiments; list missing parts under limitations |
| 31 Dec | Submission | Release `v1.0` on GitHub | None |

## 4. Work items

### Phase 1. Simulation data and model selection

| ID | Area | Work item | Deliverable | Done when | Depends on | Start | Due |
|---|---|---|---|---|---|---|---|
| 1.1 | Management | Set up team workflow | Issues for phase 1 | Every phase 1 item has an issue and an owner | None | 10 Nov | 12 Nov |
| 1.2 | Data | Freeze the task specification | `docs/task-spec.md` | Merged with status "frozen" | None | 10 Nov | 13 Nov |
| 1.3 | Simulation | Randomise scenes during recording | `tools/twin/twin_record.py`, tests | 50 demonstrations regenerate identically from a seed | 1.2 | 12 Nov | 19 Nov |
| 1.4 | Data | Bilingual command generator | Script and command files | ≥ 30 commands per pair and language; tests pass | 1.2 | 12 Nov | 19 Nov |
| 1.5 | Data | Held-out test commands | `configs/test_commands_v1.csv` | 100–200 commands, separate from training data | 1.2 | 13 Nov | 23 Nov |
| 1.6 | Model | Training environment | Updated `docs/getting-started.md` | SmolVLA and Evo-1 each run 100 training steps | None | 10 Nov | 16 Nov |
| 1.7 | Model | Compare and select the base VLA | ADR 0005 with measurements | ADR 0005 accepted | 1.3, 1.6, 1.8 | 17 Nov | 23 Nov |
| 1.8 | Evaluation | Automated simulation evaluation script | Script, tests | Runs N seeded episodes for a policy or the expert, writes JSONL, reports metrics with Wilson intervals | 1.2 | 13 Nov | 21 Nov |
| 1.9 | Report | Chapter 1: introduction | Draft | Compiles, cites sources | None | 12 Nov | 15 Nov |
| 1.10 | Report | Chapter 2: background and related work | Draft and bibliography | Compiles; every citation has a DOI or arXiv ID | None | 12 Nov | 15 Nov |

### Phase 2. Training and simulated evaluation

| ID | Area | Work item | Deliverable | Done when | Depends on | Start | Due |
|---|---|---|---|---|---|---|---|
| 2.1 | Data | Simulated dataset v1 | `panthera_twin_v1` on the Hugging Face Hub | 500–1,000 successful episodes; dataset card with statistics | 1.3, 1.4 | 24 Nov | 28 Nov |
| 2.2 | Model | Fine-tune the main model | Checkpoint, config, training log | Checkpoint runs through `policy_bridge` in simulation | 1.7, 2.1 | 28 Nov | 3 Dec |
| 2.3 | Model | Train the ACT baseline | Checkpoint, config, training log | Checkpoint runs in simulation | 2.1 | 28 Nov | 3 Dec |
| 2.4 | Evaluation | E1: simulation | Logs, generated tables | ≥ 30 episodes per condition with intervals | 1.5, 1.8, 2.2 | 3 Dec | 7 Dec |
| 2.5 | Evaluation | E2: language ablation | Logs, comparison table | ≥ 30 episodes per model | 1.8, 2.2, 2.3 | 3 Dec | 7 Dec |

### Phase 3. Real-robot integration

| ID | Area | Work item | Deliverable | Done when | Depends on | Start | Due |
|---|---|---|---|---|---|---|---|
| 3.1 | Robot | Receive the arm and connect the SDK | Connection script, notes in `docs/` | Each joint reads and moves at low speed | None | 24 Nov | 28 Nov |
| 3.2 | Safety | Safety layer for the real arm | Config and tests in `src/vla_robot_sorting/safety/` | Out-of-workspace commands rejected in tests and on the arm at low speed | 3.1 | 24 Nov | 1 Dec |
| 3.3 | Robot | Cameras and calibration | Calibration files, updated `configs/twin_cameras.json` | Calibration error recorded | 3.1 | 28 Nov | 3 Dec |
| 3.4 | Data | Real demonstrations | `panthera_real_v1` on the Hugging Face Hub | 30–50 successful episodes | 3.1, 3.3 | 1 Dec | 7 Dec |
| 3.5 | Robot | Run a policy on the real arm | Control script, test video | The policy completes one pick-and-place | 2.2, 3.2 | 3 Dec | 7 Dec |
| 3.6 | Report | Chapter 3: hardware and sensing | Draft | Compiles, all sections present | 3.2, 3.3 | 1 Dec | 10 Dec |

### Phase 4. Real-robot experiments

| ID | Area | Work item | Deliverable | Done when | Depends on | Start | Due |
|---|---|---|---|---|---|---|---|
| 4.1 | Model | Train with real data | Two checkpoints, configs, logs | Real-only and combined checkpoints run through the safety layer | 2.1, 3.4 | 8 Dec | 11 Dec |
| 4.2 | Evaluation | E3: real robot | Evaluation sheets, videos, tables | ≥ 30 episodes per condition with intervals | 3.5, 4.1 | 11 Dec | 18 Dec |
| 4.3 | Evaluation | Failure analysis, figures and tables | Scripts and generated outputs | Every reported number traces to a log | 2.4, 2.5, 4.2 | 15 Dec | 20 Dec |
| 4.4 | Robot | Demo video | Video ≤ 3 min | Published with the release | 4.2 | 15 Dec | 20 Dec |
| 4.5 | Model | Extension: household items on the real arm | Real demonstrations, table | ≥ 30 real episodes | 2.4, 3.4 | 8 Dec | 20 Dec |
| 4.6 | Report | Chapter 4: method | Draft | Compiles, all sections present | 2.2, 3.5 | 1 Dec | 14 Dec |

Household items are part of the simulated dataset and the E1 evaluation (task specification 1.3). Item 4.5 extends them to the real arm and starts only if phase 2 finishes on time.

### Phase 5. Write-up and release

| ID | Area | Work item | Deliverable | Done when | Depends on | Start | Due |
|---|---|---|---|---|---|---|---|
| 5.1 | Report | Chapter 5: experiments | Draft | Compiles, all sections present | 4.3 | 16 Dec | 24 Dec |
| 5.2 | Report | Chapter 6 and abstract | Draft | Compiles, all sections present | 5.1 | 21 Dec | 26 Dec |
| 5.3 | Report | Presentation | Slides | One rehearsal done | 5.1 | 26 Dec | 30 Dec |
| 5.4 | Management | Submit and release | Release `v1.0` with dataset and checkpoint links | Submitted on time | 5.2, 5.3 | 31 Dec | 31 Dec |

## 5. Requirement coverage

| Objective (README) | Work items |
|---|---|
| 1. VLA survey | 1.10 |
| 2. Hardware and sensing | 3.1, 3.2, 3.3, 3.6 |
| 3. Fine-tuning for recognition, localisation and pick-and-place | 1.3–1.7, 2.1–2.3, 4.1 |
| 4. Complete sorting application | 3.5, 4.2, 4.4 |
| 5. Evaluation | 1.8, 2.4, 2.5, 4.2, 4.3 |

## 6. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| The arm arrives after 30 Nov | Less time for real data and E3 | Build and test the safety layer and policy client in simulation first; reduce E3 conditions |
| Free GPU quota runs out | Slower training | Use the RTX 5090 first; budget up to 500,000 VND for rented GPUs |
| Weak understanding of Vietnamese commands | Low success on Vietnamese | Report the measured gap; raise the share of Vietnamese commands in training |
| Simulation machine available 2–3 sessions per week | Slower data generation and evaluation | Run recording and evaluation headless in overnight batches |
