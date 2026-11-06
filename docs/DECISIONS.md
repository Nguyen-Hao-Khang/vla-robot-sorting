# Decision Ledger (V2)

This ledger reconciles every earlier decision with [`PROJECT_STATE.md`](PROJECT_STATE.md). It was provisionally approved by the PO on 2026-10-27.

- **Status values:** KEEP / REVISE / DROP.
- **Rule for new decisions:** add a row with date, decision, reason and approver. Never edit an old row silently; supersede it with a new row instead.

## V2 decisions (2026-10-27, product owner)

| ID | Decision | Reason |
|---|---|---|
| V2-01 | The main system is a **flat end-to-end VLA** running locally. VLM + VLA is a variant | Topic requirement 3; PO goal of minimising run-time API use; TA guidance |
| V2-02 | Two-level sorting task: attribute first, semantic category second | Keeps the week-1 Q2 scope (open categories) while giving a reportable first level |
| V2-03 | VI/EN/code-switch commands are evaluated and reported. Error attribution is logged but is no longer a claim (old C4) | PO choice |
| V2-04 | Retained claims: C1 (revised: flat vs VLM + VLA data threshold), C2 (unseen category cost), C3 (revised: sim → real). No hard % thresholds | PO choice; thresholds were researcher-set and untested |
| V2-05 | MLLM may be called online during development. Offline use at run time is encouraged, not required | PO clarification |
| V2-06 | Data: Twin v15 expert + RoboTwin AI code-gen + MLLM instructions, in LeRobot format; 500–1,000 sim demos and 30–50 real demos | PO choice; TA suggestion on RoboTwin data generation |
| V2-07 | Modular pipeline and Twin v15 become the demo expert, the safety-layer reference and the upper-bound reference | PO choice |
| V2-08 | Real-robot results are required. Go/no-go is on 2026-11-30 | PO choice; robot expected in November |
| V2-09 | Evidence process stays as in Sprint 1 (release bundle, SHA256, source-lock, independent auditor) for headline results | PO choice |
| V2-10 | Report: six chapters, one dedicated to hardware. Literature survey redone from scratch. No conference submission | PO choice |
| V2-11 | Do not buy a camera before the robot arrives. Build the simulated scene to match the real table | PO choice |
| V2-12 | VLA model choice is deferred to the PO survey (due T2) | PO choice |
| V2-14 | (pending, M0) Choose the main VLA between SmolVLA and Evo-1 by a measured bake-off: fp16/fp32 inference on a GTX 1660 Ti, 1,000-step fine-tune on a Kaggle T4, Vietnamese tokenisation. ACT stays the no-language baseline | See `docs/research/v2-vla-model-survey.md` |
| V2-13 | The robustness defences against "it is just fine-tuning" are: a human-written held-out command set, an instruction-data strategy comparison, and a held-out category | PO choice |

## Week 1–2 decisions (D01–D28; there are no D07–D09)

| ID | Original | V2 | Note |
|---|---|---|---|
| D01 | arXiv preprints with traceable code count as peer-reviewed | REVISE | Allowed, but every number must be quoted from the original |
| D02 | Harness-vs-E2E thesis, C1–C4 with thresholds | REVISE | Replaced by the V2 thesis (V2-04) |
| D03 | Scenario A/B with a week-5 go/no-go | REVISE | Go/no-go kept; date 2026-11-30 (V2-08) |
| D04 | Five-chapter report, about 18,000 words | REVISE | Six chapters (V2-10) |
| D05 | Conference paper option X/Y | DROP | No submission |
| D06 | Postpone writing; engineering first | DROP | Writing runs in parallel from T1 |
| D10 | Exclude Isaac Sim | KEEP | GPU has no RT cores |
| D11 | MuJoCo on CPU / Windows | DROP | Already replaced by RoboTwin (AD-01) |
| D12 | UR5e + SO-101 embodiments | DROP | Panthera-HT only |
| D13 | `EmbodimentSpec.task_mask` | DROP | Week-1 code not used |
| D14 | Single `ArmBackend` boundary | DROP | Replaced by ADR-0001 contracts |
| D15 | Custom DLS IK | DROP | Replaced by CuRobo (now expert only) |
| D16 | World-frame orientation error, no `mju_subQuat` | DROP | MuJoCo-specific |
| D17 | Quintic + SLERP interpolation | DROP | VLA action chunks plus safety-layer limits |
| D18 | Weld gripper | DROP | Replaced by PhysX gripper |
| D19 | Overhead camera convention | DROP | RoboTwin head + wrist cameras |
| D20 | Temporal ensembling removed from harness loop | REVISE | Re-evaluate when the VLA is chosen (chunking/smoothness) |
| D21 | IK branch locking | DROP | Week-1 code not used |
| D22 | Camera extrinsics from a calibrated file, never read from the simulator | KEEP | Mandatory for the real robot |
| D23 | Kabsch with determinant fix | DROP | Week-1 code not used; revisit at calibration |
| D24 | Geometric step ladder | DROP | Week-1 code not used |
| D25 | Safety gate: uniform velocity scaling, whole-segment workspace check, latched E-Stop | KEEP | Core of the real-robot safety layer |
| D26 | JSONL logging with fsync | REVISE | Sprint 1 evidence process + LeRobot datasets |
| D27 | Mandatory timesheet | DROP | C2 is measured in demos, GPU hours and changed lines |
| D28 | PCA heuristic grasping first | DROP | Replaced by the geometry-first generator (expert only) |

## Sprint 1 decisions

| ID | Original | V2 | Note |
|---|---|---|---|
| AD-01 | Panthera-HT + RoboTwin, pinned SHAs | KEEP | |
| AD-02 | Show-Harness is reference only; its 9-token action set is not used | KEEP | |
| AD-03 | SE(3) grasps come from the geometry layer; the VLM never emits motor commands | **REVISE** | A locally trained VLA **may** emit joint actions, always through the safety layer. A cloud VLM / API still **never** emits motor commands |
| AD-04 | Smooth motion via CuRobo + time dilation | REVISE | Main path: VLA action chunks with velocity/acceleration/jerk checks in the safety layer. CuRobo stays inside the demo expert |
| AD-05 | Closed-loop, receding-horizon execution | KEEP | |
| AD-06 | Two feedback time scales | KEEP | Applies to the VLM + VLA variant |
| AD-07 | Rigid proxies for thin objects | KEEP | |
| AD-08 | Gemini backend; fallback listed as `gemini-3.8-flash` | REVISE | The actual fallback is `gemini-3.5-flash-lite`; a local VLM is added |
| ADR-0001 | Project-owned contracts | KEEP | Add an `ActionPolicy` contract for the VLA |
| ADR-0002 | Grasp + motion stack | REVISE | Accepted on 2026-10-27, scoped to the demo expert and safety baseline |
| Rule | Never loosen safety thresholds to obtain a PASS | KEEP | |
| Rule | Simulator identity is labelled `SIM_GROUND_TRUTH` | KEEP | The expert may use ground truth; the VLA policy may **not** |
| Rule | Repository language is English | KEEP | |
