# V2 literature register (Chapter 2, rebuilt from scratch)

**Started:** 2026-10-27. This register replaces the week-1 Layer-1 survey as the citation base. The week-1 survey and its 56 entries are kept only as historical context.

## Rules

- An entry enters the report only after a team member reads the original and fills in "Read by" and "Key facts".
- Numbers are copied from the original paper, not from surveys, blogs or AI summaries.
- "ID checked" means the identifier and title were confirmed on the publisher's or arXiv's page on 2026-10-27. It does **not** mean the paper has been read.

| # | Family | Work | Identifier | ID checked | Read by | Key facts (from the original) |
|---|---|---|---|---|---|---|
| 1 | A. Foundation VLA | OpenVLA-OFT: Fine-Tuning VLA Models — Optimizing Speed and Success (RSS 2025) | arXiv 2502.19645 | yes | | |
| 2 | A | π0.5: a VLA with Open-World Generalization (CoRL 2025) | arXiv 2504.16054 | yes | | |
| 3 | A | Survey: VLA Models for Embodied AI (Ma et al., IEEE TNNLS) | arXiv 2405.14093 | yes (PDF in project folder) | Partly (explainer Part 4) | |
| 4 | A | Survey: VLA Models — Concepts, Progress, Applications and Challenges (Sapkota et al.) | arXiv 2505.04769 | yes (PDF in project folder) | Partly (explainer Part 4) | Some numbers in this survey are inconsistent; verify against primary papers |
| 5 | B. Small / efficient VLA | SmolVLA | arXiv 2506.01844 | yes | | |
| 6 | B | TinyVLA (RA-L 2025) | arXiv 2409.12514 | yes | Architecture checked | Pythia-based VLMs 0.4–1.3B + diffusion-policy head (see `v2-vla-model-survey.md`) |
| 7 | B | FAST: Efficient Action Tokenization (RSS 2025) | arXiv 2501.09747 | yes | | |
| 8 | C. Automatic data generation | RoboTwin 2.0: Scalable Data Generator and Benchmark with Strong Domain Randomization | arXiv 2506.18088 | yes | | |
| 9 | C | MimicGen (CoRL 2023) | arXiv 2310.17596 | yes | | |
| 10 | C | GenSim: Generating Robotic Simulation Tasks via LLMs (ICLR 2024) | arXiv 2310.01361 | yes | | |
| 11 | D. Multilingual VLA | When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in VLA Models (ACL 2026) | arXiv 2606.11906 | yes | Summary checked | Includes Vietnamese (translated LIBERO): OpenVLA-OFT 97.1 → 59.6, π0.5 96.9 → 57.3; inference-time step-wise alignment |
| 12 | D | Beyond English: Uncovering the Multilingual Gap in VLA Models | arXiv 2606.15714 | yes | Summary checked | zh/fr/ru/ar only (no Vietnamese); MPCA alignment |
| 13 | E. Hierarchical vs flat | Hi Robot: Open-Ended Instruction Following with Hierarchical VLA Models (ICML 2025) | arXiv 2502.19417 | yes | | |
| 14 | E | VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in VLA Models (NeurIPS 2025) | arXiv 2506.17561 | yes | | |
| 15 | E | What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents | arXiv 2606.10267 | yes | | |
| 16 | E | Show-Harness: Just a VLM Agent Can Play Robots | arXiv 2609.10522 | yes (PDF in project folder) | Partly (explainer Part 4) | |
| 17 | F. Sim-to-real | Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation (RSS 2025) | arXiv 2503.24361 | yes | | |
| 18 | F | Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv) | arXiv 2405.05941 | yes | | |
| 19 | G. Sorting applications | ROBOCYCLE: Autonomous Dual-Arm Robotic Manipulation and Coordination for Recycling | arXiv 2607.03616 | yes | | |
| 20 | G | Colosseum V2: Benchmarking Generalization for VLA Models | arXiv 2605.27759 | yes (search result) | | |

## Still to find

- **Benchmark:** the original LIBERO paper (benchmark used in E1).
- **Safety:** at least two papers on safety layers or runtime monitoring for learned manipulation policies (family H).
- **Hardware:** Panthera-HT vendor documentation (SDK, LeRobot integration) for Chapter 3.
- **Calibration:** hand-eye calibration and camera calibration references (ChArUco) for Chapter 3.
