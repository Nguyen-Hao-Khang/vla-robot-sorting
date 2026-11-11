# ADR 0005: Base VLA model

- Status: Proposed
- Decision due: 23 November 2026 ([project plan](../project-plan.md), item 1.7)

## Context

Constraints:

- inference on a 6 GB GPU without bf16 (GTX 1660 Ti);
- fine-tuning in a few hours on an RTX 5090 or a Kaggle T4;
- understanding of Vietnamese commands;
- available in LeRobot, to share the dataset format and training tools.

## Options

| Model | Size | Language backbone | Notes |
|---|---|---|---|
| SmolVLA | ~0.45 B parameters | SmolLM2, trained mostly on English | Pretrained on LeRobot community data (SO-100 arms) |
| Evo-1 | ~0.77 B parameters | Qwen2.5, multilingual | Available in LeRobot |

ACT serves as the baseline in every case and is not a candidate for the main model.

## Measurements

Both models train on the same small dataset (about 50 simulated episodes):

1. VRAM and inference latency on the GTX 1660 Ti in fp16;
2. wall-clock time for 10,000 fine-tuning steps on the RTX 5090;
3. simulated success over 30 English and 30 Vietnamese episodes.

## Decision

Pending. This record will be updated with the measurement table and the choice.
