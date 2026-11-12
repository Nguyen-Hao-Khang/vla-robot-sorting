# ADR 0005: Base VLA model

- Status: Proposed
- Decision due: 23 November 2026 ([project plan](../project-plan.md), item 1.7)

## Context

Constraints:

- inference on a 6 GB GPU without bf16 (GTX 1660 Ti);
- fine-tuning in a few hours on a Kaggle T4 (16 GB), so the project does not depend on a team member's GPU;
- understanding of Vietnamese commands;
- available in LeRobot, to share the dataset format and training tools.

## Options

| Model | Size | Language backbone | Notes |
|---|---|---|---|
| SmolVLA | 0.45 B parameters, 0.10 B trained | SmolVLM2-500M (16 layers kept), language model trained mostly on English | Pretrained on LeRobot community data (SO-100 arms) ([arXiv:2506.01844](https://arxiv.org/abs/2506.01844)); LeRobot 0.4.4 and later |
| Evo-1 | 0.78 B parameters, 0.12 B trained in stage 1 | InternVL3-1B (14 layers kept), Qwen2.5 tokenizer | No robot-data pretraining ([arXiv:2511.04555](https://arxiv.org/abs/2511.04555)); in LeRobot `main` from commit `e581c1d` (version 0.7.1, Python 3.12) |

ACT serves as the baseline in every case and is not a candidate for the main model.

## Measurements

Planned:

1. GPU memory and inference latency on the GTX 1660 Ti;
2. training on the Kaggle T4;
3. simulated success over 30 English and 30 Vietnamese episodes after training both models on the same dataset (about 50 simulated episodes).

### Results so far (12 November 2026)

| Measurement | SmolVLA | Evo-1 | Script |
|---|---|---|---|
| 100 training steps, GTX 1660 Ti (6 GB, Windows) | Passes: batch 2, 1.1 s per step, about 2.3 GB | Fails before the first step: out of memory with eager attention; cuBLAS execution failure with SDPA attention | `lerobot-train` ([getting started](../getting-started.md#43-smoke-tests-plan-item-16)) |
| 100 training steps, Kaggle T4, batch 8 | Passes: peak 3.9 GB, loss 0.28 at step 100 | Passes (fp16, eager attention): peak 5.6 GB, loss 0.54 at step 100 | `notebooks/kaggle_smoke_vla.ipynb` |
| One 50-action chunk, GTX 1660 Ti | 678 ms (fp32, 10 denoising steps), 1.0 GB | 5.4 s (fp16, eager attention, 32 denoising steps); 4.8 s with 10 steps; 2.0 GB | `tools/train/infer_bench.py` |
| Tokens per word, 80 Vietnamese test commands | 3.23 (English: 1.15) | 1.34 (English: 1.14) | `tools/train/tokenizer_stats.py` |

The Kaggle runs include model download in their wall time (393 s and 330 s), so they say nothing about training speed yet.

Reading of the results:

- At 15 Hz a 50-action chunk covers 3.3 s of motion. SmolVLA predicts the next chunk well within that time on the GTX 1660 Ti. Evo-1 needs 4.8--5.4 s, longer than the chunk, so the arm would wait between chunks on this GPU. Most of that time is the vision-language backbone, because fewer denoising steps save little.
- SmolVLA's tokenizer splits a Vietnamese word into about three tokens, against about one for English. Evo-1's tokenizer treats both languages alike. Whether this changes success on Vietnamese commands is what measurement 3 tests.

## Decision

Pending measurement 3, which needs the first simulated dataset (plan item 1.3). If measurement 3 shows no clear difference, the inference latency on the deployment GPU decides.
