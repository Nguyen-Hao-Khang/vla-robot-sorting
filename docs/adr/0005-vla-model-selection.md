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

### Training and open-loop results (15 November 2026)

Both candidates and the ACT baseline were trained on Kaggle (T4, batch 8, LeRobot at commit `e581c1d`) on two datasets: the 80-episode pilot and the 450-episode seed-1 dataset (136,739 frames; 209 English and 241 Vietnamese commands). Each seed-1 run had to fit in one 12-hour session, so the seed-1 VLA runs cover less than one epoch.

| Run | Steps | Epochs | Wall time | Final training loss |
|---|---|---|---|---|
| SmolVLA, pilot | 10,000 | 3.3 | ~11.4 h | 0.037 |
| SmolVLA, seed 1 | 8,000 | 0.47 | 492 min | 0.089 |
| Evo-1, pilot | 10,000 | 3.3 | 2 sessions (resumed at 5,000) | 0.104 |
| Evo-1, seed 1 | 10,000 | 0.59 | 442 min | 0.152 |
| ACT, pilot | 20,000 | 6.6 | 67 min | 0.104 |
| ACT, seed 1 | 50,000 | 2.9 | 161 min | 0.099 |

The losses are not comparable across models: SmolVLA and Evo-1 report flow-matching losses, ACT an L1 plus KL loss.

An open-loop check (`tools/train/open_loop_check.py`, GTX 1660 Ti, Windows) fed 12 recorded episodes to each checkpoint through `policy_bridge.py`: 6 from the seed-1 dataset and 6 from the pilot dataset, half English and half Vietnamese. The table reports the mean absolute joint error between the executed chunk and the recorded actions. The chunk-hold baseline keeps the joint positions at the start of each chunk for the whole chunk; a policy that has learned anything should be well below it. Each seed-1 model has not seen the pilot episodes and each pilot model has not seen the seed-1 episodes.

| Checkpoint | Joint error, seed-1 episodes (rad) | Joint error, pilot episodes (rad) | Chunk-hold baseline (rad) | Chunk latency | Peak GPU memory |
|---|---|---|---|---|---|
| ACT, seed 1 | 0.040 (trained on) | 0.123 | 0.26 | 0.10 s | 0.2 GB |
| SmolVLA, seed 1 | 0.065 (trained on) | 0.071 | 0.19--0.21 | 0.78 s | 1.0 GB |
| Evo-1, seed 1 | 0.418 (trained on) | 0.408 | 0.19--0.21 | 5.2 s | 3.4 GB |
| ACT, pilot | 0.129 | 0.040 (trained on) | 0.26 | 0.10 s | 0.2 GB |
| SmolVLA, pilot | 0.101 | 0.034 (trained on) | 0.19--0.21 | 0.78 s | 1.0 GB |
| Evo-1, pilot | 0.268 | 0.235 (trained on) | 0.19--0.21 | 5.2 s | 3.4 GB |

Reading:

- SmolVLA transfers best to episodes it has not seen (0.071 rad against 0.065 on its own data); ACT fits its training data closely but its error triples on unseen episodes.
- Both Evo-1 checkpoints are worse than the chunk-hold baseline, even on their own training episodes: consecutive predicted actions jump by tenths of a radian. The likely cause is training, not inference: in stage 1 the vision-language backbone is frozen and the action head starts from random weights, and LeRobot's default learning rate for Evo-1 (1e-5, cosine decay) is low for a head trained from scratch in 10,000 steps. Evo-1 therefore needs a second training run (higher learning rate) before measurement 3 can compare it fairly with SmolVLA.
- English and Vietnamese commands give similar open-loop errors for every checkpoint. Open-loop error does not show whether the policy picks the named object; the closed-loop simulation (measurement 3) does.

### First closed-loop results (15 November 2026)

Seed-9 block scenes, built-in commands, 30 English and 30 Vietnamese episodes per checkpoint, safety profile `simulation` (before the gripper tolerance of ADR 0004's revision), commit `f197abd` with the dual-arm `take_action` fix of this change. The scripted expert succeeds in 220 of 220 episodes on the same scenes.

| Checkpoint | Language | Success | Target selected | Grasped | Safety stops |
|---|---|---|---|---|---|
| ACT, seed 1 | en | 1/30 | 2/30 | 6/30 | 23/30 |
| ACT, seed 1 | vi | 0/30 | 2/30 | 5/30 | 23/30 |
| SmolVLA, seed 1 | en | 0/30 | 1/30 | 5/30 | 2/30 |
| SmolVLA, seed 1 | vi | 2/30 | 3/30 | 8/30 | 2/30 |

Both checkpoints are far from usable. ACT ignores the command (it lifts the same objects in English and Vietnamese, as expected without language input), and SmolVLA picks the named object in 4 of its 13 grasps, close to chance with two to four objects. Before more training, the replay check of the [evaluation protocol](../evaluation-protocol.md) tests whether the action interface itself loses success, and shorter chunk execution is tried at serving time.

## Decision

Pending measurement 3, which needs the first simulated dataset (plan item 1.3). If measurement 3 shows no clear difference, the inference latency on the deployment GPU decides.
