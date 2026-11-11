# Data and training

## 1. Sources

| Source | Produced by | Target size |
|---|---|---|
| Simulated demonstrations | Scripted expert on randomised scenes, recorded by `tools/twin/twin_record.py` | 500–1,000 successful episodes |
| Training commands | Language model paraphrases in Vietnamese and English for each (object, container) pair | ≥ 30 per pair and language |
| Test commands | Written by people outside the team through a form; never used for training | 100–200 |
| Real demonstrations | Leader–follower teleoperation on the Panthera-HT | 30–50 episodes |

### 1.1 Scene randomisation

A policy that sees the same layout with the same command learns positions instead of reading the command. The generator therefore:

- varies object positions, colours and counts between layouts;
- pairs each layout with several commands that target different objects;
- records the random seed so the dataset can be rebuilt exactly.

### 1.2 Commands

Each generated command stores the model name and version, the prompt and the generation date. Duplicates and commands that point at the wrong object are filtered out. The human-written test set stays separate from all training data.

## 2. Formats

### 2.1 Raw episodes (`twin_record.py`)

```
<out>/episode_000000/meta.json            command, scene level, object, container, success, fps
<out>/episode_000000/states.npy           float32 [T, D]  joint positions + gripper
<out>/episode_000000/actions.npy          float32 [T, D]  joint-position target (next state)
<out>/episode_000000/<camera>/000000.jpg  320×240 frames
```

The sampling rate defaults to 15 Hz (`configs/twin_cameras.json`).

```sh
python tools/twin/twin_record.py --out data/twin_raw --episodes 50 --levels 1,2,3 --seed 0
```

### 2.2 LeRobot dataset (`lerobot_convert.py`)

```sh
python tools/twin/lerobot_convert.py --raw data/twin_raw --root data/panthera_twin_v1 \
    --repo-id <user>/panthera_twin_v1
```

Datasets carry a version suffix (`panthera_twin_v1`, `panthera_twin_v2`). A version is immutable once a reported model has been trained on it.

## 3. Training

| Model | Role | Hardware |
|---|---|---|
| Selected VLA (SmolVLA or Evo-1) | Main model | RTX 5090 or Kaggle |
| ACT (Action Chunking with Transformers) | Baseline without language | Any GPU with 6 GB or more |

Every reported training run stores:

- the repository commit;
- the dataset version;
- the full configuration: hyperparameters, seed, step count;
- the final checkpoint and the training log, uploaded to the Hugging Face Hub or Kaggle.

Turing GPUs (GTX 1660 Ti, Kaggle T4) lack hardware bf16; use fp16 or fp32 there.

`notebooks/kaggle_train_panthera_twin.ipynb` contains the ACT and SmolVLA training commands used on Kaggle.

## 4. Running a policy in simulation

```sh
# LeRobot environment
python tools/twin/policy_bridge.py serve --checkpoint <checkpoint>/pretrained_model \
    --camera-map wrist=camera1,workspace=camera2

# RoboTwin environment, second terminal
python tools/twin/twin_console.py --executor policy --policy-server 127.0.0.1:5555 --level 2
```

`--camera-map` applies to SmolVLA fine-tuned from `smolvla_base`, which expects cameras named `camera1` to `camera3`.

Evaluation and reporting: [evaluation protocol](evaluation-protocol.md).
