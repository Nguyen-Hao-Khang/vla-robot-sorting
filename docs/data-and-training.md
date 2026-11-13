# Data and training

## 1. Sources

| Source | Produced by | Target size |
|---|---|---|
| Simulated demonstrations | Scripted expert on randomised scenes, recorded by `tools/twin/twin_record.py` | 500–1,000 successful episodes |
| Training commands | Language model paraphrases in Vietnamese and English for each (object, container) pair | ≥ 30 per pair and language |
| Test commands | One free command per evaluation episode, drafted with `tools/data/draft_test_commands.py` and edited and approved by the team; never used for training | 100–200 |
| Real demonstrations | Leader–follower teleoperation on the Panthera-HT | 30–50 episodes |

### 1.1 Scene randomisation

A policy that sees the same layout with the same command learns positions instead of reading the command. The generator therefore:

- varies object positions, colours and counts between layouts;
- pairs each layout with several commands that target different objects;
- records the random seed so the dataset can be rebuilt exactly.

### 1.2 Commands

Commands come from three sources ([task specification](task-spec.md#53-sources-of-commands)):

1. built-in templates in `tools/twin/sorting_task.py`;
2. templates written by a language model with `tools/data/generate_command_templates.py`, stored in `configs/command_templates.json` together with the model names, prompts and date;
3. held-out test commands in `configs/test_commands_v1.csv`, edited and approved row by row by the team and used only for evaluation.

A template has one `{obj}` and one `{dest}` placeholder. The generator keeps only templates with exactly these two placeholders, 3–20 words, and the right script (Vietnamese diacritics for `vi`, ASCII for `en`), and drops duplicates. `twin_record.py` fills the placeholders with scene-unique references, so every (object, container) pair receives every template.

```sh
uv run python tools/data/generate_command_templates.py --per-kind 40
```

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
python tools/twin/twin_record.py --out data/twin_raw --episodes 50 --task sorting --seed 0 --languages en,vi
```

`--task sorting` (default) builds a new procedural scene from [the task specification](task-spec.md) for every layout and records `--episodes-per-layout` commands in it; `meta.json` stores the command kind, the object and container references, the allowed targets and the full scene. Each episode draws its command language uniformly from `--languages`. `--task levels` records the fixed scene levels instead. With `configs/command_templates.json` present, 70% of commands use its language-model templates and the rest use the built-in templates.

Two recordings with the same `--seed` must agree. Each layout and each command draws from its own generator derived from the seed, and Python, NumPy and PyTorch are reseeded before every episode, so a failed episode (which ends its layout) does not shift the ones after it; `meta.json` stores the `episode_key` (layout and command index). `tools/twin/compare_recordings.py` pairs episodes by that key, requires identical task definitions (command, references, targets, destination, scene, spawn poses, success) and reports how far the joint trajectories differ after time normalisation, since GPU motion planning and contact physics are not bit-exact:

```sh
python tools/twin/twin_record.py --out data/repro_a --episodes 50 --seed 0
python tools/twin/twin_record.py --out data/repro_b --episodes 50 --seed 0
python tools/twin/compare_recordings.py data/repro_a data/repro_b
```

Result on 13 November 2026 (commit `acc8d24`, two runs of 50 episodes with seed 0): 49 episodes paired, all 49 with identical task definitions; the one unpaired episode in each run follows a failed expert attempt in the other run. Joint trajectories differ by a median of 0.094 rad (maximum 0.40 rad) after time normalisation and by a median of 2 frames in length, so recordings are reproducible at the task level but not trajectory by trajectory. The scripted expert failed 3 of 52 and 3 of 53 attempts; failed attempts are not saved.

`tools/data/dataset_card.py <raw dir>` writes a dataset card (README.md) with the split by language, command form, references, scene sizes and destination attributes, so a recording can be checked against the task specification before training.

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

`notebooks/kaggle_train_panthera_twin.ipynb` contains the SmolVLA and ACT training runs used on Kaggle. `tools/train/kaggle_overnight.sh` records demonstrations, uploads them as a Kaggle dataset and starts the notebook on a Kaggle GPU in one unattended run:

```sh
conda activate RoboTwin
EPISODES=80 bash tools/train/kaggle_overnight.sh
```

It needs a Kaggle API token in `~/.kaggle/kaggle.json`.

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
