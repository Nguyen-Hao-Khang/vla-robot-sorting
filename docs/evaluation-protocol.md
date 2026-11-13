# Evaluation protocol

This protocol fixes the measurements before any experiment runs. Metrics and thresholds do not change after results are seen.

## 1. Metrics

| Metric | Definition |
|---|---|
| Task success rate | Fraction of episodes that meet all success criteria in the [task specification](task-spec.md#6-success-criteria) |
| Target selection rate | Fraction of episodes where the robot grasps an allowed object (the named object, or any class member for class commands), regardless of placement |
| Grasp success rate | Fraction of episodes where the robot lifts an object off the table |
| Episode time | From command receipt to gripper release |
| Safety interventions | Number of actions the safety layer rejects per episode |
| Failure phase | Phase in which a failed episode stopped: approach, grasp, lift, transport, place |
| Sorting accuracy | In a sorting run, fraction of objects that end in the container of their class |

In simulation, a script computes the metrics from simulator state. On the real robot, an operator fills an evaluation sheet and records video of every episode.

## 2. Experiments

| ID | Question | Conditions |
|---|---|---|
| E1 | How often does the fine-tuned VLA succeed in simulation? | Vietnamese vs English commands; training-style commands vs held-out test commands; object vs class commands; criterion (colour, shape, size); object family (blocks, real-kit pieces, household items) |
| E2 | How much does language input contribute? | Fine-tuned VLA vs ACT on identical scenes and seeds |
| E3 | How much performance transfers to the real arm? | Trained on simulation only, real only, simulation + real |

Extensions, time permitting: household items on the real arm; robustness to camera pose, lighting and distractor objects.

## 3. Trials and statistics

- At least 30 episodes per condition.
- Success rates reported with Wilson 95% confidence intervals.
- Compared conditions share the same scene seeds.
- All failed episodes are reported. No episode is excluded after results are seen.

Wilson interval for $k$ successes in $n$ episodes, $\hat p = k/n$, $z = 1.96$:

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

## 4. Traceability

Each results table records:

- the repository commit and dataset version;
- the model checkpoint;
- the list of scene seeds;
- the path to the raw evaluation log.

A script in this repository generates every figure and table from the raw logs. Numbers are never copied by hand.

## 5. Running a simulated evaluation

`tools/twin/twin_eval.py` runs seeded episodes in RoboTwin with the scripted expert or a learned policy served by `tools/twin/policy_bridge.py`, and writes one JSON line per episode.

```sh
# RoboTwin environment
python tools/twin/twin_eval.py run --executor expert --episodes 30 --seed 1 --language en \
    --out results/expert_en.jsonl
python tools/twin/twin_eval.py run --executor policy --policy-server 127.0.0.1:5555 \
    --checkpoint smolvla_v1 --episodes 30 --seed 1 --language vi --command-source generated \
    --out results/smolvla_vi_generated.jsonl
# any environment
python tools/twin/twin_eval.py summarise results/*.jsonl --by executor,language,command_kind
```

- **Pairing.** Episode `i` of a run with seed `S` uses the episode seed `S × 100000 + i`. The scene, the layout, the target, the destination and the references depend only on that seed, so two runs with the same seed differ only in the condition under test (language, command source, executor).
- **Command sources.** `builtin` uses the templates in `sorting_task.py`, `generated` the language-model templates in `configs/command_templates.json`, `human` the held-out test commands in `configs/test_commands_v1.csv`. With `human`, the episodes come from the file: each row names its episode seed, and only rows marked `Reviewed=yes` are used. A template run with `--seed 9` covers the same scenes as the file, so the two command sources are paired.
- **Judging.** Object locations are read from the simulator before and after the episode and checked against [section 6 of the task specification](task-spec.md#6-success-criteria). Episodes longer than 60 s fail. Policies act at 15 Hz, so a policy episode has at most 900 actions.
- **Safety layer in simulation.** Policy actions pass a stand-in filter: joint targets outside the joint limits are clamped and counted as interventions; a step larger than 0.35 rad from the current joint state stops the episode.
- **Target selection and grasping.** An object counts as lifted when it rises 3 cm above its resting height. The first lifted object decides target selection.
- **Failure phase.** For the expert, the failed executor step mapped to approach, grasp, lift, transport or place. For a policy, inferred from what was lifted: nothing or a wrong object (grasp), an allowed object that did not reach any container (transport), an allowed object in a container but the episode still failed (place). Safety stops and timeouts are reported as such.
- **Resume.** Rerunning with the same `--out` skips episodes already in the file.

Each record holds the commit, the checkpoint label, the command source and the seeds, which covers the traceability items of section 4.
