# Evaluation protocol

This protocol fixes the measurements before any experiment runs. Metrics and thresholds do not change after results are seen.

## 1. Metrics

| Metric | Definition |
|---|---|
| Task success rate | Fraction of episodes that meet all success criteria in the [task specification](task-spec.md#4-success-criteria) |
| Target selection rate | Fraction of episodes where the robot grasps the named object, regardless of placement |
| Grasp success rate | Fraction of episodes where the robot lifts an object off the table |
| Episode time | From command receipt to gripper release |
| Safety interventions | Number of actions the safety layer rejects per episode |
| Failure phase | Phase in which a failed episode stopped: approach, grasp, lift, transport, place |

In simulation, a script computes the metrics from simulator state. On the real robot, an operator fills an evaluation sheet and records video of every episode.

## 2. Experiments

| ID | Question | Conditions |
|---|---|---|
| E1 | How often does the fine-tuned VLA succeed in simulation? | Vietnamese vs English commands; training-set commands vs held-out human-written commands |
| E2 | How much does language input contribute? | Fine-tuned VLA vs ACT on identical scenes and seeds |
| E3 | How much performance transfers to the real arm? | Trained on simulation only, real only, simulation + real |

Extensions, time permitting: sorting by semantic category; robustness to camera pose, lighting and distractor objects.

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
