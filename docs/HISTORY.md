# Project history (September – October 2026)

A short record of what was built before V2. The current plan is in [`PROJECT_STATE.md`](PROJECT_STATE.md) and decisions are in [`DECISIONS.md`](DECISIONS.md).

## Weeks 1–2 (mid-September): research framing

- **Literature survey.** The survey covered about 56 sources. The working thesis was a "decoupled semantic harness": a cloud VLM handles reasoning, kept separate from grasp generation and kinematic control. The thesis claimed this harness would match end-to-end VLAs, needing less data and less effort to reconfigure. End-to-end VLAs were treated only as baselines.
- **Prototype.** A synthetic 6-DoF arm was built in MuJoCo with a custom IK solver: 98% convergence, 0.045 mm median error.
- **Superseded.** That code and most of these decisions were replaced in Sprint 1 (see `DECISIONS.md`, D01–D28).

## Sprint 1 (1–17 October): modular pipeline on simulated Panthera-HT

**Stack**
- Panthera-HT in RoboTwin (SAPIEN/PhysX), Ubuntu 22.04 on a GTX 1660 Ti 6 GB.
- Motion: CuRobo v0.7.8, with MPlib as fallback.
- Grasping: a geometry-first 6-DoF grasp generator.
- Execution: closed loop.
- Safety: fail-closed layer.

**Accepted evidence** (manifests in `experiments/accepted/`)
- S1.1: simulator, cameras and joint state.
- S1.2: planning and retiming. CuRobo median planning time is about 75 ms. Retiming respects 2 rad/s² and 500 rad/s³.
- S1.3: grasp generation 20/20; replanning after the object is moved 20/20.
- A-007: when the planner fails, the robot does not move.
- S1-M: full 8-phase pick-and-place, 3/3 seeds, independently audited (task `move_pillbottle_pad`).
- Operator baseline: two fixed commands, one English and one Vietnamese, both 8/8 end to end, audited.

**Limits that must be stated with these results**
- Object identity came from the simulator's actor registry, not from vision.
- Only two fixed commands were tested, and Gemini was not in the accepted path.
- Safety is valid only in simulation.
- There was no sensor timestamp synchronisation.
- No VLA was trained.

## Digital twin v1–v15 (17–19 October)

**What it is**
- A persistent RoboTwin scene, `scripts/linux/twin_*.py`.
- Free Vietnamese or English commands go to a Gemini planner, which returns a JSON plan.
- The twin executes the plan with grasp checks, obstacle-aware CuRobo, placement checks, and replanning with recovery.
- It has seven scene levels.
- It is a demo and analysis tool, not acceptance evidence.

**Statistics over 51 manual sessions** (`scripts/linux/twin_report_stats.py`)
- Pick-and-place steps: 141/277 (51%) overall; 32/52 (62%) in v15.
- Tetrahedron grasps: 0/24.
- Failures concentrate in lift (47) and grasp (42).

**Lessons kept for V2**
1. The CuRobo world provided by RoboTwin contains only the table. Obstacles must be added explicitly.
2. RoboTwin's `choose_best_pose` keeps the most tilted feasible pose.
3. Panthera-HT reach is limited when the gripper points straight down: 0.86–0.92 m, versus 1.07–1.10 m when tilted 35–45° (`twin_kinematics.py`). At lift height, 0% of table cells are reachable vertically, versus 69% at a 40° tilt.
4. The all-zero home pose sits on the joint-2/3 limits. It must be moved inside the limits.
5. Fake backends and unit tests do not model reach. Live runs are required.
6. Every log must record the code version (`twin_version`).

**Learning pipeline**
- `twin_record.py`: scripted-expert demos at 15 Hz.
- `lerobot_convert.py`: converts recordings to a LeRobot dataset.
- `notebooks/kaggle_train_panthera_twin.ipynb`: trains on Kaggle.
- `policy_bridge.py`: serves a trained policy to the twin.
- Status: only smoke-tested. This is the starting point for V2.
