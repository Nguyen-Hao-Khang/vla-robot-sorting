# Panthera RoboTwin Compatibility Audit for Sprint 1

**Audit date:** 2026-10-02
**Target:** Native Ubuntu 22.04; Ryzen 7 4800HS, 16 GB RAM, GTX 1660 Ti Max-Q (6 GB).
**Verdict:** **Conditional go for S1.1.** Panthera-HT + RoboTwin remains canonical. This document is a bring-up plan, not a claim that the target laptop has passed it.

## Conclusion and evidence status

The official Panthera integration publishes an explicit single-arm RGB environment and an upstream smoke path with CuRobo planning, SAPIEN simulation, HDF5, head/wrist video, trajectory, scene, and instruction artifacts. This is strong evidence that the canonical stack is bring-up-able on supported Linux/NVIDIA systems. It is not evidence for this exact 6 GB Turing laptop.

The blocking question is whether the target runs the integration's CUDA 12.8, PyTorch 2.7.1, SAPIEN 3 beta, and CuRobo 0.7.8 stack with Vulkan and two camera streams concurrently. Turing SM 7.5 remains supported by CUDA 12, but no primary source found publishes a GTX 1660 Ti result for this Panthera configuration. First run the documented unmodified single-arm RGB smoke and capture VRAM/RAM/FPS evidence before changing rendering or planners.

| Label | Meaning |
| --- | --- |
| **Verified** | Direct official repository, documentation, or NVIDIA release-note fact. |
| **Engineering inference** | Conservative recommendation derived from verified constraints; needs target-machine measurement. |
| **Open empirical question** | Cannot be called compatible until measured on native Ubuntu. |

## 1. Authoritative upstream inventory

Commit IDs were resolved from public upstream remotes on 2026-10-02. They are a reference snapshot; experiments must use immutable detached SHAs rather than moving branches.

| Component | Authority and reference snapshot | Sprint 1 authority |
| --- | --- | --- |
| Panthera platform/model | [Panthera-HT_Main](https://github.com/HighTorque-Robotics/Panthera-HT_Main), main [cd5e255](https://github.com/HighTorque-Robotics/Panthera-HT_Main/commit/cd5e25578470d62b1d3b090e926b34fed12112c0) | **Robot platform/model authority.** Documents Python/C++/ROS 2 and RoboTwin support; not the simulator lock. |
| Physical SDK/control examples | [Panthera-HT_SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK), main [3712b3a](https://github.com/HighTorque-Robotics/Panthera-HT_SDK/commit/3712b3a3c2a4e6cbbc836a5d0b228a79d1a185c3) | **Physical SDK authority.** Out of the S1.1 simulator execution path. |
| Panthera simulated embodiment | [Panthera-HT_RoboTwin](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin), main [2aee31a](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin/commit/2aee31ad75cb7b0cd779540441c6025853a5b60d) | **Canonical S1.1 authority.** Contains Panthera URDF/SRDF/meshes/CuRobo config plus single-arm/camera/SAPIEN/TOPP/CuRobo/HDF5/video support. |
| Base simulator | [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), main [ea8b211](https://github.com/RoboTwin-Platform/RoboTwin/commit/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f), [official install guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) | Base simulation authority; Panthera integration wins where lock/layout differs. |
| Physics/rendering | [SAPIEN 3 installation guide](https://sapien-sim.github.io/docs/user_guide/getting_started/installation.html) | Runtime requirements and offscreen verification authority. |
| GPU planner | [NVlabs cuRobo](https://github.com/NVlabs/curobo), tag v0.7.8 = [d64c4b0](https://github.com/NVlabs/curobo/commit/d64c4b005459db10c5dd867d8b30a87d5bda9bdb) | Panthera requires this exact tag editable at envs/curobo. Newer upstream has a changed API. |
| Semantic/reference harness | [Show-Harness](https://github.com/showlab/Show-Harness), main [403f068](https://github.com/showlab/Show-Harness/commit/403f0685fdf4c0c8eb67c6d9c3137da96a58fdfe) | **Reference only.** Its [simulator contract](https://github.com/showlab/Show-Harness/blob/main/docs/simulators.md) uses a nine-token incremental action lattice, not this project’s locked full-SE(3) controller. Its provider, view, proprioception, planning/history, and recovery patterns are relevant references. |

### Pinning strategy

1. Clone Panthera-HT_RoboTwin into an ext4 Ubuntu worktree and detach at 2aee31ad75cb7b0cd779540441c6025853a5b60d.
2. Panthera-HT_RoboTwin vendors RoboTwin as ordinary files, not as a top-level Git submodule. Record the outer Panthera SHA and the vendored RoboTwin Git tree hash (git rev-parse HEAD:RoboTwin). At the pinned commit that tree hash is 3b0e2538a956c4c10c61c9311646a8d38e576598.
3. Treat standalone RoboTwin main as a comparison/reference repo only; its current HEAD must not be substituted into the Panthera integration checkout.
4. Verify envs/curobo resolves to d64c4b005459db10c5dd867d8b30a87d5bda9bdb before editable installation.
5. Pin Panthera main/SDK, standalone RoboTwin, and Show-Harness as reference repos only; never overwrite the integration checkout with them.
6. Per episode capture the Panthera outer SHA, vendored RoboTwin tree hash, environment/package inventory, driver/Vulkan records, config SHA-256, seed, and command line.

## 2. Minimal dependency matrix

Generic RoboTwin documents CUDA 12.1 as recommended. The canonical Panthera integration documents its own lock, which wins here: Python 3.10.20, PyTorch 2.7.1 plus CUDA 12.8, SAPIEN 3.0.0b1, MPlib 0.2.1, and CuRobo 0.7.8.

| Item | Status | Required value / decision |
| --- | --- | --- |
| OS/filesystem | **Required** | Native Ubuntu 22.04 LTS and ext4 worktree. RoboTwin supports Linux/NVIDIA GPU simulation/rendering but documents no WSL rendering support. |
| NVIDIA driver | **Required** | Linux driver capable of CUDA 12.8; prefer supported branch >= 570.26. Measure actual driver, GPU visibility, and Vulkan health. NVIDIA lists 570.26 as CUDA 12.8 GA Linux driver. |
| Environment | **Required** | Conda environment RoboTwin, Python 3.10.20. |
| PyTorch | **Required** | torch 2.7.1, torchvision 0.22.1, cu128 index; [official PyTorch matrix](https://pytorch.org/get-started/previous-versions/). |
| CUDA compiler | **Required for CuRobo** | nvcc 12.8 matching torch.version.cuda; use conda cuda-compiler 12.8 only if missing/mismatched. |
| Vulkan | **Required** | libvulkan1 plus NVIDIA ICD; vulkaninfo summary must pass. Vulkan tools is diagnostic only. |
| SAPIEN | **Required** | sapien 3.0.0b1. Do not float it to current SAPIEN during smoke. |
| Planner/timing | **Required** | mplib 0.2.1, toppra 0.6.9, CuRobo v0.7.8, warp-lang 1.12.0, Ninja 1.13.0. |
| Core Python packages | **Required** | Exact Panthera pinned NumPy/SciPy/Open3D/Gymnasium/Trimesh/Transforms3D/HDF5/OpenCV/ImageIO/Pillow/PyYAML/Hugging Face/setuptools/wheel list; see [integration README](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin#12-robotwin-collection-environment). |
| Open3D | **Required** | 0.18.0 even RGB-only, because camera module imports it. |
| FFmpeg | **Required for evidence** | Executable present; install conda-forge ffmpeg 7 only if absent. |
| Git/unzip | **Required** | System tools if absent, for checkout/assets. |
| libegl1/libxext6 | **Recommended** | Add only if headless SAPIEN lacks them. |
| PyTorch3D/point clouds | **Optional; initially disabled** | No PyTorch3D or point-cloud output for S1.1. Panthera documents missing pytorch3d as expected for RGB-only. |
| ROS 2 | **Optional; not S1.1** | No ROS 2 in official Panthera simulation smoke; add only at real-robot integration. |
| Docker/local VLM | **Optional; excluded** | Native Ubuntu is canonical; a local VLM would compete for 6 GB. |

Check before installation. Add only missing git, unzip, libvulkan1, optional vulkan-tools, and (if offscreen SAPIEN requires them) libegl1/libxext6. The Panthera integration says not to add mesa-vulkan-drivers by default on an NVIDIA host.

## 3. GTX 1660 Ti / 6 GB feasibility

### Verified facts

* NVIDIA lists Turing compute capability 7.5 as ongoing CUDA support in its [architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html). Confirm with target PyTorch, do not infer.
* SAPIEN needs an NVIDIA GPU for GPU simulation. Ray tracing needs RTX or AMD equivalent; GTX 1660 Ti is not RTX. Do not make ray tracing or denoising a requirement.
* Current cuRobo documentation says GPU greater than Turing and 4 GB VRAM. This project uses older v0.7.8, so that is a warning, not a v0.7.8 verdict.
* PyTorch 2.7.1 cu128 wheels exist for Linux x86_64; that does not prove compiled extensions work on SM 7.5.

| Risk | Initial control | Status |
| --- | --- | --- |
| nvcc/PyTorch mismatch | Require torch CUDA 12.8, nvcc 12.8, and matching CUDA_HOME. | Verified rule |
| CuRobo fails to compile/import for SM 7.5 | Use exact v0.7.8 first. TORCH_CUDA_ARCH_LIST=7.5 is a diagnostic build experiment only if normal build fails, not an assumed fix. | Open empirical question |
| GPU OOM | One arm/process/env, head+wrist RGB only, no viewer/ray tracing/denoiser/depth/point cloud/local VLM, batch size one; profile renderer and planner separately. | Engineering inference |
| Host RAM/swap pressure | Close browsers/IDEs/local servers; one worker; record RSS/swap. | Engineering inference |
| Mixed precision | Consider later only for validated project-owned neural perception; do not treat it as SAPIEN/CuRobo tuning. | Engineering inference |

**Camera guidance:** locked minimum is exactly head RGB plus wrist RGB. Start at repository RGB defaults. If telemetry demonstrates pressure, inspect the pinned config schema and commit an overlay lowering both views to 424x240, then 320x240, retaining aspect/FOV. Never invent YAML fields. Lower pixels are an engineering inference and an experiment factor.

**Headless:** likely feasible but must be tested. SAPIEN’s official offscreen test writes sapien_offscreen.png; Panthera's documented path produces video/HDF5. Viewer replay is diagnostic only.

## 4. Lowest-memory canonical configuration

| Area | Initial setting | Evidence |
| --- | --- | --- |
| Embodiment | One centered panthera-6dof, task config suffix panthera_single, 6 arm plus 1 gripper | Verified |
| Task | move_pillbottle_pad / move_pillbottle_pad_panthera_single | Verified official quick start |
| Cameras | Head RGB + wrist RGB only | Verified availability; locked minimum |
| Render | Offscreen raster/default, no viewer/ray tracing/denoiser | Engineering inference |
| Physics | One environment, upstream timestep/substeps | Avoid unbaselined contact changes |
| Planning | Prove CuRobo v0.7.8, then short receding-horizon use. If it fails/OOM, retain RoboTwin and use MPlib + TOPPRA for full-SE(3) path/timing. | Components verified; policy locked |
| Scene | One object/destination, no batch/randomization sweep/extra obstacles | Engineering inference |
| Point cloud | Disabled; no PyTorch3D | Verified optional rule |
| Logs | HDF5, trajectory, scene, seed, two videos, terminal log, GPU CSV, time output, manifest | Verified outputs + reproducibility requirement |

## 5. Smoke-test procedure

Use [ubuntu-robotwin-first-smoke-test.md](../runbooks/ubuntu-robotwin-first-smoke-test.md). It is an ordered native-Ubuntu procedure with expected evidence for driver, Vulkan, package environment, asset layout, SAPIEN, robot state/action HDF5, safe motion, two camera videos, and telemetry.

The official Panthera command is:

~~~bash
python -u script/collect_data.py move_pillbottle_pad move_pillbottle_pad_panthera_single
~~~

Expected artifact classes are episode HDF5, trajectory, instruction JSON, video, scene info, and seed. The integration documents single-arm shapes arm=[T,6], gripper=[T], vector=[T,7]. Live state API names are not documented authoritatively, so the runbook inspects recorded HDF5 state/action rather than inventing a controller call; a live project-owned state adapter belongs in S1.2.

## 6. Failure diagnosis matrix

| Symptom | Diagnose | Safe response |
| --- | --- | --- |
| No GPU or torch CUDA false | nvidia-smi, lspci kernel driver, PyTorch probe | Stop and repair native driver before RoboTwin. |
| CUDA mismatch | Compare nvcc, torch.version.cuda, CUDA_HOME | Align all at 12.8; recreate environment, do not mix toolkits. |
| Vulkan/SAPIEN launch failure | libvulkan, vulkaninfo, SAPIEN offscreen | Repair driver/ICD/EGL; keep viewer off. |
| pip/import failure | pip check, pip freeze vs exact pins | Recreate pinned RoboTwin environment; do not ad-hoc upgrade. |
| missing pytorch3d only | Confirm no point-cloud/DP3 run | Expected in RGB-only mode. |
| CuRobo path/planner error | Check envs/curobo SHA and editable location | Restore required v0.7.8 layout; diagnose first root error. |
| CuRobo OOM | Compare offscreen-only, collection, planner telemetry | Reduce optional load then MPlib/TOPPRA fallback in RoboTwin. |
| Missing assets | Published URDF/SRDF/config/mesh/object/texture tests | Use official downloader; no asset scattering/NTFS runtime. |
| One/zero camera video | FFmpeg, ffprobe, HDF5/scene/log | Camera gate fails until head and wrist streams decode. |
| Very low FPS | Power/thermal/utilization/time/camera metadata | Remove optional visuals then reduce verified camera fields; do not claim an unmeasured rate. |

## 7. Fallback ladder

1. Single arm, one process, two RGB cameras, no viewer/multi-env batch, then a pinned lower-resolution overlay.
2. Disable point cloud/PyTorch3D, depth/segmentation saving, ray tracing, denoising, high-quality replay. Keep RGB proof.
3. Retain Panthera RoboTwin/SAPIEN and use shipped MPlib + TOPPRA for full-SE(3) plans/timing if CuRobo is not viable.
4. Use CPU/offline only for config/HDF5/trajectory/language/unit-test work; it cannot certify canonical GPU simulation.
5. Last resort: run same pinned Ubuntu bundle on a compatible Linux/NVIDIA host and record hardware. Never silently switch simulator or use NTFS as runtime.

## 8. Windows preparation checklist

- [ ] Place this audit, smoke runbook, TASK.md, expected-evidence checklist, and manifest template under D:\DAMH-Bridge\outgoing\s1_1_<timestamp>.
- [ ] Create repos.lock with URL, outer Panthera SHA, vendored RoboTwin tree hash, reference standalone RoboTwin SHA, retrieval date, and relevant file/config hashes.
- [ ] Prepare GPU sampler, time wrapper, Vulkan probe, package inventory, HDF5/video checks.
- [ ] Require GPU/Vulkan output, packages, SHAs, assets, two decodable videos, HDF5 schema, seed, peak VRAM/RAM/FPS, and full log/failure classification.
- [ ] Do not create a camera/config overlay until exact field names are inspected in pinned checkout.
- [ ] Bridge only portable instructions/logs/metrics/patches/configs/checkpoints. Ext4 hosts clone/build/assets/Conda.
- [ ] Return a portable archive into D:\DAMH-Bridge\incoming\s1_1_<timestamp>.

## 9. Risk register and next task bundle

**Blocking:** native driver/Vulkan/SM 7.5 may not serve PyTorch and CuRobo; CuRobo may not compile/fit beside two-camera rendering; incomplete assets/editable path may masquerade as compatibility failure.

**Non-blocking:** viewer/ray-tracing failure after headless proof; omitted PyTorch3D; ROS 2, policy conversion/training, local VLM, Docker, point cloud, and dual arm are not S1.1 gates; an individual task failure is not general stack failure.

**Exact next bundle:** boot Ubuntu 22.04; use ext4; run the smoke runbook unchanged through the official single-arm command; archive logs/telemetry/manifest/checksums/full episode; copy only portable archive to D:\DAMH-Bridge\incoming.

**Smallest empirical questions:** Does target PyTorch see SM 7.5 with CUDA/nvcc 12.8 aligned? Does SAPIEN offscreen use the NVIDIA Vulkan ICD? Does editable CuRobo initialize Panthera? What are peak VRAM/RAM and camera FPS? If OOM, does MPlib/TOPPRA preserve full 6-DoF in canonical RoboTwin?

## Primary sources

* [Panthera platform](https://github.com/HighTorque-Robotics/Panthera-HT_Main)
* [Panthera RoboTwin integration](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin) and [single-arm task verification](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin/blob/main/RoboTwin/task_config/README.md)
* [RoboTwin install documentation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
* [SAPIEN installation/offscreen test](https://sapien-sim.github.io/docs/user_guide/getting_started/installation.html)
* [CuRobo v0.7.8](https://github.com/NVlabs/curobo/tree/v0.7.8)
* [NVIDIA CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), [architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html), and [compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
* [PyTorch 2.7.1 matrix](https://pytorch.org/get-started/previous-versions/)
* [Show-Harness simulator documentation](https://github.com/showlab/Show-Harness/blob/main/docs/simulators.md)
