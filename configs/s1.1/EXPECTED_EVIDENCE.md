# S1.1 Expected Evidence

A PASS requires evidence for every mandatory item below.

## Host/runtime

- [ ] Ubuntu release and kernel captured.
- [ ] `nvidia-smi` succeeds and identifies GTX 1660 Ti with memory size.
- [ ] NVIDIA kernel driver is in use.
- [ ] Vulkan summary succeeds and identifies NVIDIA device/ICD.
- [ ] Python/Conda paths and versions captured.
- [ ] PyTorch reports CUDA 12.8 and `torch.cuda.is_available() == True`.
- [ ] PyTorch reports compute capability `(7, 5)`.
- [ ] `nvcc --version` captured and compatible with the pinned build.
- [ ] `pip check` passes.

## Source/provenance

- [ ] Panthera-HT_RoboTwin outer SHA is `2aee31ad75cb7b0cd779540441c6025853a5b60d`.
- [ ] Vendored `RoboTwin/` tree hash is `3b0e2538a956c4c10c61c9311646a8d38e576598`.
- [ ] CuRobo v0.7.8 checkout is `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`.
- [ ] Checkout working tree status captured.
- [ ] Conda explicit list and pip freeze captured.
- [ ] Panthera asset presence checks pass.

## Rendering/simulation

- [ ] SAPIEN offscreen test produces non-empty image.
- [ ] No blocking Vulkan/SAPIEN exception.
- [ ] Project-owned observation probe records both `head_camera` and `wrist_camera` RGB dimensions.
- [ ] Observation evidence records the resource-bounded render profile (`1 spp`, path depth `2`, denoiser `none`).
- [ ] Per-camera progress proves which wrist/head capture operations completed.
- [ ] Project-owned observation probe records six actual Panthera qpos values from `Robot.get_left_arm_real_jointState()`.
- [ ] Vendor drive-target vector is kept distinct from actual qpos and is not used as proprioception evidence.
- [ ] Single-arm setup reports one articulation and one wrist camera.
- [ ] One single-arm Panthera smoke episode completes.
- [ ] No GPU OOM.
- [ ] CuRobo planner initializes and participates in the canonical smoke, or the failure is clearly recorded for fallback evaluation.

## Episode artifacts

- [ ] HDF5 episode exists and is non-empty.
- [ ] HDF5 dataset names/shapes are listed.
- [ ] Dense trajectory artifact exists.
- [ ] Seed exists.
- [ ] Scene information exists.
- [ ] Instruction JSON exists.
- [ ] Head-camera video is non-empty and decodable.
- [ ] Wrist-camera video is non-empty and decodable.
- [ ] Video resolution/FPS/frame count captured.

## Performance/diagnostics

- [ ] Per-second GPU utilization/VRAM telemetry captured during the observation gate and canonical episode.
- [ ] SAPIEN rendering-device information is captured when the installed CLI exposes it.
- [ ] Peak VRAM derived or visible from telemetry.
- [ ] GNU time output captures elapsed time and max host RSS.
- [ ] Terminal smoke log preserved.
- [ ] Evidence files/checksums generated.
- [ ] Result bundle SHA256 generated.

## Result classification

Choose exactly one:

- [ ] **PASS** — all mandatory gates passed.
- [ ] **CONDITIONAL PASS** — core canonical stack runs but resource headroom/performance requires a measured lower-memory overlay.
- [ ] **CUROBO BLOCKED / ROBOTWIN PASS** — SAPIEN/Panthera/cameras run, but CuRobo fails or OOMs; evaluate MPlib+TOPPRA fallback without changing simulator.
- [ ] **HOST RUNTIME BLOCKED** — driver/Vulkan/SAPIEN fails; repair host runtime before interpreting RoboTwin compatibility.
- [ ] **OTHER FAIL** — describe exact first failing gate and attach evidence.

Do not label the laptop compatible without a PASS or explicitly qualified conditional result.
