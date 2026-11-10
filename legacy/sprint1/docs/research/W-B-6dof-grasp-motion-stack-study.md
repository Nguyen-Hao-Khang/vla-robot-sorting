# W-B — 6-DoF Grasp-to-Motion Stack Study

**Workstream:** S1.2–S1.3 engineering research

**Status:** Decision-ready, pending the Ubuntu validation spikes in this document

**Date:** 2026-10-02
**Scope guardrail:** This study retains Panthera-HT plus RoboTwin as the canonical embodiment and simulator. It does not alter the locked Sprint 1 architecture, object-specific 6-DoF grasp requirement, continuous collision-aware motion requirement, or closed-loop/receding-horizon execution requirement.

## 1. Executive recommendation

Use a project-owned, geometry-first parallel-jaw candidate generator and measurable ranker as the Sprint 1 primary grasp path. It produces full object-specific SE(3) grasp poses and is feasible to validate on the target GTX 1660 Ti 6 GB without adding a second large learned-model runtime. In controlled RoboTwin experiments, simulator object pose and collision geometry may initialize the generator only when the experiment manifest records:

~~~yaml
geometry_source: sim_ground_truth
privileged_geometry: true
evaluation_claim: simulation-control-oracle
~~~

This is a simulation control condition, not an end-to-end perception result. The same project-owned interface must subsequently accept estimated object geometry.

Use pinned CuRobo v0.7.8 as the preferred IK and motion-planning adapter only if B1 and B2 validate import, collision behaviour, and memory coexistence on the target laptop. The executable fallback is MPlib 0.2.1 with the same project-owned data contracts and a collision recheck in RoboTwin/SAPIEN. Parameterize accepted geometric joint paths with TOPPRA 0.6.9 for velocity- and acceleration-bounded timing; a project-owned jerk measurement gate and duration dilation closes the gap that TOPPRA does not claim to solve.

Do not make AnyGrasp, GraspNet-baseline, or Contact-GraspNet a Sprint 1 runtime dependency. AnyGrasp is license- and binary-gated; GraspNet-baseline is an older, separate CUDA/PyTorch environment; Contact-GraspNet documents an 8 GB-or-greater inference expectation. They remain controlled comparison tracks after the canonical stack is working.

## 2. Evidence policy and locked pins

The repository lock is authoritative for the project integration point. The vendored RoboTwin directory is ordinary content in Panthera-HT_RoboTwin, not an independently checked-out Git submodule. Record the outer integration commit and the vendored tree hash together.

| Component | Project pin / evidence | Authority and role |
|---|---|---|
| Panthera integration | Panthera-HT_RoboTwin commit 2aee31ad75cb7b0cd779540441c6025853a5b60d | Project-pinned integration repository; canonical robot plus RoboTwin glue. |
| Vendored RoboTwin tree | tree 3b0e2538a956c4c10c61c9311646a8d38e576598 | Reproducibility identifier for vendored content; do not run rev-parse inside the vendored directory. |
| RoboTwin upstream reference | RoboTwin ea8b211 | Upstream task and simulator documentation reference. |
| Panthera main / SDK / ROS 2 references | cd5e, 3712, afcdc | Robot model, SDK, and ROS 2 reference material respectively; ROS 2 is not a first-smoke prerequisite. |
| CuRobo | v0.7.8, d64c4b005459db10c5dd867d8b30a87d5bda9bdb | Required project pin if CuRobo adapter is selected; use its v1-era API, not current main documentation as a compatibility promise. |
| MPlib | 0.2.1 | RoboTwin-native planning path named by the integration material; validate exact wheel/source compatibility. |
| TOPPRA | 0.6.9 | MIT upstream package used only for time parameterization. |

Every experiment manifest shall contain the integration commit, vendored tree hash, adapter name/version, container or Conda environment export hash, configuration hash, random seed, scene/task identifier, and generated artifact hashes.

## 3. Candidate grasp technologies

### 3.1 Comparison

| Technology | Primary-source status and maintenance posture | Inputs and outputs | License / integration implications | GTX 1660 Ti assessment | Sprint 1 disposition |
|---|---|---|---|---|---|
| Project geometry-first generator | Project-owned; maintenance is controlled by this project and versioned with the experiment manifest. | Calibrated object pose plus mesh/proxy and gripper geometry; emits SE(3), jaw width, approach, clearance and provenance | Project-controlled. No model weights or third-party runtime. | Low incremental GPU use; collision queries may run CPU or planner GPU. | **Primary.** |
| [GraspNet-baseline](https://github.com/graspnet/graspnet-baseline) | graspnet/graspnet-baseline at 280c215129f759ed8649cb4e89fc5dfee55f4f80; freeze this commit because upstream compatibility maintenance is not a Sprint 1 assumption. | Point cloud; grasp group includes translation, rotation matrix, width and score; [upstream API](https://github.com/graspnet/graspnetAPI) provides NMS and collision filtering utilities. | Repository [license](https://github.com/graspnet/graspnet-baseline/blob/main/LICENSE) declares research/non-commercial terms. README specifies Python 3, PyTorch 1.6, Open3D and custom PointNet2/KNN compilation. | No upstream 6 GB commitment; isolate into a legacy environment and measure. | Optional research comparator, never required for first bring-up. |
| [AnyGrasp SDK](https://github.com/graspnet/anygrasp_sdk) | graspnet/anygrasp_sdk at b8eaafc9eca7babd5208e7a5ade3c561060be4c5; vendor-controlled binary SDK, so upgrades require a new license and regression check. | Point cloud/depth-derived input; candidates, scores and region/approach steering according to SDK README. | Binary SDK. Its README requires feature-ID license/weight access. Technical availability and license must be verified before use. | No primary 6 GB guarantee. MinkowskiEngine and rendering must not be assumed to coexist. | Optional only after license and B6 evidence. |
| [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) | NVlabs/contact_graspnet at 721706067cc7f7889c32afe7b4f7955eb90bd866; its documented TensorFlow 2.2-era environment means upstream compatibility is not assumed. | Raw point cloud, or depth in metres plus intrinsics; optional segmentation/cropped regions for object-wise grasps. | [NVIDIA Source Code license](https://github.com/NVlabs/contact_graspnet/blob/master/LICENSE.md), non-commercial. README states tested Python 3.7, TensorFlow 2.2, CUDA 11.1. | README says one NVIDIA GPU of at least 8 GB for inference, although it may work with less. | Not a target-laptop primary candidate. |

Verified facts in this table come from the linked upstream READMEs/API repositories. The selection statements are engineering inferences for the locked machine and must be confirmed by B6 before a learned path is enabled.

### 3.2 Geometry-first generator

The generator starts from a target object frame and collision geometry. It samples physically meaningful contact pairs on opposing surfaces, aligns the gripper closing direction with their normal-aware span, and samples non-top-down approach rotations about the closing axis. It rejects candidates that violate jaw opening, fingertip/hand clearance, table clearance, approach corridor clearance, reachability prefilter, or object identity constraints.

Each accepted candidate retains its exact provenance: source geometry, object-frame transform, calibrated camera-to-world and world-to-robot transforms, sampled contact references, and the collision scene version. Generation never converts a pose to only XYZ or Euler angles.

For a depth-based later path, reconstruct point geometry only after the head/wrist camera intrinsics and extrinsics pass the smoke-test calibration checks. A camera timestamp and transform timestamp must be carried with every candidate; stale geometry is a planning failure, not a silent fallback.

### 3.3 Candidate ranking

The ranker normalizes each feasible candidate within one scene version:

~~~text
score = 0.28*reachability
      + 0.22*clearance_margin
      + 0.18*grasp_quality
      + 0.15*approach_clearance
      + 0.10*motion_cost
      + 0.07*object_identity_confidence
~~~

The initially fixed weights are a pre-registration, not a learned assertion. Log every raw factor and the normalization range. A candidate is eligible only if:

1. object identity confidence is at least the scenario threshold;
2. IK and collision checks pass for pre-grasp and grasp poses;
3. approach corridor clearance exceeds the configured margin;
4. the planner reports a path meeting the timing and jerk gates; and
5. the scene/transform version is still current at execution.

The benchmark reports success by candidate rank, score calibration versus final grasp success, clearance, time-to-first-feasible, and failure reason. This makes ranker changes comparable without conflating semantic selection, grasp geometry, and motion planning.

## 4. Planning and timing technology assessment

### 4.1 CuRobo v0.7.8

[CuRobo](https://github.com/NVlabs/curobo) is Apache-2.0 and provides GPU-oriented IK, collision checking and motion generation. The project pin is v0.7.8; current main documentation includes newer API material and must not be treated as proof that this pin works with the target stack. Its candidate contract needs robot joint state, robot model, target SE(3), joint limits, collision-world geometry, and optional warm-start state. It must return either a typed success result with a joint trajectory and collision-scene version, or a typed failure result such as IK failure, optimization failure, collision rejection, timeout, or invalid input.

The primary risk is shared VRAM: SAPIEN Vulkan rendering, depth/point-cloud processing, CuRobo CUDA tensors and any learned model can compete in the same 6 GB budget. Validate after renderer warm-up and with both required cameras active. Use a single process, single scene, one active target, one candidate batch at a time, one planning query at a time, and no concurrent learned inference. Cache robot/world static geometry and planner instances after warm-up. Allocate a hard admission rule from observed free memory; the exact threshold is empirical B2 evidence, not a documented upstream value.

### 4.2 MPlib 0.2.1

[MPlib](https://github.com/haosulab/MPlib) is the RoboTwin-native fallback path. Its documented planner APIs support pose IK, joint-space plans, collision checking, joint limits, and a time-step argument for time-optimal path parameterization. It can be used from project adapters without exporting third-party objects beyond the boundary.

Use MPlib if CuRobo does not install, fails at 6 GB under the canonical renderer, or cannot demonstrate collision-safe, bounded-latency replanning. MPlib is not a scope reduction: the generator remains full SE(3), motion remains collision aware and timing remains measurable. Recheck every selected path in the currently active RoboTwin/SAPIEN collision scene before execution.

MPlib is MIT-licensed. Its upstream README documents pre-built pip packages for Ubuntu 20.04 or later and Python 3.8 or later; the precise 0.2.1 wheel/source combination and its compatibility with the selected simulator environment remain B3 evidence, not an assumption.

### 4.3 TOPPRA 0.6.9 and execution timing

[TOPPRA](https://github.com/hungpham2511/toppra) is MIT-licensed and calculates a time parameterization for a fixed geometric path under velocity and acceleration constraints. It is not a jerk-constrained trajectory generator. Feed it a collision-cleared C2 joint path, project Panda/Panthera joint velocity and acceleration limits into the parameterizer, then uniformly dilate duration or refit a local C2/quintic segment if finite-difference jerk exceeds the project gate.

The ExecutionSegment carries position, velocity, acceleration, time-from-start, segment start state, scene version, controller sample period and predicted stop state. The robot interface samples or interpolates only inside the locally approved segment. Long plans must be cut into a short horizon; after each observation/control checkpoint the system validates scene and state versions, estimates tracking error and replans when the threshold is crossed. No precomputed full trajectory may be blindly executed.

## 5. Stable project-owned interfaces

No external class from CuRobo, MPlib, SAPIEN, GraspNet or an SDK crosses a project boundary.

~~~python
@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    object_id: str
    grasp_pose: Pose6D             # xyz metres, xyzw quaternion, explicit frame
    pregrasp_pose: Pose6D
    approach_direction: Vector3    # unit vector in grasp_pose.frame_id
    jaw_width_m: float
    contact_points: tuple[Vector3, Vector3] | None
    source: str                    # geometry_first, graspnet, anygrasp, ...
    geometry_source: str           # sim_ground_truth, reconstructed_depth, ...
    scene_version: str
    transform_version: str
    raw_quality: float | None
    metadata: Mapping[str, JsonValue]

@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    total: float
    factors: Mapping[str, float]
    eligible: bool
    rejection_reasons: tuple[str, ...]

class GraspGenerator(Protocol):
    def generate(self, request: GraspRequest) -> Result[list[GraspCandidate]]: ...

class GraspRanker(Protocol):
    def rank(self, candidates: Sequence[GraspCandidate],
             snapshot: PlanningSnapshot) -> Result[list[CandidateScore]]: ...

class IKSolver(Protocol):
    def solve(self, target: Pose6D, snapshot: PlanningSnapshot) -> Result[IKResult]: ...

class MotionPlanner(Protocol):
    def plan(self, request: MotionRequest,
             snapshot: PlanningSnapshot) -> Result[GeometricTrajectory]: ...

class Parameterizer(Protocol):
    def parameterize(self, path: GeometricTrajectory,
                     limits: DynamicLimits) -> Result[Trajectory]: ...
~~~

Trajectory stores a named joint order, sample times in seconds, position, velocity and acceleration arrays, limits used, interpolation type, maximum measured jerk, scene version, and planner provenance. A parameterizer cannot alter joint ordering, frame identity, quaternion convention, or source scene version.

All Result values are explicit discriminated values: success, invalid_input, transform_stale, no_candidate, collision_rejected, ik_failed, planning_timeout, optimization_failed, dynamic_limit_failed, tracking_error, execution_stale, backend_unavailable, and internal_error. They carry machine-readable diagnostics plus a bounded human-readable message. This extends the existing project Pose6D and planning contracts rather than replacing them.

## 6. Canonical low-memory runtime configuration

| Area | First configuration | Reason |
|---|---|---|
| Robot | One Panthera arm; no dual-arm planning | Retains the requested single-arm scope and reduces state/collision work. |
| Cameras | Head RGB-D plus one wrist RGB-D; 640 by 480, one active depth stream per planning observation | Preserves required perception views; avoids high-resolution multi-camera pressure. |
| Renderer | Prefer headless Vulkan/EGL-supported path established by W-A; no viewer-dependent logic | Allows reproducible Linux gates and minimizes desktop compositing contention. |
| Scene | Table, destination bin, target, and only necessary distractors; simple collision proxies where validated | Keeps object-specific 6-DoF and obstacle checking while bounding collision complexity. |
| Point clouds | Disabled by default for first planning smoke; construct only bounded ROI clouds when learned comparison is enabled | Geometry-first simulation control uses object collision geometry; protects VRAM. |
| CuRobo | One warmed planner, batch size one, sequential candidate evaluation, static collision cache | Avoids unmeasured allocation growth. |
| Timing | TOPPRA with conservative robot limits; 100 Hz controller sample unless controller documentation dictates otherwise | Makes velocity, acceleration, jerk and interpolation auditable. |
| Logging | Structured JSONL plus bounded per-trial artifacts, no raw video by default | Preserves traceability without disk and CPU overhead. |

Mixed precision and reduced batch size matter only for a learned comparison or CuRobo tensors after B1 proves numerical correctness. They are not a blanket remedy: mixed precision must pass pose, collision-margin, and success-agreement checks against a reference mode before activation.

## 7. Benchmark protocol

Pre-register 100 deterministic grasp-and-place trials: 20 each in five strata: unobstructed, near-bin edge, non-top-down object orientation, mild clutter, and visually ambiguous distractor. Use fixed task/scene seeds, fixed integration and vendored-tree hashes, configuration hash, and a documented reset procedure. Run 10 warm-up trials and exclude them from steady-state timing summaries while retaining their telemetry.

For closed-loop testing, run 60 deterministic replanning perturbations: 20 target-pose changes, 20 destination-pose changes, and 20 obstacle/scene updates. Apply the perturbation only at a named checkpoint, record the observed state and timestamp, then measure time to a newly collision-cleared accepted segment.

Report:

| Category | Required metrics |
|---|---|
| Grasp | target identity correctness, grasp success, grasp pose translation/rotation error where ground truth exists, jaw width validity, contact/clearance margin |
| Planning | candidate count, first-feasible latency, plan latency p50/p95, IK success, collision-free path rate, failure code distribution |
| Motion | path length, duration, max velocity/acceleration/finite-difference jerk, tracking error, min static and dynamic clearance, segment splice discontinuity |
| Closed loop | replan latency p50/p95, stale-segment rejections, perturbation recovery rate, post-replan grasp/place success |
| Resources | process RSS, GPU memory used/free before and after renderer/planner/cameras, GPU utilization, FPS and frame latency |
| Reproducibility | all identifiers above, software manifests, seed, source/transform/scene versions, artifact checksums |

No result may call itself vision-based if it used simulator ground-truth geometry. Report that condition separately.

## 8. Minimum Ubuntu validation spikes

Do not request a Ubuntu reboot merely to read this document. These are the smallest task bundle to run once the user schedules a native Ubuntu gate.

| Spike | Hypothesis | Exact test and evidence | Pass / fail criterion | Decision |
|---|---|---|---|---|
| B1 CuRobo install and warm-up | Pinned v0.7.8 imports and solves one fixed pose with Panthera data | Create isolated environment, record pip/conda lock, import CuRobo, build robot/collision world, execute 10 warm-up IK/plan calls and save status, latency, nvidia-smi memory | Pass: all calls return typed results, at least one feasible collision-free fixed-pose result, no unbounded allocation growth. Fail: install/API/driver failure or no feasible baseline. | Enable or reject CuRobo primary investigation. |
| B2 Coexistence memory | Canonical low-memory renderer plus both cameras plus CuRobo fits reliably | Launch headless Panthera/RoboTwin scene, start head+wrist RGB-D at 640x480, warm 30 frames, run 30 sequential planning calls; capture GPU memory and FPS at each phase | Pass: no OOM/device loss, stable memory after warm-up, and project-defined usable FPS/latency recorded. Fail: OOM, renderer reset, or escalating memory. | Keep CuRobo or move to MPlib. |
| B3 MPlib path | MPlib 0.2.1 executes the same fixed SE(3) request and collision checks | Isolated environment; run fixed pregrasp/grasp IK, one joint plan, collision recheck, TOPPRA timing; save joint path and diagnostics | Pass: typed results plus collision-cleared timed path. Fail: unavailable/incompatible package or collision mismatch. | Validate executable fallback. |
| B4 Timing and splice | Parameterization meets declared velocity/acceleration and jerk gates | Parameterize one representative path, finite-difference samples, then splice replan segment at a checkpoint; persist maxima and position/velocity discontinuities | Pass: limits and jerk gate met after allowed duration dilation; no unsafe discontinuity. | Accept timing adapter or adjust conservative limits/interpolation. |
| B5 Geometry-first grasp | Generator produces reachable full 6-DoF non-top-down candidates | In a fixed one-object scene, generate 100 candidates, rank, solve pregrasp+grasp IK and collision check; record rejection histogram | Pass: at least one feasible candidate and all output fields/provenance populated. | Enable primary grasp path. |
| B6 Learned comparator admission | A chosen learned package can coexist without destabilizing canonical runtime | Run only one selected model in its own environment, one RGB-D frame/ROI, batch one; record license evidence, imports, memory and candidate conversion | Pass: legal access, correct SE(3) conversion, no OOM, measurable candidate value. | Admit only as optional comparator. |
| B7 Receding-horizon recovery | Planner can replace a stale short segment after a perturbation | During a fixed approach, move target/obstacle state at a checkpoint, invalidate scene version, reobserve, replan and execute only new segment in simulation | Pass: stale segment rejected and recovery success recorded. | Enable closed-loop execution gate. |

Each spike produces a JSON result, raw console log, configuration, package export, manifest, and telemetry CSV in the bridge-compatible results bundle. A failed spike is evidence, not a reason to swap simulators.

## 9. Fallback ladder

1. Keep Panthera plus RoboTwin and reduce to the canonical low-memory configuration: one arm, two 640x480 RGB-D cameras, simple scene, bounded ROI, sequential planning.
2. Disable optional visual features: viewer, raw video capture, high-resolution cameras, dense global point clouds, learned inference concurrency, unnecessary distractors.
3. Preserve full SE(3) and use MPlib 0.2.1 plus RoboTwin/SAPIEN collision recheck and TOPPRA timing if CuRobo is not validated.
4. Where scientifically acceptable, run offline CPU geometry generation/ranking and CPU planning for deterministic analysis; do not represent offline throughput as closed-loop performance.
5. Use a separate compatible Linux/NVIDIA host only after the above evidence shows the 6 GB laptop cannot meet the pre-registered stability or latency gate. Preserve the same pins, contracts, benchmark and canonical simulator.

## 10. Blocking and non-blocking risks

### Blocking until measured

- Whether CuRobo v0.7.8 installs and runs against the Panthera integration on the exact Ubuntu, driver, CUDA and PyTorch combination.
- Whether Vulkan/SAPIEN rendering, two depth cameras, and CuRobo coexist within 6 GB after warm-up.
- Availability and exact package compatibility of MPlib 0.2.1 on the selected environment.
- Robot-model/joint-order, collision geometry, controller limits and camera extrinsic correspondence in the installed Panthera integration.
- A measurable safe segment-splice gate for receding-horizon execution.

### Non-blocking

- Learned grasp comparison selection. The primary geometry-first path does not require it.
- AnyGrasp license access and feature-ID activation.
- Whether mixed precision improves a future comparator.
- ROS 2 integration. It is not required for the simulation planning smoke or S1.2 decision.

## 11. Primary sources

- [Panthera-HT_RoboTwin project pin](../../configs/s1.1/repos.lock.yaml)
- [CuRobo repository and documentation](https://github.com/NVlabs/curobo)
- [MPlib repository](https://github.com/haosulab/MPlib)
- [TOPPRA repository](https://github.com/hungpham2511/toppra)
- [GraspNet-baseline repository](https://github.com/graspnet/graspnet-baseline)
- [GraspNet API repository](https://github.com/graspnet/graspnetAPI)
- [AnyGrasp SDK repository](https://github.com/graspnet/anygrasp_sdk)
- [Contact-GraspNet repository](https://github.com/NVlabs/contact_graspnet)
- [RoboTwin repository](https://github.com/robotwin-platform/RoboTwin)

Primary source statements above are deliberately limited to source/docs claims. All target-laptop suitability, ordering, score weights, and pass gates are engineering proposals pending the listed Ubuntu evidence.
