#!/usr/bin/env bash
set -u -o pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"
ROBOTWIN_ROOT="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run the manipulation campaign as root."
  exit 7
fi
if ! findmnt -rn "$BRIDGE_MOUNT" >/dev/null 2>&1; then
  echo "DATA is not mounted at $BRIDGE_MOUNT."
  exit 2
fi
if [[ ! -w "$BRIDGE/incoming" ]]; then
  echo "$BRIDGE/incoming is not writable."
  exit 8
fi

SESSION_NAME="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
SESSION_DIR="$BRIDGE/outgoing/$SESSION_NAME"
(
  cd "$SESSION_DIR" || exit 4
  sha256sum -c SHA256SUMS.txt
) || exit 4

bash "$SESSION_DIR/bootstrap_from_bridge.sh" || exit 5
cd "$PROJECT_ROOT" || exit 6

CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
fi
if [[ -z "$CONDA_EXE" ]]; then
  echo "conda executable not found"
  exit 90
fi

EXPECTED_HEAD="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["git_head"])
PY
)"
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "HEAD mismatch after bootstrap: expected=$EXPECTED_HEAD actual=$ACTUAL_HEAD"
  exit 91
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="experiments/results/sprint1_manipulation_campaign_$STAMP"
EVIDENCE="$HOME/damh-evidence/sprint1_manipulation_campaign_$STAMP"
mkdir -p "$RESULT_DIR" "$EVIDENCE"

git status --short --branch > "$RESULT_DIR/git-status-before.txt"
git rev-parse HEAD > "$RESULT_DIR/git-head.txt"
nvidia-smi > "$RESULT_DIR/nvidia-smi-start.txt"
nvidia-smi   --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total   --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>&1 &
GPU_PID=$!

run_native() {
  local label="$1"
  local timeout_s="$2"
  shift 2
  echo
  echo "=== $label ==="
  /usr/bin/time -v -o "$RESULT_DIR/${label}-time.txt"     timeout --signal=TERM --kill-after=30s "${timeout_s}s"     "$@" > >(tee "$RESULT_DIR/${label}.log") 2>&1
  local rc=$?
  printf '%s=%s\n' "$label" "$rc" >> "$RESULT_DIR/stage-return-codes.txt"
  echo "$label rc=$rc"
  return "$rc"
}

echo "=== Accepted prerequisite manifests ==="
python3 - <<'PY'
import json, pathlib
root = pathlib.Path("experiments/accepted")
requirements = {
    "native_s1_3_20261007.json": "S1.3",
    "native_s1_6_integrated_20261007.json": "S1.6-INTEGRATED",
}
for name, gate in requirements.items():
    path = root / name
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("schema") == "accepted-native-gate-v1", path
    assert data.get("gate_id") == gate, (path, data.get("gate_id"))
    assert data.get("classification") in {"PASS", "PASS_AUDITED"}, path
    print(f"{gate}: {data['classification']} ({path})")
PY
PREREQ_RC=$?

SYNC_RC=98
MANIP_PASS_COUNT=0
MANIP_AUDIT_PASS_COUNT=0
READINESS_RC=98
CLOSURE_CANDIDATE_RC=98
NATIVE_CLOSURE_RC=98

echo
echo "### Stage A - target-aware MPlib smoke"
run_native "target-sync-seed-0" 600   "$CONDA_EXE" run -n RoboTwin python -u   scripts/linux/probe_panthera_mplib_target_sync.py   --robotwin-root "$ROBOTWIN_ROOT"   --project-root "$PROJECT_ROOT"   --output "$RESULT_DIR/target-sync-seed-0.json"   --seed 0
SYNC_RC=$?

echo
echo "### Stage B - project-owned manipulation seeds 0,1,2"
if [[ "$PREREQ_RC" -eq 0 && "$SYNC_RC" -eq 0 ]]; then
for seed in 0 1 2; do
  label="manipulation-seed-${seed}"
  run_native "$label" 3600     "$CONDA_EXE" run -n RoboTwin python -u     scripts/linux/probe_panthera_project_manipulation.py     --robotwin-root "$ROBOTWIN_ROOT"     --project-root "$PROJECT_ROOT"     --output "$RESULT_DIR/${label}.json"     --seed "$seed"
  rc=$?
  audit_rc=99

  if [[ -f "$RESULT_DIR/${label}.json" ]]; then
    python3 - "$RESULT_DIR/${label}.json"       > "$RESULT_DIR/${label}-phase-summary.txt" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
print(f"passed={data.get('passed')}")
print(f"error={data.get('error')}")
print(f"failure_phase={data.get('failure_phase')}")
print(f"final_phase={data.get('final_phase')}")
print(f"task_success_after_release={data.get('task_success_after_release')}")
print(f"task_success_final={data.get('task_success_final')}")
print(f"mplib_target_present={(data.get('mplib_world_initial') or {}).get('target_present')}")
print(f"B5_passed={(data.get('B5') or {}).get('passed')}")
attachment = data.get("carried_object_attachment") or {}
print(f"support_contact_geometry={attachment.get('support_contact_geometry')}")
print(f"carried_start_state={data.get('carried_start_state_diagnostic')}")
table_diag = data.get("lift_table_disabled_diagnostic") or {}
support_escape = data.get("support_escape_diagnostic") or {}
print(
    "support_escape="
    f"triggered:{support_escape.get('triggered')} "
    f"passed:{support_escape.get('passed')} "
    f"requested:{support_escape.get('requested_escape_m')} "
    f"required:{support_escape.get('required_clearance_m')} "
    f"object_dz:{support_escape.get('object_delta_z_m')} "
    f"post_start:{support_escape.get('post_start_state')} "
    f"fingers:{support_escape.get('finger_contacts_after')} "
    f"contacts:{support_escape.get('contacts')}"
)
print(
    "lift_table_disabled="
    f"disabled:{table_diag.get('obstacle_disabled')} "
    f"reenabled:{table_diag.get('obstacle_reenabled')} "
    f"start:{table_diag.get('start_state')} "
    f"plan_success:{table_diag.get('plan_success')} "
    f"native_status:{table_diag.get('native_status')} "
    f"path_samples:{table_diag.get('path_samples')} "
    f"executed:{table_diag.get('executed')}"
)
for index, stage in enumerate(data.get("stages") or []):
    if not isinstance(stage, dict):
        continue
    contacts = stage.get("contacts") or {}
    collision = stage.get("collision_validation") or {}
    failure = stage.get("failure") or {}
    failure_details = failure.get("details") or {}
    native_diag = failure_details.get("native_diagnostics") or {}
    native_start = native_diag.get("start_state") or {}
    planner_policy = stage.get("planner_policy") or {}
    print(
        f"stage[{index}]={stage.get('stage')} "
        f"passed={stage.get('passed')} "
        f"planner={stage.get('planner_id')} "
        f"plan={stage.get('plan_success')} "
        f"dynamic={stage.get('dynamic_safe')} "
        f"collision_safe={stage.get('collision_safe')} "
        f"execution={stage.get('execution_status')} "
        f"tracking={stage.get('max_tracking_error_rad')} "
        f"robot_target={contacts.get('robot_target_contact_frames')} "
        f"target_pad={contacts.get('target_pad_contact_frames')} "
        f"target_table={contacts.get('target_table_contact_frames')} "
        f"robot_table={contacts.get('robot_table_contact_frames')} "
        f"failure_code={failure.get('code')} "
        f"native_status={failure_details.get('native_status')} "
        f"native_valid_query={native_diag.get('valid_query')} "
        f"native_start_valid={native_start.get('valid')} "
        f"native_start_status={native_start.get('status')} "
        f"planner_strategy={planner_policy.get('strategy')} "
        f"selected_planner={planner_policy.get('selected_planner_id')} "
        f"primary_success={planner_policy.get('primary_success')} "
        f"primary_vendor={planner_policy.get('primary_vendor_status')} "
        f"fallback_attempted={planner_policy.get('fallback_attempted')} "
        f"fallback_success={planner_policy.get('fallback_success')} "
        f"policy={stage.get('contact_policy')} "
        f"object_dz={stage.get('object_delta_z_m')}"
    )
    if collision.get("collisions"):
        first = collision["collisions"][0]
        print(
            "  first_collision="
            f"waypoint:{first.get('waypoint_index')} "
            f"roles:{first.get('matched_environment_roles')}"
        )
    if stage.get("stage") in {"close_gripper", "release"}:
        print(f"  result={stage.get('result')}")
        print(f"  contacts_after={stage.get('contacts_after')}")
PY
    cat "$RESULT_DIR/${label}-phase-summary.txt"
  fi

  if [[ "$rc" -eq 0 ]]; then
    MANIP_PASS_COUNT=$((MANIP_PASS_COUNT + 1))
    PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin       python scripts/audit_project_manipulation.py "$RESULT_DIR/${label}.json"       > >(tee "$RESULT_DIR/${label}-audit.log") 2>&1
    audit_rc=$?
    if [[ "$audit_rc" -eq 0 ]]; then
      MANIP_AUDIT_PASS_COUNT=$((MANIP_AUDIT_PASS_COUNT + 1))
    fi
  fi
  printf 'manipulation-seed-%s-audit=%s\n' "$seed" "$audit_rc"     >> "$RESULT_DIR/stage-return-codes.txt"
done
else
  echo "Manipulation seeds skipped because accepted prerequisites or target-sync smoke failed."
fi


RECOVERY_DIAG_COUNT=0
RECOVERY_PASS_COUNT=0
SEED1_RECOVERY_ACCEPTED=0
SEED1_EFFECTIVE_LABEL="manipulation-seed-1"
SEED1_EFFECTIVE_CANDIDATE=""

echo
echo "### Stage B2 - seed1 deterministic acquisition recovery"
SEED1_JSON="$RESULT_DIR/manipulation-seed-1.json"
SEED1_NEEDS_RECOVERY=1
if [[ -f "$SEED1_JSON" ]]; then
  python3 - "$SEED1_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
approach = next(
    (
        stage for stage in data.get("stages", [])
        if isinstance(stage, dict) and stage.get("stage") == "approach"
    ),
    None,
)
policy = (
    approach.get("planner_policy")
    if isinstance(approach, dict) and isinstance(approach.get("planner_policy"), dict)
    else {}
)
failure = approach.get("failure") if isinstance(approach, dict) else None
collision = approach.get("collision_validation") if isinstance(approach, dict) else None
collisions = collision.get("collisions") if isinstance(collision, dict) else None
contacts = approach.get("contacts") if isinstance(approach, dict) else None
contact_policy = (
    approach.get("contact_policy")
    if isinstance(approach, dict) and isinstance(approach.get("contact_policy"), dict)
    else {}
)
target_collision_rejected = (
    isinstance(approach, dict)
    and approach.get("plan_success") is True
    and approach.get("collision_safe") is False
    and policy.get("fallback_success") is True
    and isinstance(collisions, list)
    and any(
        isinstance(row, dict)
        and "target" in (row.get("matched_environment_roles") or [])
        for row in collisions
    )
)
fallback_planning_failed = (
    policy.get("fallback_success") is False
    and isinstance(failure, dict)
    and failure.get("code") == "optimization_failed"
)
execution_target_contact_rejected = (
    isinstance(approach, dict)
    and approach.get("plan_success") is True
    and approach.get("dynamic_safe") is True
    and approach.get("collision_safe") is True
    and approach.get("execution_status") == "completed"
    and isinstance(contacts, dict)
    and int(contacts.get("robot_target_contact_frames", 0)) > 0
    and isinstance(contact_policy, dict)
    and contact_policy.get("allow_robot_target_contact") is False
    and policy.get("fallback_success") is True
)
needs = (
    isinstance(approach, dict)
    and approach.get("passed") is False
    and approach.get("planner_id") == "robotwin-mplib-rrt"
    and isinstance(policy, dict)
    and policy.get("strategy") == "mplib_target_aware_acquisition"
    and policy.get("primary_success") is False
    and policy.get("primary_failure_code") == "optimization_failed"
    and policy.get("primary_vendor_status") == "screw plan failed"
    and policy.get("fallback_allowed") is True
    and policy.get("fallback_attempted") is True
    and policy.get("fallback_planner_id") == "robotwin-mplib-rrt"
    and (
        fallback_planning_failed
        or target_collision_rejected
        or execution_target_contact_rejected
    )
)
raise SystemExit(0 if needs else 1)
PY
  SEED1_NEEDS_RECOVERY=$?
fi

if [[ "$SEED1_NEEDS_RECOVERY" -eq 0 ]]; then
  expanded_label="manipulation-seed-1-expanded"
  run_native "$expanded_label" 3600 \
    "$CONDA_EXE" run -n RoboTwin python -u \
    scripts/linux/probe_panthera_project_manipulation.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --project-root "$PROJECT_ROOT" \
    --output "$RESULT_DIR/${expanded_label}.json" \
    --seed 1 \
    --b5-evaluate-all-height-groups
  expanded_rc=$?
  printf 'seed1-expanded=%s\n' "$expanded_rc" >> "$RESULT_DIR/stage-return-codes.txt"

  expanded_audit_rc=99
  if [[ "$expanded_rc" -eq 0 && -f "$RESULT_DIR/${expanded_label}.json" ]]; then
    PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
      python scripts/audit_project_manipulation.py "$RESULT_DIR/${expanded_label}.json" \
      > >(tee "$RESULT_DIR/${expanded_label}-audit.log") 2>&1
    expanded_audit_rc=$?
    if [[ "$expanded_audit_rc" -eq 0 ]]; then
      # The expanded candidate must be explicitly eligible, distinct from the
      # fixed seed1 candidate, and proven by the independent native auditor.
      expanded_candidate="$(python3 - "$RESULT_DIR/${expanded_label}.json" "$SEED1_JSON" <<'PY'
import json,sys
expanded=json.load(open(sys.argv[1],encoding="utf-8"))
original=json.load(open(sys.argv[2],encoding="utf-8"))
b5=expanded.get("B5") or {}
selected=(b5.get("selected") or {}).get("candidate_id")
original_id=((original.get("B5") or {}).get("selected") or {}).get("candidate_id")
ranking=b5.get("ranking") or []
if (
    expanded.get("passed") is True
    and isinstance(selected,str) and bool(selected)
    and selected != original_id
    and any(
        isinstance(row,dict)
        and row.get("candidate_id")==selected
        and row.get("eligible") is True
        for row in ranking
    )
):
    print(selected)
PY
)"
      if [[ -n "$expanded_candidate" ]]; then
        RECOVERY_PASS_COUNT=$((RECOVERY_PASS_COUNT + 1))
        SEED1_RECOVERY_ACCEPTED=1
        SEED1_EFFECTIVE_LABEL="$expanded_label"
        SEED1_EFFECTIVE_CANDIDATE="$expanded_candidate"
        MANIP_PASS_COUNT=$((MANIP_PASS_COUNT + 1))
        MANIP_AUDIT_PASS_COUNT=$((MANIP_AUDIT_PASS_COUNT + 1))
        echo "Seed1 expanded deterministic candidate independently audited PASS."
      else
        expanded_audit_rc=96
        echo "Seed1 expanded candidate provenance check rejected."
      fi
    fi
  fi
  printf 'seed1-expanded-audit=%s\n' "$expanded_audit_rc" \
    >> "$RESULT_DIR/stage-return-codes.txt"

  if [[ -f "$RESULT_DIR/${expanded_label}.json" && "$SEED1_RECOVERY_ACCEPTED" -eq 0 ]]; then
    python3 - "$RESULT_DIR/${expanded_label}.json" \
      > "$RESULT_DIR/seed1-recovery-candidates.txt" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
b5 = data.get("B5") or {}
selected = (b5.get("selected") or {}).get("candidate_id")
eligible = [
    row.get("candidate_id")
    for row in (b5.get("ranking") or [])
    if isinstance(row, dict)
    and row.get("eligible") is True
    and row.get("candidate_id")
    and row.get("candidate_id") != selected
]
for candidate_id in eligible[:2]:
    print(candidate_id)
PY

    recovery_index=0
    while IFS= read -r candidate_id; do
      [[ -n "$candidate_id" ]] || continue
      recovery_index=$((recovery_index + 1))
      RECOVERY_DIAG_COUNT=$((RECOVERY_DIAG_COUNT + 1))
      label="manipulation-seed-1-recovery-${recovery_index}"
      run_native "$label" 3600 \
        "$CONDA_EXE" run -n RoboTwin python -u \
        scripts/linux/probe_panthera_project_manipulation.py \
        --robotwin-root "$ROBOTWIN_ROOT" \
        --project-root "$PROJECT_ROOT" \
        --output "$RESULT_DIR/${label}.json" \
        --seed 1 \
        --b5-evaluate-all-height-groups \
        --candidate-id "$candidate_id"
      rc=$?
      audit_rc=99
      if [[ "$rc" -eq 0 ]]; then
        PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
          python scripts/audit_project_manipulation.py "$RESULT_DIR/${label}.json" \
          > >(tee "$RESULT_DIR/${label}-audit.log") 2>&1
        audit_rc=$?
        if [[ "$audit_rc" -eq 0 ]]; then
          RECOVERY_PASS_COUNT=$((RECOVERY_PASS_COUNT + 1))
          if [[ "$SEED1_RECOVERY_ACCEPTED" -eq 0 ]]; then
            SEED1_RECOVERY_ACCEPTED=1
            SEED1_EFFECTIVE_LABEL="$label"
            SEED1_EFFECTIVE_CANDIDATE="$candidate_id"
            MANIP_PASS_COUNT=$((MANIP_PASS_COUNT + 1))
            MANIP_AUDIT_PASS_COUNT=$((MANIP_AUDIT_PASS_COUNT + 1))
          fi
        fi
      fi
      printf '%s-audit=%s\n' "$label" "$audit_rc" \
        >> "$RESULT_DIR/stage-return-codes.txt"
      if [[ "$SEED1_EFFECTIVE_LABEL" == "$label" ]]; then
        break
      fi
    done < "$RESULT_DIR/seed1-recovery-candidates.txt"
  fi
else
  echo "Seed 1 recovery skipped: no supported acquisition-recovery failure signature."
fi

SEED2_RECOVERY_ACCEPTED=0
SEED2_EFFECTIVE_LABEL="manipulation-seed-2"
SEED2_EFFECTIVE_CANDIDATE=""
SEED2_JSON="$RESULT_DIR/manipulation-seed-2.json"

echo
echo "### Stage B3 - seed2 one ranked alternative grasp after placement failure"
if [[ -f "$SEED2_JSON" ]]; then
  SEED2_ALTERNATIVE="$(python3 - "$SEED2_JSON" <<'PY'
import json,sys
with open(sys.argv[1],encoding="utf-8") as handle:
    data=json.load(handle)
place=next((s for s in data.get("stages",[])
    if isinstance(s,dict) and s.get("stage")=="place_above_support"),None)
if not isinstance(place,dict) or place.get("passed") is not False:
    raise SystemExit(0)
failure=place.get("failure") or {}
details=failure.get("details") or {}
status=str(details.get("native_status") or "")
if failure.get("code")!="optimization_failed" or not (
    status=="IK Fail" or status=="Start state is out of joint limits"
):
    raise SystemExit(0)
b5=data.get("B5") or {}
selected=(b5.get("selected") or {}).get("candidate_id")
if b5.get("passed") is not True or not isinstance(selected,str):
    raise SystemExit(0)
eligible=[
    row["candidate_id"]
    for row in (b5.get("ranking") or [])
    if isinstance(row,dict)
    and row.get("eligible") is True
    and isinstance(row.get("candidate_id"),str)
    and row["candidate_id"]!=selected
]
if eligible:
    print(eligible[0])
PY
)"
  if [[ -n "$SEED2_ALTERNATIVE" ]]; then
    RECOVERY_DIAG_COUNT=$((RECOVERY_DIAG_COUNT + 1))
    label="manipulation-seed-2-ranked-alternative"
    run_native "$label" 3600       "$CONDA_EXE" run -n RoboTwin python -u       scripts/linux/probe_panthera_project_manipulation.py       --robotwin-root "$ROBOTWIN_ROOT"       --project-root "$PROJECT_ROOT"       --output "$RESULT_DIR/${label}.json"       --seed 2 --b5-evaluate-all-height-groups       --candidate-id "$SEED2_ALTERNATIVE"
    alternative_rc=$?
    alternative_audit_rc=99
    if [[ "$alternative_rc" -eq 0 && -f "$RESULT_DIR/${label}.json" ]]; then
      PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin         python scripts/audit_project_manipulation.py         "$RESULT_DIR/${label}.json"         > >(tee "$RESULT_DIR/${label}-audit.log") 2>&1
      alternative_audit_rc=$?
      if [[ "$alternative_audit_rc" -eq 0 ]]; then
        # Fail closed on provenance/eligibility, even if native/audit claim PASS.
        if python3 - "$RESULT_DIR/${label}.json" "$SEED2_JSON" "$SEED2_ALTERNATIVE" <<'PY'
import json,sys
new=json.load(open(sys.argv[1],encoding="utf-8"))
old=json.load(open(sys.argv[2],encoding="utf-8"))
candidate=sys.argv[3]
b5=new.get("B5") or {}
selected=(b5.get("selected") or {}).get("candidate_id")
original=((old.get("B5") or {}).get("selected") or {}).get("candidate_id")
ok=(
    new.get("seed")==2 and new.get("passed") is True
    and selected==candidate and candidate!=original
    and any(
        isinstance(item,dict) and item.get("candidate_id")==candidate
        and item.get("eligible") is True
        for item in (b5.get("ranking") or [])
    )
)
raise SystemExit(0 if ok else 1)
PY
        then
          SEED2_RECOVERY_ACCEPTED=1
          SEED2_EFFECTIVE_LABEL="$label"
          SEED2_EFFECTIVE_CANDIDATE="$SEED2_ALTERNATIVE"
          RECOVERY_PASS_COUNT=$((RECOVERY_PASS_COUNT + 1))
          MANIP_PASS_COUNT=$((MANIP_PASS_COUNT + 1))
          MANIP_AUDIT_PASS_COUNT=$((MANIP_AUDIT_PASS_COUNT + 1))
        else
          alternative_audit_rc=96
        fi
      fi
    fi
    printf 'seed2-ranked-alternative=%s audit=%s\n'       "$alternative_rc" "$alternative_audit_rc"       >> "$RESULT_DIR/stage-return-codes.txt"
  fi
fi


python3 - \
  "$RESULT_DIR/manipulation-seed-resolution.json" \
  "$SEED1_EFFECTIVE_LABEL" \
  "$SEED1_RECOVERY_ACCEPTED" \
  "$SEED1_EFFECTIVE_CANDIDATE" \
  "$SEED2_EFFECTIVE_LABEL" \
  "$SEED2_RECOVERY_ACCEPTED" \
  "$SEED2_EFFECTIVE_CANDIDATE" <<'PY'
import json, pathlib, sys

output = pathlib.Path(sys.argv[1])
seed1_label = sys.argv[2]
seed1_recovered = sys.argv[3] == "1"
seed1_candidate = sys.argv[4] or None
seed2_label = sys.argv[5]
seed2_recovered = sys.argv[6] == "1"
seed2_candidate = sys.argv[7] or None

rows = []
for seed in (0, 1, 2):
    label = (
        seed1_label if seed == 1
        else seed2_label if seed == 2
        else f"manipulation-seed-{seed}"
    )
    rows.append(
        {
            "seed": seed,
            "source_label": label,
            "probe_relative": f"{label}.json",
            "audit_log_relative": f"{label}-audit.log",
            "recovered": (
                (seed == 1 and seed1_recovered)
                or (seed == 2 and seed2_recovered)
            ),
            "candidate_id": (
                seed1_candidate if seed == 1 and seed1_recovered
                else seed2_candidate if seed == 2 and seed2_recovered
                else None
            ),
        }
    )

output.write_text(
    json.dumps(
        {
            "schema": "manipulation-seed-resolution-v1",
            "policy": "deterministic-ranked-recovery-v1",
            "seeds": rows,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo
echo "### Stage C - provisional native closure candidate"
if [[ "$MANIP_PASS_COUNT" -eq 3 && "$MANIP_AUDIT_PASS_COUNT" -eq 3 ]]; then
  set +e
  PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/build_manipulation_closure_candidate.py \
    --result-dir "$RESULT_DIR" \
    --linux-git-head "$ACTUAL_HEAD" \
    --seed-resolution "$RESULT_DIR/manipulation-seed-resolution.json" \
    --output "$RESULT_DIR/manipulation-closure-candidate.json" \
    > >(tee "$RESULT_DIR/manipulation-closure-candidate.log") 2>&1
  CLOSURE_CANDIDATE_RC=$?
else
  echo "Closure candidate skipped: manipulation fixed-seed acceptance is not 3/3 audited PASS."
fi
printf 'closure-candidate=%s\n' "$CLOSURE_CANDIDATE_RC" >> "$RESULT_DIR/stage-return-codes.txt"

echo
echo "### Stage D - current readiness snapshot"
READINESS_ARGS=(--output "$RESULT_DIR/readiness.json")
if [[ "$CLOSURE_CANDIDATE_RC" -eq 0 ]]; then
  READINESS_ARGS+=(--manipulation-candidate "$RESULT_DIR/manipulation-closure-candidate.json")
fi
set +e
PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
  python scripts/audit_sprint1_readiness.py "${READINESS_ARGS[@]}" \
  > >(tee "$RESULT_DIR/readiness.log") 2>&1
READINESS_RC=$?
printf 'readiness=%s\n' "$READINESS_RC" >> "$RESULT_DIR/stage-return-codes.txt"

echo
echo "### Stage E - Sprint-1 native closure audit"
if [[ "$CLOSURE_CANDIDATE_RC" -eq 0 && "$READINESS_RC" -eq 0 ]]; then
  set +e
  python3 scripts/audit_sprint1_native_closure.py \
    "$RESULT_DIR/readiness.json" \
    --output "$RESULT_DIR/sprint1-native-closure.json" \
    > >(tee "$RESULT_DIR/sprint1-native-closure.log") 2>&1
  NATIVE_CLOSURE_RC=$?
else
  echo "Native closure audit skipped: no valid manipulation closure candidate."
fi
printf 'native-closure=%s\n' "$NATIVE_CLOSURE_RC" >> "$RESULT_DIR/stage-return-codes.txt"

kill "$GPU_PID" >/dev/null 2>&1 || true
wait "$GPU_PID" 2>/dev/null || true

python3 - "$RESULT_DIR" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
gpu = root / "gpu.csv"
if gpu.is_file():
    for row in csv.reader(gpu.read_text(encoding="utf-8", errors="replace").splitlines()):
        if len(row) < 6:
            continue
        try:
            rows.append(
                {
                    "gpu_util_percent": float(row[2].strip()),
                    "memory_util_percent": float(row[3].strip()),
                    "memory_used_mb": float(row[4].strip()),
                    "memory_total_mb": float(row[5].strip()),
                }
            )
        except ValueError:
            pass
summary = {
    "samples": len(rows),
    "peak_gpu_util_percent": max((r["gpu_util_percent"] for r in rows), default=None),
    "peak_memory_util_percent": max((r["memory_util_percent"] for r in rows), default=None),
    "peak_memory_used_mb": max((r["memory_used_mb"] for r in rows), default=None),
    "memory_total_mb": max((r["memory_total_mb"] for r in rows), default=None),
}
(root / "resource-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
)
PY

FIRST_BLOCKER=""
if [[ "$PREREQ_RC" -ne 0 ]]; then
  FIRST_BLOCKER="accepted prerequisite manifest validation"
elif [[ "$SYNC_RC" -ne 0 ]]; then
  FIRST_BLOCKER="target-aware MPlib smoke"
elif [[ "$MANIP_PASS_COUNT" -lt 3 ]]; then
  FIRST_BLOCKER="project-owned manipulation native probe"
elif [[ "$MANIP_AUDIT_PASS_COUNT" -lt 3 ]]; then
  FIRST_BLOCKER="project-owned manipulation independent audit"
fi

CAMPAIGN_ACCEPTED=0
if [[ -z "$FIRST_BLOCKER" ]]; then
  CAMPAIGN_ACCEPTED=1
fi

cat > "$RESULT_DIR/CAMPAIGN_RESULT.txt" <<EOF
PREREQ_RC=$PREREQ_RC
SYNC_RC=$SYNC_RC
MANIP_PASS_COUNT=$MANIP_PASS_COUNT
MANIP_AUDIT_PASS_COUNT=$MANIP_AUDIT_PASS_COUNT
RECOVERY_DIAG_COUNT=$RECOVERY_DIAG_COUNT
RECOVERY_PASS_COUNT=$RECOVERY_PASS_COUNT
SEED1_RECOVERY_ACCEPTED=$SEED1_RECOVERY_ACCEPTED
SEED1_EFFECTIVE_LABEL=$SEED1_EFFECTIVE_LABEL
SEED1_EFFECTIVE_CANDIDATE=$SEED1_EFFECTIVE_CANDIDATE
SEED2_RECOVERY_ACCEPTED=$SEED2_RECOVERY_ACCEPTED
SEED2_EFFECTIVE_LABEL=$SEED2_EFFECTIVE_LABEL
SEED2_EFFECTIVE_CANDIDATE=$SEED2_EFFECTIVE_CANDIDATE
READINESS_RC=$READINESS_RC
CLOSURE_CANDIDATE_RC=$CLOSURE_CANDIDATE_RC
NATIVE_CLOSURE_RC=$NATIVE_CLOSURE_RC
CAMPAIGN_ACCEPTED=$CAMPAIGN_ACCEPTED
FIRST_BLOCKER=$FIRST_BLOCKER
EOF

FOLLOWUP_RC=0
python3 scripts/classify_native_manipulation_followup.py "$RESULT_DIR" --json   > "$RESULT_DIR/FOLLOWUP_DECISION.json" || FOLLOWUP_RC=$?
if [[ "$FOLLOWUP_RC" -eq 0 ]]; then
  python3 scripts/classify_native_manipulation_followup.py "$RESULT_DIR"     > "$RESULT_DIR/FOLLOWUP_DECISION.txt"
  FOLLOWUP_PRIMARY_DECISION="$(
    python3 - "$RESULT_DIR/FOLLOWUP_DECISION.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("primary_decision", "UNKNOWN"))
PY
  )"
else
  FOLLOWUP_PRIMARY_DECISION="CLASSIFIER_ERROR"
  printf 'classifier_rc=%s\n' "$FOLLOWUP_RC" > "$RESULT_DIR/FOLLOWUP_DECISION.txt"
fi
printf 'followup_classifier=%s\n' "$FOLLOWUP_RC" >> "$RESULT_DIR/stage-return-codes.txt"

# Additive evidence only. Digest failures can never upgrade native acceptance.
DIAGNOSTIC_DIGEST_RC=0
python3 scripts/analyze_native_manipulation_campaign.py "$RESULT_DIR" \
  --json-output "$RESULT_DIR/DIAGNOSTIC_DIGEST.json" \
  --text-output "$RESULT_DIR/DIAGNOSTIC_DIGEST.txt" \
  > "$RESULT_DIR/diagnostic-digest.log" 2>&1 || DIAGNOSTIC_DIGEST_RC=$?
printf 'diagnostic-digest=%s\n' "$DIAGNOSTIC_DIGEST_RC" >> "$RESULT_DIR/stage-return-codes.txt"

git status --short --branch > "$RESULT_DIR/git-status-after.txt"
git diff > "$RESULT_DIR/native-working-tree.patch"
git diff --cached > "$RESULT_DIR/native-staged.patch"

cp -a "$RESULT_DIR/." "$EVIDENCE/"
if [[ "$CAMPAIGN_ACCEPTED" -eq 1 ]]; then
  echo "PASS" > "$EVIDENCE/PASS"
  RESULT_CLASSIFICATION="PASS"
  CAMPAIGN_RC=0
else
  printf 'FIRST_BLOCKER=%s\n' "$FIRST_BLOCKER" > "$EVIDENCE/PARTIAL"
  RESULT_CLASSIFICATION="PARTIAL"
  CAMPAIGN_RC=40
fi

(
  cd "$EVIDENCE"
  find . -type f ! -path './SHA256SUMS.txt' -print0     | sort -z     | xargs -0 sha256sum > SHA256SUMS.txt
)

echo
echo "=== Exporting manipulation campaign evidence ==="
DAMH_RESULT_CLASSIFICATION_OVERRIDE="$RESULT_CLASSIFICATION" DAMH_RESULT_SCOPE_OVERRIDE="Sprint1 project-owned manipulation campaign" DAMH_FIRST_FAIL_OVERRIDE="$FIRST_BLOCKER" DAMH_EVIDENCE_DIR_OVERRIDE="$EVIDENCE"   bash scripts/linux/collect_to_bridge.sh
COLLECT_RC=$?
sync

echo
cat "$RESULT_DIR/CAMPAIGN_RESULT.txt"
echo "FOLLOWUP_PRIMARY_DECISION=$FOLLOWUP_PRIMARY_DECISION"
echo "Evidence export exit: $COLLECT_RC"
ls -1dt "$BRIDGE/incoming/"* 2>/dev/null | head -n 1 || true

if [[ "$COLLECT_RC" -ne 0 ]]; then
  echo "Evidence export failed. Preserve/export evidence before reboot."
  exit "$COLLECT_RC"
fi

if [[ "$CAMPAIGN_ACCEPTED" -eq 1 ]]; then
  echo "Manipulation campaign acceptance PASS."
else
  echo "Manipulation campaign completed PARTIAL; all seed diagnostics were preserved."
fi
exit "$CAMPAIGN_RC"
