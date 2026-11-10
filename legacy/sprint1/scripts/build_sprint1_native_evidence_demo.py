# ruff: noqa: E501, RUF001  # Embeds standalone HTML/CSS/JS with literal wide strings.
"""Build a self-contained, read-only Sprint 1 evidence replay from independently sealed native artifacts.

This is NOT a live RoboTwin session, synchronized video, visual object detector, Gemini call,
or an end-to-end image-to-action system. It deliberately keeps the two capture sessions apart.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
from pathlib import Path

STAGES = (
    "pregrasp", "approach", "close_gripper", "lift",
    "transport", "place_above_support", "release", "retreat",
)
EXPECTED_LINUX_HEAD = "0cfa6b32f72668a1613c45c0f8260e52d79bbd4a"
SHA_ROW = re.compile(r"^([0-9a-f]{64})\s+[*]?(.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    """Validate every listed file, reject path traversal, duplicates and malformed rows."""
    root = root.resolve()
    manifest = root / "SHA256SUMS.txt"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines) > 512:
        raise ValueError("missing or oversized evidence SHA256SUMS")
    seen = set()
    for line in lines:
        match = SHA_ROW.fullmatch(line)
        if match is None:
            raise ValueError("invalid SHA256SUMS line")
        expected, relative = match.groups()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path == root or path in seen:
            raise ValueError("unsafe or repeated evidence path")
        seen.add(path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("missing or modified evidence: " + relative)
    return len(seen)


def _number(value):
    return round(float(value), 6) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _joint_values(value):
    if not isinstance(value, list):
        return []
    return [_number(x) for x in value[:8]]


def build_payload(native: Path, vision: Path, accepted: Path) -> dict:
    native, vision, accepted = native.resolve(), vision.resolve(), accepted.resolve()
    if not accepted.is_file():
        raise ValueError("no durable accepted manipulation manifest")
    manifest = json.loads(accepted.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "accepted-native-gate-v1"
            or manifest.get("classification") != "PASS_AUDITED"
            or manifest.get("gate_id") != "S1.6-MANIPULATION"
            or manifest.get("linux_git_head") != EXPECTED_LINUX_HEAD
            or manifest.get("fixed_seeds") != [0, 1, 2]
            or manifest.get("probe_pass_count") != 3
            or manifest.get("independent_audit_pass_count") != 3):
        raise ValueError("durable manipulation acceptance contract mismatch")
    if manifest.get("source_bridge_relative") != "incoming/sprint1_round2_20261015T030716Z/native-strict":
        raise ValueError("accepted manifest points to other source")
    if native.name != "native-strict" or native.parent.name != "sprint1_round2_20261015T030716Z":
        raise ValueError("unexpected native campaign")
    if vision.name != "s1v5_depth_repair_20261013T154351Z":
        raise ValueError("unexpected independently captured RGBD source")
    native_count = verify_manifest(native)
    native_export_count = verify_manifest(native.parent)
    vision_count = verify_manifest(vision)
    strict_audit = json.loads((native.parent / "native-strict-campaign-audit.json").read_text(encoding="utf-8"))
    if (strict_audit.get("strict_native_campaign_audit_passed") is not True
            or strict_audit.get("accepted_seeds") != [0, 1, 2]
            or strict_audit.get("accepted_seed_count") != 3):
        raise ValueError("strict campaign independent audit missing or incomplete")
    rows = manifest.get("seeds")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("accepted seed evidence malformed")
    result = []
    for seed in (0, 1, 2):
        row = rows[seed]
        if row.get("seed") != seed:
            raise ValueError("wrong accepted seed order")
        probe_name, log_name = row.get("probe_relative"), row.get("audit_log_relative")
        if probe_name != f"manipulation-seed-{seed}.json" or log_name != f"audit-seed-{seed}.stdout.log":
            raise ValueError("noncanonical physical evidence reference")
        probe, log = native / probe_name, native / log_name
        if sha256(probe) != row.get("probe_sha256") or sha256(log) != row.get("audit_log_sha256"):
            raise ValueError("durable acceptance differs from source bytes")
        if "AUDIT=PASS" not in log.read_text(encoding="utf-8"):
            raise ValueError("independent seed audit missing PASS")
        raw = json.loads(probe.read_text(encoding="utf-8"))
        if (raw.get("seed") != seed or raw.get("passed") is not True
                or raw.get("final_phase") != "complete"
                or raw.get("task_success_final") is not True
                or raw.get("task_success_after_release") is not True):
            raise ValueError("physical task did not pass")
        stages = raw.get("stages")
        if not isinstance(stages, list) or [x.get("stage") for x in stages] != list(STAGES):
            raise ValueError("not exactly eight physical stages")
        exported = []
        for entry in stages:
            if entry.get("passed") is not True or entry.get("execution_status") != "completed":
                raise ValueError("physical stage not completed")
            exported.append({
                "name": entry["stage"],
                "planner": entry.get("planner_id") or "direct gripper action",
                "status": "completed",
                "planning_s": _number(entry.get("planning_time_s")),
                "duration_s": _number(entry.get("trajectory_duration_s")),
                "latency_s": _number(entry.get("execution_latency_s")),
                "path_samples": entry.get("path_samples") if isinstance(entry.get("path_samples"), int) else None,
                "q_after": _joint_values(entry.get("qpos_after_rad")),
                "tracking_error_rad": _number(entry.get("max_tracking_error_rad")),
                "collision_safe": entry.get("collision_safe") if type(entry.get("collision_safe")) is bool else None,
                "yaw_deg": _number(math.degrees(entry.get("placement_yaw_offset_rad", 0))) if entry["stage"] == "place_above_support" else None,
                "strategy": (entry.get("placement_recovery") or {}).get("strategy") if entry["stage"] == "place_above_support" else None,
            })
        result.append({
            "seed": seed,
            "stages": exported,
            "probe_sha256": row["probe_sha256"],
            "audit_sha256": row["audit_log_sha256"],
            "native_head": EXPECTED_LINUX_HEAD,
            "task_success_final": True,
        })
    images = []
    for camera_id in ("head_camera", "wrist_camera"):
        stem = camera_id + "_frame_000000"
        directory = vision / "native-depth"
        manifest_path = directory / (stem + "_depth-manifest.json")
        camera = json.loads(manifest_path.read_text(encoding="utf-8"))
        file = directory / (stem + "_rgb.png")
        if (camera.get("camera_id") != camera_id
                or camera.get("frame_index") != 0
                or camera.get("source_classification") != "recorded_sim_observation"
                or camera.get("scene_revision") is not None
                or camera.get("controller_eligible") is not False
                or camera.get("frame_association_source") != "probe.get_obs single-frame camera refresh"
                or camera.get("rgb_sha256") != sha256(file)):
            raise ValueError("RGBD provenance incompatible: " + camera_id)
        images.append({
            "id": camera_id,
            "size": camera["image_size_px"],
            "rgb_sha256": sha256(file),
            "manifest_sha256": sha256(manifest_path),
            "uri": "data:image/png;base64," + base64.b64encode(file.read_bytes()).decode("ascii"),
        })
    return {
        "schema": "sprint1-offline-native-evidence-replay-v1",
        "claim": "READ_ONLY_HISTORICAL_EVIDENCE_NOT_LIVE_VLA_OR_CAMERA_SYNCHRONIZATION",
        "accepted_manifest_sha256": sha256(accepted),
        "strict_audit_sha256": sha256(native.parent / "native-strict-campaign-audit.json"),
        "native_file_checksums_verified": native_count,
        "native_export_file_checksums_verified": native_export_count,
        "vision_file_checksums_verified": vision_count,
        "native": result,
        "vision": images,
    }


HTML = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAMH / Sprint 1 — Native evidence replay</title>
<style>
:root{font-family:Inter,Segoe UI,system-ui,sans-serif;color:#e5edf8;background:#0b1220}
*{box-sizing:border-box}body{margin:0}button{font:inherit;cursor:pointer}button:focus-visible{outline:2px solid #90c6ff;outline-offset:2px}
.wrap{max-width:1230px;margin:auto;padding:28px 22px 50px}
.kicker{color:#89baff;font-size:12px;letter-spacing:.14em;font-weight:750;text-transform:uppercase}
h1{font-size:clamp(24px,3vw,38px);margin:10px 0}p{line-height:1.55;color:#aebdd1}
.pill{display:inline-block;border:1px solid #45546c;border-radius:99px;padding:6px 10px;font-size:12px;color:#d2dfef}
.notice{margin:20px 0;padding:14px 18px;border:1px solid #4c5e79;border-radius:12px;background:#162034;font-size:13px;line-height:1.6}
.grid{display:grid;grid-template-columns:minmax(0,1.28fr) minmax(0,.82fr);gap:18px}
.card{border:1px solid #29394e;border-radius:15px;background:#101b2c;padding:20px;min-width:0}
h2{font-size:17px;margin:0 0 16px}h3{font-size:14px;margin:18px 0 9px;color:#c1d8f4}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.btn{padding:10px 14px;border:1px solid #3d5773;color:#d9e6f7;background:#19283e;border-radius:9px}
.btn:hover,.btn.on{background:#315d88;border-color:#88baff;color:#fff}
.btn:disabled{opacity:.45;cursor:not-allowed}
.monitor{background:#070d17;border:1px solid #243146;border-radius:12px;overflow:hidden;text-align:center;min-height:180px;display:grid;place-items:center}
.monitor img{display:block;width:100%;height:auto;max-height:320px;object-fit:contain}
.small{font-size:12px;color:#9eafc6;line-height:1.6}
.timeline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.stage{min-height:68px;padding:10px 8px;text-align:left;overflow-wrap:anywhere;font-size:12px;background:#15243a;border:1px solid #334862;color:#d6e2f3;border-radius:9px}
.stage.on{outline:2px solid #9ccaff;background:#23527a}
.stage .num{display:block;color:#91baff;font-size:10px;margin-bottom:5px}
dl{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:0}
dl>div{border-bottom:1px solid #27394f;padding:8px 0}
dt{font-size:11px;color:#90a5bf;margin-bottom:5px}dd{font-size:13px;margin:0;overflow-wrap:anywhere}
.mono{font-family:ui-monospace,Consolas,monospace}
.joints{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.joints span{border:1px solid #29394e;border-radius:7px;padding:9px 6px;text-align:center;font-size:12px}
.footer{margin-top:20px;font-size:12px;line-height:1.8;color:#94a5b8;overflow-wrap:anywhere}
@media(max-width:850px){.grid{grid-template-columns:1fr}}@media(max-width:460px){.timeline{grid-template-columns:repeat(2,1fr)}.wrap{padding:16px 12px}.card{padding:14px}.joints{grid-template-columns:repeat(2,1fr)}}
</style></head>
<body><main class="wrap">
<div class="kicker">DAMH · Panthera-HT · Sprint 1</div>
<h1>Native manipulation · Evidence replay</h1>
<p>Chọn seed, từng giai đoạn và camera để kiểm tra dữ liệu từ các lần chạy RoboTwin/SAPIEN đã nghiệm thu.</p>
<span class="pill">3/3 fixed seeds · Strict native PASS</span> <span class="pill">8/8 stages per seed</span> <span class="pill">Windows review · PASS_DURABLE</span>
<div class="notice"><b>Đây là bản phát lại bằng chứng, KHÔNG PHẢI mô phỏng live.</b>
Ảnh RGB-D bên dưới được ghi trong phiên camera khác ngày 13/10; các giai đoạn chuyển động thuộc native campaign 15/10.
Không đồng bộ hình ảnh với từng giai đoạn, không Gemini live, không nhận diện object ID từ ảnh, không gửi lệnh robot.</div>
<div class="grid">
<section class="card">
<h2>1 · Quan sát RGB-D độc lập</h2>
<div class="row" id="cameras"></div>
<div class="monitor" style="margin-top:12px"><img id="camimage" alt="RGB camera snapshot from independent RoboTwin capture"></div>
<p id="camdesc" class="small"></p>
<div style="margin-top:18px">
<h2>2 · Native task đã ghi</h2>
<div class="row" id="seeds"></div>
<p class="small">Ngữ cảnh thao tác: <b>move_pillbottle_pad</b> — gắp, nâng, di chuyển và đặt chai lên pad. Đây là nhiệm vụ kịch bản, không được tạo ra từ Gemini trong lượt chạy này.</p>
<div class="row" style="margin-bottom:12px"><button class="btn" id="play" type="button">▶ Phát lại các giai đoạn</button><button class="btn" id="pause" type="button" disabled>❚❚ Dừng</button><button class="btn" id="reset" type="button">↺ Về đầu</button></div>
<div class="timeline" id="timeline"></div>
</div>
</section>
<section class="card">
<h2>3 · Bằng chứng vật lý của giai đoạn</h2>
<div id="stagehead" class="pill"></div>
<h3>Thực thi và bộ lập kế hoạch</h3>
<dl id="details"></dl>
<h3>Trạng thái khớp cuối giai đoạn (rad)</h3>
<div class="joints" id="joints"></div>
<p class="small">Các giá trị khớp là bản ghi đo sau giai đoạn (nếu có). Đây không phải hoạt họa 3D, không phải mô phỏng đường đi liên tục.</p>
<h3>Biên kiểm chứng</h3>
<p class="small" id="stageboundary"></p>
</section>
</div>
<div class="footer">
<div><b>Native Linux source</b> <span id="head" class="mono"></span></div>
<div><b>Durable acceptance SHA256</b> <span id="accepted" class="mono"></span></div>
<div><b>Independent strict audit SHA256</b> <span id="audit" class="mono"></span></div>
<div><b>Data classification:</b> source-checked historical replay; not online semantic perception, real-time control or video synchronization.</div>
</div>
</main>
<script>
const D=@@DATA@@;
const root=document.querySelector("main");
let seed=0,stage=0,camera=0,timer=null;
const names={pregrasp:"Tiếp cận trước gắp",approach:"Tiếp cận",close_gripper:"Đóng kẹp",lift:"Nâng vật",transport:"Di chuyển",place_above_support:"Đưa trên pad",release:"Thả vật",retreat:"Rút tay"};
const $=id=>document.getElementById(id);
function stop(){if(timer!==null){clearInterval(timer);timer=null;} $("play").disabled=false;$("pause").disabled=true;}
function makeButton(label,active,handler){const b=document.createElement("button");b.className="btn"+(active?" on":"");b.type="button";b.textContent=label;b.addEventListener("click",handler);return b;}
function render(){
$("seeds").replaceChildren(...D.native.map((x,i)=>makeButton("Seed "+x.seed+(x.task_success_final?" ✓":""),i===seed,()=>{stop();seed=i;stage=0;render();})));
$("cameras").replaceChildren(...D.vision.map((x,i)=>makeButton(x.id==="head_camera"?"Head camera":"Wrist camera",i===camera,()=>{camera=i;render();})));
const im=D.vision[camera];$("camimage").src=im.uri;
$("camdesc").textContent=im.id+" · "+im.size.join(" × ")+" px · recorded native RGB-D from independent session · not frame-aligned with manipulation · RGB SHA "+im.rgb_sha256.slice(0,16)+"…";
const run=D.native[seed];
$("timeline").replaceChildren(...run.stages.map((x,i)=>{
const b=document.createElement("button");b.type="button";b.className="stage"+(i===stage?" on":"");
const k=document.createElement("span");k.className="num";k.textContent=String(i+1).padStart(2,"0")+" · PASS";
b.append(k,document.createTextNode(names[x.name]||x.name));
b.addEventListener("click",()=>{stop();stage=i;render();});return b;
}));
const x=run.stages[stage];$("stagehead").textContent="Seed "+seed+" · "+(stage+1)+"/8 · "+x.name+" · "+x.status;
const fields=[
["Planner",x.planner],["Physical execution",x.status],["Planning time",x.planning_s===null?"not recorded":x.planning_s+" s"],
["Trajectory duration",x.duration_s===null?"not recorded":x.duration_s+" s"],
["Execution latency",x.latency_s===null?"not recorded":x.latency_s+" s"],
["Path samples",x.path_samples===null?"not recorded":String(x.path_samples)],
["Collision validation",x.collision_safe===null?"n/a":(x.collision_safe?"PASS":"NOT PASS")],
["Tracking max error",x.tracking_error_rad===null?"not recorded":x.tracking_error_rad+" rad"]];
if(x.yaw_deg!==null){fields.push(["Placement yaw",x.yaw_deg+"°"],["Placement policy",x.strategy||"not recorded"]);}
$("details").replaceChildren(...fields.map(([k,v])=>{const box=document.createElement("div"),dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=k;dd.textContent=v;box.append(dt,dd);return box;}));
$("joints").replaceChildren(...x.q_after.map((v,i)=>{const a=document.createElement("span");a.textContent="q"+(i+1)+": "+(v===null?"n/a":v.toFixed(3));return a;}));
$("stageboundary").textContent="Accepted native seed "+seed+" · probe SHA "+run.probe_sha256.slice(0,20)+"… · independent seed audit SHA "+run.audit_sha256.slice(0,20)+"… · verified against durable manifest. Camera imagery from another capture is for context only.";
}
$("play").addEventListener("click",()=>{stop();$("play").disabled=true;$("pause").disabled=false;if(stage===7){stage=0;render();}timer=setInterval(()=>{if(stage>=7){stop();return;}stage++;render();},1250);});
$("pause").addEventListener("click",stop);
$("reset").addEventListener("click",()=>{stop();stage=0;render();});
$("head").textContent=D.native[0].native_head;$("accepted").textContent=D.accepted_manifest_sha256;$("audit").textContent=D.strict_audit_sha256;render();
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-strict", type=Path, required=True)
    parser.add_argument("--vision-export", type=Path, required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("refuse to overwrite existing demo")
    payload = build_payload(args.native_strict, args.vision_export, args.accepted_manifest)
    source_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("@@DATA@@", source_json), encoding="utf-8")
    print("SPRINT1_NATIVE_EVIDENCE_DEMO=PASS_READ_ONLY")
    print("strict_native_seeds=3 stages_each=8 accepted=PASS_AUDITED")
    print("separate_native_rgbd_cameras=2; image_to_action_verified=False")
    print("html=" + str(output))
    print("sha256=" + sha256(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
