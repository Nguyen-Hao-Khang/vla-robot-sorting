# ruff: noqa: E501, RUF001  # HTML/CSS/JavaScript literals and intentional typographic labels
"""Self-contained, offline S1-I geometry replay dashboard.

Trust boundary: this renderer accepts only the v2 synthetic fixture contract.
It cannot certify native execution, motion planning, or robot safety.
"""

from __future__ import annotations

import json
from typing import Any


def render_fixture_dashboard(data: dict[str, Any]) -> str:
    """Render an evidence-labeled standalone HTML report, without external assets."""
    if (
        data.get("schema") != "semantic-to-grasp-fixture-demo-v2"
        or data.get("source") != "synthetic_fixture"
        or data.get("network_calls") != 0
        or data.get("native_simulation_calls") != 0
        or data.get("physics_steps") != 0
    ):
        raise ValueError("only offline synthetic v2 fixture evidence can be rendered")
    if not data.get("cases") or not data.get("scene"):
        raise ValueError("fixture must include scene and cases")
    for case in data["cases"]:
        if (
            case.get("feasibility_checked") is not False
            or case.get("executed") is not False
            or not case.get("candidates")
            or case.get("candidate_count") != len(case["candidates"])
        ):
            raise ValueError("cases must have unexecuted geometry-only candidates")
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace(
        "&", "\\u0026"
    ).replace("<", "\\u003c").replace(">", "\\u003e").replace(
        "\u2028", "\\u2028"
    ).replace("\u2029", "\\u2029")
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S1-I | Semantic to grasp geometry replay</title>
<style>
:root{color-scheme:dark;--bg:#090f19;--panel:#121d2c;--panel2:#182638;--line:#2d4056;--text:#e9f1fa;--muted:#9fb1c5;--cyan:#65dfdc;--amber:#ffc96e;--red:#ff8c88;--green:#8de4ae}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 70% 10%,#17283a,var(--bg) 55%);color:var(--text);font:14px/1.55 system-ui,-apple-system,"Segoe UI",Arial,sans-serif}
header,.body{max-width:1440px;margin:auto;padding:25px 30px}header{padding-bottom:10px}
h1{font-size:29px;letter-spacing:-.6px;margin:4px 0}h2{font-size:15px;letter-spacing:.4px;margin:0 0 13px}h3{font-size:13px;margin:15px 0 7px}p{margin:6px 0;color:var(--muted)}
.kicker{color:var(--cyan);text-transform:uppercase;letter-spacing:2.4px;font-size:11px;font-weight:800}
.flag{display:inline-block;border:1px solid #8c6228;color:var(--amber);background:#352818;border-radius:6px;padding:4px 9px;font-size:11px;font-weight:800;margin:4px 8px 4px 0}
.flag.dark{border-color:var(--line);background:#182738;color:var(--muted)}
.body{display:grid;grid-template-columns:260px minmax(460px,1fr) 330px;gap:16px;align-items:start;padding-top:10px}
.panel{border:1px solid var(--line);border-radius:14px;padding:18px;background:linear-gradient(160deg,#142235,#101a29);min-width:0;box-shadow:0 8px 22px #0002}
aside{display:flex;flex-direction:column;gap:14px}
.case{display:block;width:100%;text-align:left;color:var(--text);background:var(--panel2);border:1px solid #2d4056;padding:12px;border-radius:9px;cursor:pointer;margin-bottom:8px;font:inherit}
.case:hover,.candidate:hover{border-color:var(--cyan)}.case.active,.candidate.active{border-color:var(--cyan);box-shadow:inset 3px 0 var(--cyan)}
.case small{display:block;color:var(--muted);font-size:11px;margin-top:3px}.status{font-size:10px;font-weight:800;letter-spacing:.8px;color:var(--cyan)}.status.bad{color:var(--red)}
.step{border-left:2px solid #36536d;padding:3px 10px;margin:11px 0}.step strong{display:block;font-size:12px}.step span{color:var(--muted);font-size:11px}
.sceneTitle{display:flex;justify-content:space-between;gap:8px;align-items:center}.meta{color:var(--muted);font-size:11px}
svg{width:100%;display:block;background:#0a1423;border:1px solid #294159;border-radius:9px}
.drawing{margin-top:11px}.legend{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin-top:9px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
#candidateList{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:12px}
.candidate{font:12px system-ui;color:var(--text);border:1px solid var(--line);background:#19293b;border-radius:8px;text-align:left;padding:9px;cursor:pointer}
.candidate small{display:block;color:var(--muted);font-size:10px}
.mono,pre{font:11px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;white-space:pre-wrap;color:#c4d4e4}
pre{background:#091421;border:1px solid #2b4053;padding:12px;border-radius:9px;margin:0}
.kv{display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid #293b50;padding:7px 0;font-size:11px}.kv span{color:var(--muted)}.kv strong{font-size:11px;overflow-wrap:anywhere;text-align:right}
.warning{background:#332b1b;color:#ffe0a2;border:1px solid #6e502d;border-radius:9px;padding:12px;font-size:12px;margin-top:12px}
.good{color:var(--green)}.bad{color:var(--red)}#diagnosis{min-height:40px}
footer{max-width:1440px;margin:auto;padding:12px 30px 35px;color:#90a2b7;font-size:12px}
@media(max-width:1150px){.body{grid-template-columns:230px 1fr}.inspector{grid-column:1/-1}}
@media(max-width:740px){header,.body{padding:15px}.body{display:block}.panel,aside{margin-bottom:14px}#candidateList{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header>
<div class="kicker">VLA Robot Sorting · Sprint 1 / S1-I</div>
<h1>Semantic → Grounding → Grasp geometry</h1>
<p>Interactive scene and 6-DoF grasp proposal replay. Data derived from the repository's deterministic fixture, not a live camera or robot.</p>
<span class="flag">SYNTHETIC FIXTURE · NO PHYSICS</span>
<span class="flag dark">GEMINI: NOT CALLED</span>
<span class="flag dark">IK / COLLISION: NOT TESTED</span>
<span class="flag dark">EXECUTION: NOT RUN</span>
</header>
<div class="body">
<aside>
<section class="panel"><h2>01 · Command / scenario</h2><div id="cases"></div></section>
<section class="panel"><h2>02 · Evidence stages</h2><div id="stages"></div></section>
</aside>
<main class="panel">
<div class="sceneTitle"><h2>03 · Fixture scene (world coordinates)</h2><span class="meta" id="sceneRevision"></span></div>
<p id="scenarioText"></p>
<div class="drawing"><div class="meta">TOP VIEW · X–Y plane · metres (not robot camera imagery)</div>
<svg id="topView" viewBox="0 0 700 420" role="img" aria-label="Synthetic fixture XY diagram"></svg></div>
<div class="legend"><span><i class="dot" style="background:#fa7776"></i>target</span><span><i class="dot" style="background:#548add"></i>other object</span><span><i class="dot" style="background:#8de4ae"></i>destination</span><span><i class="dot" style="background:#ffc96e"></i>pregrasp → grasp</span></div>
<div class="drawing"><div class="meta">SIDE VIEW · X–Z plane · selected candidate only</div>
<svg id="sideView" viewBox="0 0 700 220" role="img" aria-label="Fixture XZ grasp geometry projection"></svg></div>
<h3>Geometry proposals · ranked by raw geometric heuristic, not by physical feasibility</h3>
<div id="candidateList"></div>
</main>
<section class="panel inspector">
<h2>04 · Inspection / verification</h2>
<div id="facts"></div>
<h3>Selected 6-DoF candidate</h3><pre id="candidateDetails">Select a scenario.</pre>
<h3>Fail-closed diagnostics</h3><div id="diagnosis"></div>
<div class="warning"><strong>Safety boundary</strong><br>These coordinates are geometric proposals only. Ranking does not imply reachability, collision clearance, safe grip, trajectory, contact-free retreat, or successful sorting. Never send directly to actuators.</div>
</section>
</div>
<footer>Standalone offline artifact. Scene objects, normalized intents, candidate poses and rejection codes are computed by project code; visual geometry is an explanatory projection, not a RoboTwin screenshot.</footer>
<script id="fixtureData" type="application/json">__EVIDENCE_JSON__</script>
<script>
"use strict";
const evidence=JSON.parse(document.getElementById("fixtureData").textContent);
const scenarios=[...evidence.cases.map(c=>({type:"pass",value:c})) ,...evidence.rejections.map(c=>({type:"reject",value:c}))];
let selectedCase=0,selectedCandidate=0;
const NS="http://www.w3.org/2000/svg";
function el(tag,attrs={},textValue){const node=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));if(textValue!==undefined)node.textContent=String(textValue);return node}
function node(tag,content,cls){const item=document.createElement(tag);if(cls)item.className=cls;if(content!==undefined)item.textContent=String(content);return item}
function clear(elem){elem.replaceChildren()}
const fmt=v=>typeof v==="number"?v.toFixed(3):String(v);
const vec=a=>"["+a.map(fmt).join(", ")+"]";
function plot(svg,side,sc,candidate){
  clear(svg);
  const yy=side?175:235,scale=side?470:470;
  const point=p=>[350+p[0]*scale,yy-(side?p[2]:p[1])*scale];
  for(let n=-5;n<=5;n++){
    const pos=350+n*47;
    svg.appendChild(el("line",{x1:pos,y1:0,x2:pos,y2:side?220:420,stroke:"#1c3045","stroke-width":1}));
    const posY=yy+n*47;
    svg.appendChild(el("line",{x1:0,y1:posY,x2:700,y2:posY,stroke:"#1c3045","stroke-width":1}));
  }
  svg.appendChild(el("line",{x1:0,y1:yy,x2:700,y2:yy,stroke:"#47627a","stroke-dasharray":"4 5"}));
  svg.appendChild(el("line",{x1:350,y1:0,x2:350,y2:side?220:420,stroke:"#47627a","stroke-dasharray":"4 5"}));
  svg.appendChild(el("text",{x:12,y:20,fill:"#9fb1c5","font-size":13},"+X →"));
  svg.appendChild(el("text",{x:12,y:37,fill:"#9fb1c5","font-size":13},side?"+Z ↑":"+Y ↑"));
  for(const dest of sc.destinations){
    const [x,y]=point(dest.pose.position_xyz),t=dest.tolerance_xyz_m;
    const w=Math.max(t[0]*2*scale,32),h=Math.max((side?t[2]:t[1])*2*scale,20);
    svg.appendChild(el("rect",{x:x-w/2,y:y-h/2,width:w,height:h,rx:5,fill:"#8de4ae33",stroke:"#8de4ae","stroke-dasharray":"5 3"}));
    svg.appendChild(el("text",{x:x+18,y:y-17,fill:"#9be7bb","font-size":13},dest.destination_id));
  }
  for(const obj of sc.objects){
    const [x,y]=point(obj.pose.position_xyz),sizes=obj.size_xyz_m;
    const w=Math.max(sizes[0]*scale,20),h=Math.max((side?sizes[2]:sizes[1])*scale,20);
    const target=obj.object_id===(scenarios[selectedCase].value.target||"red-block");
    const color=target?"#fa7776":"#548add";
    svg.appendChild(el("rect",{x:x-w/2,y:y-h/2,width:w,height:h,rx:3,fill:obj.visibility==="visible"?color+"80":"#64768b22",stroke:color,"stroke-width":2}));
    svg.appendChild(el("text",{x:x+12,y:y-17,fill:"#ddeafa","font-size":13},obj.object_id+(obj.visibility==="visible"?"":" [OCCLUDED]")));
  }
  if(candidate){
    const p=point(candidate.pregrasp_pose.position_xyz),q=point(candidate.grasp_pose.position_xyz);
    svg.appendChild(el("line",{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:"#ffc96e","stroke-width":4,"stroke-linecap":"round"}));
    svg.appendChild(el("circle",{cx:p[0],cy:p[1],r:7,fill:"#ffc96e",stroke:"#563c19","stroke-width":2}));
    svg.appendChild(el("circle",{cx:q[0],cy:q[1],r:5,fill:"#fdf5d8"}));
    svg.appendChild(el("text",{x:p[0]+10,y:p[1]+22,fill:"#ffe2aa","font-size":12},"pregrasp"));
  }
}
function kv(parent,key,value){const row=node("div",undefined,"kv");row.append(node("span",key),node("strong",value));parent.append(row)}
function stage(parent,label,status,detail){const row=node("div",undefined,"step");row.append(node("strong",label+" · "+status),node("span",detail));parent.append(row)}
function render(){
  const scenario=scenarios[selectedCase],record=scenario.value;
  const pass=scenario.type==="pass",candidate=pass?record.candidates[selectedCandidate]:null;
  const scenarioScene=pass?evidence.scene:record.scene;
  const caseBox=document.getElementById("cases");clear(caseBox);
  scenarios.forEach((entry,i)=>{
    const value=entry.value,b=node("button",entry.type==="pass"?value.language.toUpperCase()+" · "+value.command:value.label,"case"+(i===selectedCase?" active":""));
    b.type="button";
    b.prepend(node("span",entry.type==="pass"?"PROPOSAL":"SAFE REJECTION","status"+(entry.type==="reject"?" bad":"")));
    b.append(node("small",entry.type==="pass"?value.candidate_count+" geometry proposals":value.failure_code));
    b.addEventListener("click",()=>{selectedCase=i;selectedCandidate=0;render()});caseBox.append(b)
  });
  document.getElementById("sceneRevision").textContent=scenarioScene.scene_revision+" · "+scenarioScene.transform_revision;
  document.getElementById("scenarioText").textContent=pass?record.command:"Adversarial test: "+record.label;
  const steps=document.getElementById("stages");clear(steps);
  stage(steps,"Structured intent",pass?"SCHEMA VALID":"INPUT / SCENE CHANGED",pass?"Strict fixture schema; no language-model call":"One fault injected into an otherwise valid fixture");
  stage(steps,pass?"Catalog grounding":"Bridge safety gate",pass?"MATCHED":"REFUSED",pass?record.target+" → "+record.destination:record.failure_code);
  stage(steps,"Geometric grasp",pass?"GENERATED":"NO PROPOSAL",pass?record.candidate_count+" version-locked 6-DoF proposals":record.reason);
  stage(steps,"IK / collision","NOT MEASURED","No planner or native collision test invoked");
  stage(steps,"Robot execution","NOT RUN","0 native calls; 0 physics steps");
  plot(document.getElementById("topView"),false,scenarioScene,candidate);
  plot(document.getElementById("sideView"),true,scenarioScene,candidate);
  const list=document.getElementById("candidateList");clear(list);
  if(pass)record.candidates.forEach((c,i)=>{
    const b=node("button","#"+(i+1)+" · "+fmt(c.raw_quality),"candidate"+(i===selectedCandidate?" active":""));b.type="button";
    b.append(node("small",c.candidate_id));b.addEventListener("click",()=>{selectedCandidate=i;render()});list.append(b)
  });else list.append(node("p","No grasp candidates returned; safe rejection."));
  const facts=document.getElementById("facts");clear(facts);
  kv(facts,"Evidence source",evidence.source);kv(facts,"Semantic provider",pass?record.semantic_source:"fixture_only");
  kv(facts,"Normalized intent",pass?record.normalized_intent_id:"not executable");
  kv(facts,"Observation revision",record.observation_revision||scenarioScene.observation_revision);
  kv(facts,"Geometry source",pass?record.geometry_source:"estimated_proxy");
  kv(facts,"Physical feasibility","NOT CHECKED");kv(facts,"Execution","NOT RUN");
  const details=document.getElementById("candidateDetails");
  details.textContent=candidate?[
    "ID: "+candidate.candidate_id,
    "Grasp xyz [m]: "+vec(candidate.grasp_pose.position_xyz),
    "Grasp xyzw: "+vec(candidate.grasp_pose.orientation_xyzw),
    "Pregrasp xyz [m]: "+vec(candidate.pregrasp_pose.position_xyz),
    "Pregrasp xyzw: "+vec(candidate.pregrasp_pose.orientation_xyzw),
    "Approach vector: "+vec(candidate.approach_direction),
    "Closing vector: "+vec(candidate.closing_direction),
    "Jaw width [m]: "+fmt(candidate.jaw_width_m),
    "Raw geometric quality: "+fmt(candidate.raw_quality),
    "Coordinate frame: "+candidate.grasp_pose.frame_id
  ].join("\n"):"No candidate (fail closed).";
  const diag=document.getElementById("diagnosis");clear(diag);
  diag.append(node("div",pass?"SUCCESS: geometry proposal only. No motion acceptance is implied.":"REJECTED: "+record.failure_code,pass?"good":"bad"));
  diag.append(node("p",pass?evidence.claim_boundary:record.reason));
}
render();
</script>
</body></html>"""
    return html.replace("__EVIDENCE_JSON__", serialized)
