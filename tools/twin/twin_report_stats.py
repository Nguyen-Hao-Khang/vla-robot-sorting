#!/usr/bin/env python3
"""Summarise digital-twin session logs (twin_session_*.jsonl).

Reads only the JSONL logs the twin console writes; nothing is sent anywhere. Counts:
  * commands per decision (execute / refuse / clarify) and per final result;
  * pick-and-place steps (one per object move the robot attempted) and how many succeeded;
  * the motion where a failed step stopped (approach, grasp, lift, transit, place, ...);
  * steps per object shape (needs twin_scene.py next to this file, otherwise "unknown");
  * planning time of the language model (seconds per plan);
  * per command (from command_summary): objects asked for vs objects finished after
    rescans and replans.

Usage (inside the repo, any Python 3.10+):
  python tools/twin/twin_report_stats.py                 # logs in ~/Downloads
  python tools/twin/twin_report_stats.py --logs DIR --out DIR --since-version <twin_version>
Writes summary.md, steps.csv and commands.csv to --out (default ~/Downloads/twin_report).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCAN_RE = re.compile(
    r"matched (\d+), missed (\d+), extra (\d+), labels ok (\d+)/(\d+), mean xy error "
    r"(?:n/a|([\d.]+) cm)"
)
FALLBACK_SHAPES = (
    ("tetrahedron", "tetrahedron"),
    ("prism", "prism"),
    ("cylinder", "cylinder"),
    ("ball", "sphere"),
    ("volleyball", "sphere"),
    ("bottle", "bottle"),
    ("carton", "carton"),
    ("box", "box"),
    ("cube", "cube"),
)


def shape_table() -> dict[str, str]:
    try:
        import twin_scene as ts

        return {o.name: o.shape for lv in ts.LEVELS.values() for o in lv.objects}
    except Exception:  # optional; keyword fallback below
        return {}


def kind_right(label: str, truth_name: str) -> bool | None:
    """Object kind named correctly (shape, or the item keyword for look-alikes), colour ignored."""
    try:
        import twin_perception as tp
        import twin_scene as ts
    except Exception:  # optional
        return None
    spec = {o.name: o for lv in ts.LEVELS.values() for o in lv.objects}.get(truth_name)
    if spec is None:
        return None
    text = label.casefold()
    if spec.keywords:
        tokens = set(re.findall(r"[a-z0-9]+", text))
        return any(
            w.casefold() in tokens or (len(w) > 2 and w.casefold() in text) for w in spec.keywords
        )
    return any(w in text for w in tp.SHAPE_WORDS.get(spec.shape, (spec.shape,)))


def shape_of(name: str, table: dict[str, str]) -> str:
    if name in table:
        return table[name]
    low = name.lower()
    for key, shape in FALLBACK_SHAPES:
        if key in low:
            return shape
    return "unknown"


def version_key(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")) or (0,)


def find_logs(root: Path) -> list[Path]:
    best: dict[str, Path] = {}
    for p in root.rglob("*twin_session_*.jsonl"):
        m = re.search(r"twin_session_\d{8}T\d{6}Z", p.name)
        key = m.group(0) if m else p.name
        if key not in best or p.stat().st_size > best[key].stat().st_size:
            best[key] = p  # the same session copied twice: keep the longer file
    return sorted(best.values(), key=lambda p: p.name)


def first_failed_motion(outcome: dict) -> str:
    for m in outcome.get("motions") or []:
        if not m.get("ok", True):
            return str(m.get("name", "?"))
    if outcome.get("ok"):
        return ""
    text = str(outcome.get("summary", ""))
    if "did not move" in text or "no room" in text:
        return "refused before moving (no room / destination full)"
    if "fell over" in text or "outside" in text:
        return "placement check (fell over / outside)"
    return "other"


def object_of(outcome: dict, plan_row) -> str:
    if plan_row:
        return str(plan_row[0])
    s = str(outcome.get("summary", ""))
    return s.split(" is ")[0].split(":")[0].strip() or "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--out", type=Path, default=Path.home() / "Downloads" / "twin_report")
    ap.add_argument("--since-version", default="", help="only sessions of this version or newer")
    args = ap.parse_args(argv)

    files = find_logs(args.logs)
    if not files:
        print(f"no twin_session_*.jsonl under {args.logs}")
        return 1
    shapes = shape_table()
    steps, commands, summaries, latency, scans = [], [], [], [], []
    used = 0
    for path in files:
        version, level = "", None
        rows = []
        version = "untagged (v1-v5)"
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = d.get("event")
            if ev in ("setup", "level"):
                version = d.get("twin_version", version) or version or "untagged (v1-v5)"
                level = d.get("level", level)
            rows.append((d, version, level))
        if args.since_version and version_key(version) < version_key(args.since_version):
            continue
        used += 1
        for d, ver, lvl in rows:
            ev = d.get("event")
            if ev == "command":
                payload = d.get("payload") or {}
                decision = payload.get("decision", "none")
                commands.append(
                    {
                        "session": path.name,
                        "version": ver,
                        "level": lvl,
                        "round": d.get("round", 0),
                        "input": d.get("input", ""),
                        "decision": decision,
                        "result": d.get("result", ""),
                        "model": d.get("model", ""),
                        "plan_seconds": d.get("seconds", ""),
                    }
                )
                if isinstance(d.get("seconds"), (int, float)) and decision != "none":
                    latency.append(float(d["seconds"]))
                plan = d.get("plan") or []
                for i, o in enumerate(d.get("outcomes") or []):
                    if o.get("skipped"):
                        continue
                    name = object_of(o, plan[i] if i < len(plan) else None)
                    steps.append(
                        {
                            "session": path.name,
                            "version": ver,
                            "level": lvl,
                            "round": d.get("round", 0),
                            "object": name,
                            "shape": shape_of(name, shapes),
                            "destination": plan[i][1] if i < len(plan) else "",
                            "ok": bool(o.get("ok")),
                            "failed_motion": "" if o.get("ok") else first_failed_motion(o),
                            "summary": o.get("summary", ""),
                        }
                    )
            elif ev == "scan" and d.get("result") == "ok":
                m = SCAN_RE.search(str(d.get("summary", "")))
                if m:
                    views = d.get("views") or []
                    kinds = [
                        kind_right(str(f.get("label", "")), f["truth"])
                        for f in d.get("fused") or []
                        if f.get("truth")
                    ]
                    scans.append(
                        {
                            "session": path.name,
                            "version": ver,
                            "level": lvl,
                            "mode": "scan (workspace + wrist)"
                            if len(views) > 1
                            else "perceive (workspace)",
                            "views": len(views),
                            "models": "; ".join(
                                sorted({c.get("model", "") for c in d.get("calls") or []})
                            ),
                            "matched": int(m.group(1)),
                            "missed": int(m.group(2)),
                            "extra": int(m.group(3)),
                            "labels_ok": int(m.group(4)),
                            "kind_ok": sum(1 for k in kinds if k),
                            "mean_error_cm": float(m.group(6)) if m.group(6) else "",
                        }
                    )
            elif ev == "command_summary":
                done = d.get("done") or []
                failed = d.get("failed") or {}
                summaries.append(
                    {
                        "session": path.name,
                        "version": ver,
                        "level": lvl,
                        "input": d.get("input", ""),
                        "asked": len(done) + len(failed),
                        "done": len(done),
                        "rounds": d.get("rounds", 0),
                        "replans": d.get("replans", 0),
                        "result": d.get("result", ""),
                        "failed_objects": "; ".join(sorted(failed)),
                    }
                )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("steps.csv", steps), ("commands.csv", commands), ("scans.csv", scans)):
        if data:
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    def rate(ok: int, n: int) -> str:
        return f"{ok}/{n} ({100 * ok / n:.0f}%)" if n else "0/0"

    out = [f"# Twin log summary ({used} sessions under {args.logs})", ""]
    out += ["## Commands", "", "| Decision | Count |", "| --- | --- |"]
    for k, v in Counter(c["decision"] for c in commands if c["round"] == 0).most_common():
        out.append(f"| {k} | {v} |")
    out += ["", "First-round commands only (replans are counted in the steps below).", ""]

    out += [
        "## Pick-and-place steps",
        "",
        f"All steps: {rate(sum(s['ok'] for s in steps), len(steps))}",
        "",
    ]
    for title, key in (
        ("By version", "version"),
        ("By level", "level"),
        ("By object shape", "shape"),
    ):
        out += [f"### {title}", "", f"| {key} | succeeded / attempted |", "| --- | --- |"]
        groups: dict = defaultdict(list)
        for s in steps:
            groups[s[key]].append(s["ok"])
        order = sorted(groups, key=version_key if key == "version" else str)
        for g in order:
            out.append(f"| {g} | {rate(sum(groups[g]), len(groups[g]))} |")
        out.append("")
    out += ["### Where failed steps stopped", "", "| motion | failed steps |", "| --- | --- |"]
    for k, v in Counter(s["failed_motion"] for s in steps if not s["ok"]).most_common():
        out.append(f"| {k} | {v} |")
    out.append("")

    if summaries:
        asked = sum(s["asked"] for s in summaries)
        done = sum(s["done"] for s in summaries)
        out += [
            "## Per command after recovery (command_summary)",
            "",
            f"Objects finished / asked: {rate(done, asked)} over {len(summaries)} commands; "
            f"commands fully done: {sum(s['result'] == 'done' for s in summaries)}",
            "",
        ]
    if latency:
        out += [
            "## Language-model planning time",
            "",
            f"{len(latency)} plans: median {statistics.median(latency):.2f} s, "
            f"min {min(latency):.2f} s, max {max(latency):.2f} s",
            "",
        ]
    if scans:
        out += [
            "## Camera perception (perceive / scan, scored against the simulator)",
            "",
            "| level | mode | scans | found / present | kind right / found "
            "| colour + kind right / found | extra | mean xy error |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        groups = defaultdict(list)
        for sc in scans:
            groups[(sc["level"], sc["mode"])].append(sc)
        for (lvl, mode), rows_ in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            found = sum(r["matched"] for r in rows_)
            present = found + sum(r["missed"] for r in rows_)
            labels = sum(r["labels_ok"] for r in rows_)
            kinds_ok = sum(r["kind_ok"] for r in rows_)
            errs = [r["mean_error_cm"] for r in rows_ if r["mean_error_cm"] != ""]
            err = f"{statistics.mean(errs):.1f} cm" if errs else "n/a"
            out.append(
                f"| {lvl} | {mode} | {len(rows_)} | {rate(found, present)} "
                f"| {rate(kinds_ok, found)} "
                f"| {rate(labels, found)} "
                f"| {sum(r['extra'] for r in rows_)} | {err} |"
            )
        out.append("")
    (args.out / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nwritten: {args.out}/summary.md, steps.csv, commands.csv, scans.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
