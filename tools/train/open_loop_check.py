"""Open-loop check of trained checkpoints on recorded episodes (no simulator).

For each episode the policy sees the recorded observations frame by frame and its action is
compared with the recorded action. This checks that a checkpoint loads through policy_bridge,
measures latency and GPU memory, and gives a rough action error. It does not replace the
closed-loop evaluation in the simulator.

  python open_loop_check.py --bridge-dir <folder with policy_bridge.py> --out results.json \
      --model smolvla=<ckpt>[@wrist=camera1,workspace=camera2] --model act=<ckpt> \
      --episodes <raw dir>:<n> [<raw dir>:<n> ...]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def pick_episodes(raw: Path, n: int) -> list[Path]:
    """n episodes spread evenly, half English and half Vietnamese where possible."""
    eps = sorted(p for p in raw.glob("episode_*") if (p / "meta.json").exists())
    by_lang = {"en": [], "vi": []}
    for p in eps:
        lang = json.loads((p / "meta.json").read_text(encoding="utf-8")).get("language", "en")
        by_lang.setdefault(lang, []).append(p)
    out = []
    for lang in ("en", "vi"):
        group = by_lang.get(lang, [])
        k = min(len(group), n // 2 if lang == "en" else n - n // 2)
        if k:
            idx = np.linspace(0, len(group) - 1, k).round().astype(int)
            out += [group[i] for i in idx]
    return out


def run_episode(runner, folder: Path, camera_names: list[str], torch) -> dict:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    states = np.load(folder / "states.npy")
    actions = np.load(folder / "actions.npy")
    runner.reset()
    preds, times = [], []
    for t in range(len(states)):
        images = {c: (folder / c / f"{t:06d}.jpg").read_bytes() for c in camera_names}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds.append(runner.act(states[t].tolist(), images, meta["task"]))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    preds = np.asarray(preds, np.float32)
    err = np.abs(preds - actions)
    hold = np.abs(states - actions)  # trivial baseline: target = current state
    slow = [x for x in times if x > 0.05]  # calls that ran the model (new chunk)
    return {
        "episode": folder.name,
        "language": meta.get("language"),
        "task": meta["task"],
        "frames": len(states),
        "joint_mae_rad": float(err[:, :6].mean()),
        "gripper_mae": float(err[:, 6].mean()),
        "hold_joint_mae_rad": float(hold[:, :6].mean()),
        "hold_gripper_mae": float(hold[:, 6].mean()),
        "final_joint_err_rad": float(err[-1, :6].mean()),
        "chunk_calls": len(slow),
        "chunk_latency_s": statistics.median(slow) if slow else None,
        "step_latency_s": statistics.median(times),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-dir", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="label=ckpt[@cameramap]")
    ap.add_argument("--episodes", nargs="+", required=True, help="raw_dir:n")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.bridge_dir))
    import torch
    from policy_bridge import PolicyRunner, parse_camera_map

    sets = []
    for spec in args.episodes:
        raw, _, n = spec.rpartition(":")
        sets.append((Path(raw).name, pick_episodes(Path(raw), int(n))))
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for spec in args.model:
        label, _, rest = spec.partition("=")
        ckpt, _, cmap = rest.partition("@")
        if label in results:
            print("skip", label, "(already in", args.out, ")")
            continue
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        runner = PolicyRunner(Path(ckpt), "auto", parse_camera_map(cmap))
        load_s = time.perf_counter() - t0
        cams = ["wrist", "workspace"]
        rows = []
        for set_name, eps in sets:
            for folder in eps:
                row = run_episode(runner, folder, cams, torch)
                row["set"] = set_name
                rows.append(row)
                print(label, set_name, row["episode"], row["language"],
                      f"joint {row['joint_mae_rad']:.4f} (hold {row['hold_joint_mae_rad']:.4f})",
                      f"grip {row['gripper_mae']:.3f} chunk {row['chunk_latency_s']}", flush=True)
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        results[label] = {"checkpoint": ckpt, "policy_type": runner.policy_type,
                          "load_s": load_s, "peak_gpu_gb": peak, "episodes": rows}
        args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        del runner
    print("OPEN_LOOP_DONE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
