#!/usr/bin/env python3
"""Inference latency and GPU memory of a candidate policy on the local GPU (ADR 0005).

Builds the policy for the features of lerobot/svla_so100_pickplace, takes one real frame and
times ``predict_action_chunk`` (one full action chunk) after two warm-up calls. The weights
of the action head are untrained; latency and memory do not depend on them.

  python tools/train/infer_bench.py smolvla --pretrained <local copy of lerobot/smolvla_base>
  python tools/train/infer_bench.py evo1          # LeRobot e581c1d or later
"""

from __future__ import annotations

import argparse
import json
import time

SMOLVLA_RENAME = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=("smolvla", "evo1"))
    parser.add_argument("--pretrained", default=None, help="SmolVLA checkpoint directory")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--flow-steps", type=int, default=None, help="Evo-1 denoising steps (default: model's 32)"
    )
    args = parser.parse_args(argv)

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    dataset = LeRobotDataset("lerobot/svla_so100_pickplace", episodes=[0])
    if args.model == "smolvla":
        from lerobot.configs.policies import PreTrainedConfig

        if not args.pretrained:
            parser.error("smolvla needs --pretrained")
        cfg = PreTrainedConfig.from_pretrained(args.pretrained)
        cfg.pretrained_path = args.pretrained
        rename = SMOLVLA_RENAME
    else:
        from lerobot.policies.evo1.configuration_evo1 import Evo1Config

        # Turing GPUs have neither bfloat16 nor flash attention.
        cfg = Evo1Config(dtype=torch.float16, use_amp=False, use_flash_attn=False)
        if args.flow_steps:
            cfg.num_inference_timesteps = args.flow_steps
        rename = {}
    cfg.device = "cuda"
    try:
        policy = make_policy(cfg, ds_meta=dataset.meta, rename_map=rename)
    except TypeError:  # LeRobot 0.4.4 has no rename_map argument
        policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.eval()
    pre, _post = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    item = dataset[0]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items()}
    for old, new in rename.items():
        if old in batch:
            batch[new] = batch.pop(old)
    batch = pre(batch)
    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.inference_mode():
        for i in range(args.repeats + 2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            chunk = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            if i >= 2:
                times.append(time.perf_counter() - start)
    times.sort()
    print(
        json.dumps(
            {
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "chunk_shape": list(chunk.shape),
                "median_chunk_ms": round(1000 * times[len(times) // 2], 1),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
