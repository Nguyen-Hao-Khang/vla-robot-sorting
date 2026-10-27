# V2 VLA model survey (input to decision V2-12)

**Date:** 2026-10-27 · **Status:** updated 2026-10-27; decision by bake-off on 2026-10-31 (V2-14).

## Constraints

| Item | Value |
|---|---|
| Local GPU | GTX 1660 Ti Max-Q, 6 GB. Inference only |
| Training GPU | Free cloud GPUs: Kaggle T4 16 GB ×2 or P100 16 GB; Colab T4 16 GB |
| Language | Vietnamese, English and code-switched commands |
| Ecosystem | LeRobot. The Panthera-HT vendor ships a LeRobot integration and the Twin already exports LeRobot datasets |
| Task | Single-arm tabletop sorting, about 3 trays, 8–12 object types |

## Candidates

| Model | Size / architecture | Published results | Training memory (LeRobot hardware guide, batch 8, AdamW) | Fit |
|---|---|---|---|---|
| **SmolVLA** (arXiv 2506.01844) | ~450M. SmolVLM2-500M backbone (SigLIP + SmolLM2, half the LM layers), flow-matching action expert of ~100M, 50-step chunks | LIBERO avg 87.3 (π0 86.0, OpenVLA 76.5); Meta-World 57.3; SO-100 real 78.3 (π0 61.7) | ~10–16 GB. Can be reduced with `freeze_vision_encoder=True`, a smaller batch and gradient accumulation | **Primary candidate.** Trainable on a 16 GB T4/P100 with batch ≤ 4; BF16 weights ~0.9 GB, so 6 GB inference is plausible (must be measured) |
| **Evo-1** (arXiv 2511.04555) | 0.77B. InternVL3-1B (InternViT-300M + **Qwen2.5-0.5B**, first 14 LM layers), cross-attention DiT flow-matching head | Meta-World 80.6 (SmolVLA 68.2); LIBERO 94.8 (LeRobot run 96.65); RoboTwin 37.8; real xArm6 78% vs SmolVLA 50%, π0 73% (100 demos/task, 20 trials) | Not published for small GPUs; LeRobot reference used 2×H100. Inference 2.3 GB, 16.4 Hz on an RTX 4090d | **Primary candidate with SmolVLA.** Native in LeRobot (`evo1`). No robot pretraining. Qwen2.5 lists Vietnamese among its languages. bf16 by default, so it must run fp16/fp32 on Turing |
| **ACT** | Small transformer with action chunking, **no language input** | Standard bimanual/ALOHA baselines | ~2–6 GB | **Lower-bound baseline** (one model per task). Trains locally |
| **TinyVLA** (arXiv 2409.12514, RA-L 2025) | Own Pythia-based VLMs, 0.4–1.3B (LLaVA pipeline), LoRA on the VLM + diffusion-policy head. 14 ms/action on A6000 for the 1B variant | Real Franka 5 tasks: 94.0 (H) vs OpenVLA 68.3; Meta-World 31.6. Small trial counts in the generalisation tests | Not in LeRobot; own repository | Optional. The paper does **not** describe a ModernBERT encoder, so the TA's description needs confirming. The 0.4B variant misread instructions. Pythia is English-only, a risk for Vietnamese |
| **π0 / π0-FAST / π0.5** | 3B+ | Strong | ~24–40 GB | **Not feasible** on free GPUs or for 6 GB inference |

## Vietnamese risk

None of the candidates was trained for Vietnamese. arXiv 2606.11906 measured Vietnamese LIBERO instructions:

- OpenVLA-OFT drops from 97.1 to 59.6.
- π0.5 drops from 96.9 to 57.3.

Small backbones (SmolLM2, Pythia) are English-centric, so the gap may be larger for them. This is exactly what RQ2 measures. The VLM + VLA variant is the fallback.

## Recommendation

Shortlist **SmolVLA** and **Evo-1**. Both are native in LeRobot, so switching between them costs little. Keep **ACT** as the no-language lower bound.

Decide on 2026-10-31 with a measured bake-off:

1. Inference in fp16 or fp32 on the GTX 1660 Ti: under 5 GB and at least 5 Hz.
2. A 1,000-step fine-tune on a Kaggle T4 16 GB without running out of memory. Record seconds per step and peak memory.
3. Vietnamese tokenisation: tokens per word and the share of byte-fallback tokens, comparing SmolLM2 with Qwen2.5.

If Evo-1 passes all three, it becomes the main model and SmolVLA becomes the comparison. That comparison is robot-pretrained versus multilingual backbone. Otherwise SmolVLA is the main model and Vietnamese goes through a translation step, which becomes a variable in RQ2.

Additional risk from the literature: VLAs often ignore the instruction. Examples are LIBERO-Plus, LIBERO-PRO, LangGap (arXiv 2603.00592) and SmolVLA multi-object picking (arXiv 2602.24143). The dataset must therefore pair each scene layout with several different commands.

## Sources

- SmolVLA: https://arxiv.org/abs/2506.01844 and https://huggingface.co/blog/smolvla
- Evo-1: https://arxiv.org/html/2511.04555v2 and https://huggingface.co/docs/lerobot/evo1
- TinyVLA: https://arxiv.org/html/2409.12514v5 and https://tiny-vla.github.io
- LeRobot hardware guide: https://huggingface.co/docs/lerobot/en/hardware_guide
- Multilingual VLA gap: https://arxiv.org/abs/2606.11906 and https://arxiv.org/abs/2606.15714
