# VLA Robot Sorting — Panthera-HT

Course project HK261-DAGD1-393: **Phân loại vật thể sử dụng cánh tay robot và công nghệ Vision-Language-Action (VLA)**.

Team: Văn Quốc Huy, Nguyễn Hảo Khang, Phan Gia Phúc.

## Start here

1. [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md): current plan (V2), research questions, experiments and schedule.
2. [`docs/DECISIONS.md`](docs/DECISIONS.md): decision ledger.
3. [`docs/SETUP.md`](docs/SETUP.md): what runs on Windows, Ubuntu or Kaggle, and how to set it up.
4. [`docs/HISTORY.md`](docs/HISTORY.md): what Sprint 1 and the digital twin delivered.
5. [`CONTRIBUTING.md`](CONTRIBUTING.md): branch, commit, code and review rules.

## Repository map

| Path | Content |
|---|---|
| `src/vla_robot_sorting/` | Project package: contracts, simulation adapters, perception, semantics (Gemini), grasping, planning, control, safety, orchestration, evaluation, evidence |
| `scripts/linux/twin_*.py` | Digital twin (RoboTwin), the scripted demonstration expert and the LeRobot data tools |
| `scripts/linux/probe_*.py`, `run_*.sh` | Sprint 1 native probes and session runners |
| `scripts/*.py` | Offline tools: scenarios, evaluation, evidence audit, demos |
| `tests/` | Unit, contract and scenario tests (no simulator required) |
| `configs/` | Pinned upstream versions, gates, camera presets |
| `experiments/accepted/` | Accepted Sprint 1 evidence manifests |
| `notebooks/` | Kaggle training notebook |
| `report/` | Project report (LaTeX, Vietnamese) |
| `docs/adr/`, `docs/architecture/`, `docs/research/` | Architecture decisions, interface notes, technical studies, VLA survey, literature register |

## License

Code is released under the [MIT License](LICENSE).
