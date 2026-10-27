"""Offline VI/EN unified pipeline preflight: audited fixture, never native actuation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from vla_robot_sorting.contracts.grasping import ParallelJawGripper
from vla_robot_sorting.contracts.semantic import SourceLanguage
from vla_robot_sorting.orchestration.unified_sprint1 import UnifiedSprint1Pipeline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts" / "demo_semantic_to_grasp_fixture.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("vi", "en"), required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("refuse overwrite of existing preflight report")
    spec = importlib.util.spec_from_file_location("s1i_demo_fixture", FIXTURE)
    if spec is None or spec.loader is None:
        raise ValueError("fixture loader unavailable")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    language = SourceLanguage(args.language)
    # This deterministic interpreter is an explicit two-command fixture, never
    # a general-language or live Gemini model.
    known = {
        "vi": "Đưa khối đỏ vào khay trái",
        "en": "Put the red block in the left tray",
    }
    if args.command.strip() != known[args.language]:
        raise ValueError("unsupported fixture command; no semantic model call")
    reference = "khối đỏ" if language is SourceLanguage.VI else "red block"
    scene = fixture.fixture_scene()
    semantic = fixture.fixture_semantic(args.command, language, reference)
    report = UnifiedSprint1Pipeline().run(
        command=args.command,
        language=language,
        scene=scene,
        semantic=semantic,
        semantic_command_sha256=sha256(args.command.encode("utf-8")).hexdigest(),
        gripper=ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08),
    )
    payload = {
        "schema": "sprint1-unified-preflight-v1",
        "mode": "OFFLINE_SYNTHETIC_FIXTURE_NON_CONTROL",
        "command": args.command,
        "language": args.language,
        "semantic_source": "deterministic_fixture_not_gemini",
        "observation_source": "fixture_not_native_rgbd",
        "run": {**asdict(report), "status": report.status.value},
        "native_execution_attempted": False,
        "physics_steps": 0,
        "controller_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("UNIFIED_PREFLIGHT=" + report.status.value)
    print("native_execution_attempted=False")
    print("output=" + str(output))
    return 0 if report.executed is False else 2


if __name__ == "__main__":
    raise SystemExit(main())
