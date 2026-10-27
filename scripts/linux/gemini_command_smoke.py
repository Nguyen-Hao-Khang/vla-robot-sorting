#!/usr/bin/env python3
"""Ask Gemini to turn one free EN/VI command into a structured pick-and-place intent.

Step 2 of the interactive digital twin: proves the API key and model work on this
machine and shows how a free command is grounded against the objects actually in
the scene. One API request per call (primary model, then the fallback only if the
primary fails). No robot motion. The key is read from GEMINI_API_KEY and never printed.
  python3 scripts/linux/gemini_command_smoke.py "put the blue block in the left tray"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from vla_robot_sorting.semantics.gemini import (  # noqa: E402
    GeminiModelConfig,
    GeminiRESTTransport,
    SemanticProviderError,
)

DEFAULT_SCENE = ["pill bottle (white bottle, orange cap)", "pad (blue square)"]

SYSTEM = (
    "You convert a robot operator command (Vietnamese or English) into one structured "
    "pick-and-place intent for a single robot arm. Use ONLY objects and places from the "
    "provided scene list. If the command names something not in the scene, is ambiguous, "
    "negated, or unsafe, set decision to 'refuse' or 'clarify'. Always write 'reason' in "
    "short English. Never invent objects."
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["pick_and_place", "refuse", "clarify"]},
        "language": {"type": "string", "enum": ["vi", "en", "mixed"]},
        "target_object": {"type": ["string", "null"]},
        "destination": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "language", "target_object", "destination", "reason"],
}


def build_prompt(command: str, scene: list[str]) -> str:
    listing = "\n".join(f"- {item}" for item in scene)
    return f"Scene objects and places:\n{listing}\n\nOperator command:\n{command}"


def validate(payload: dict, scene: list[str]) -> list[str]:
    problems = []
    if payload.get("decision") not in ("pick_and_place", "refuse", "clarify"):
        problems.append("decision")
    if payload.get("decision") == "pick_and_place":
        for key in ("target_object", "destination"):
            value = payload.get(key)
            if not value or not any(value.split(" (")[0] in item for item in scene):
                problems.append(f"{key}_not_in_scene")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command")
    parser.add_argument(
        "--scene", action="append", help="scene item (repeatable); default: pill bottle + pad"
    )
    args = parser.parse_args(argv)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY_MISSING: export GEMINI_API_KEY in this terminal.")
        return 3
    scene = args.scene or DEFAULT_SCENE
    defaults = GeminiModelConfig()
    config = GeminiModelConfig(
        primary_model=os.environ.get("VLM_PRIMARY_MODEL", defaults.primary_model),
        fallback_model=os.environ.get("VLM_FALLBACK_MODEL", defaults.fallback_model),
    )
    transport = GeminiRESTTransport(api_key=key, timeout_s=60)
    for model in (config.primary_model, config.fallback_model):
        try:
            payload = transport.generate_json(
                model_id=model,
                system_instruction=SYSTEM,
                user_prompt=build_prompt(args.command, scene),
                response_schema=SCHEMA,
            )
        except SemanticProviderError as exc:
            print(f"MODEL_FAILED model={model} error={exc}")
            continue
        print(f"MODEL={model}")
        print("INTENT=" + json.dumps(payload, ensure_ascii=False))
        problems = validate(payload, scene)
        print("SCENE_CHECK=" + ("OK" if not problems else ",".join(problems)))
        return 0
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
