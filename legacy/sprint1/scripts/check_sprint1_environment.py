"""Read-only, no-network Sprint1 Windows preflight; never output secret values."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from vla_robot_sorting.semantics.gemini import GeminiModelConfig

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("numpy", "h5py", "pillow", "pytest", "ruff")
EXPECTED_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 13)


def _git(*args: str, cwd: Path = ROOT) -> str | None:
    try:
        process = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode:
        return None
    return process.stdout.strip()


def _user_scope_gemini_key_present() -> bool:
    """Windows HKCU user env; return a boolean, never the credential."""
    if os.name != "nt":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "GEMINI_API_KEY")
    except OSError:
        return False
    return isinstance(value, str) and bool(value.strip())


def make_report(*, environ: dict[str, str] | None = None) -> dict[str, object]:
    settings = os.environ if environ is None else environ
    version = sys.version_info
    valid_python = EXPECTED_PYTHON <= version[:2] < MAX_PYTHON_EXCLUSIVE
    packages = {}
    for package in DEPENDENCIES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    current_head = _git("rev-parse", "HEAD")
    clean = _git("status", "--porcelain")
    main = _git("branch", "--show-current")
    key_present = bool(settings.get("GEMINI_API_KEY", "").strip())
    user_key_present = _user_scope_gemini_key_present() if environ is None else False
    google_key_present = bool(settings.get("GOOGLE_API_KEY", "").strip())
    primary = settings.get("VLM_PRIMARY_MODEL") or GeminiModelConfig().primary_model
    fallback = settings.get("VLM_FALLBACK_MODEL") or GeminiModelConfig().fallback_model
    valid_models = bool(primary and fallback and primary != fallback)
    root_env_exists = (ROOT / ".env").exists()

    # Keep scope explicit: offline readiness is different from live cloud,
    # native simulator, model permission, and final Sprint1 acceptance.
    offline_ok = bool(
        valid_python
        and all(packages.values())
        and current_head
        and main == "main"
        and clean == ""
        and valid_models
    )
    return {
        "schema": "damh-sprint1-local-preflight-v1",
        "network_requests": 0,
        "secrets_printed": False,
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable)),
            "supported": valid_python,
        },
        "dependencies": packages,
        "git": {"branch": main, "head": current_head, "clean": clean == ""},
        "environment": {
            "gemini_key_present_in_process": key_present,
            "gemini_key_present_in_user_profile": user_key_present,
            "google_key_present_in_process": google_key_present,
            "ambiguous_sdk_key_precedence": key_present and google_key_present,
            "repo_dotenv_exists": root_env_exists,
            "repo_dotenv_autoloaded": False,
            "primary_model": primary,
            "fallback_model": fallback,
            "models_distinct": valid_models,
        },
        "gates": {
            "windows_offline": "PASS" if offline_ok else "BLOCKED",
            "gemini_credentials": (
                "READY_TO_SMOKE_TEST" if (key_present or user_key_present) else "AWAIT_USER_KEY"
            ),
            "gemini_live_response": "NOT_RUN",
            "ubuntu_native": "NOT_RUN",
            "sprint1_end_to_end": "NOT_ACCEPTED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = make_report()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gates"]["windows_offline"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
