"""Native RGB-D auditor must run from scripts/ with no PYTHONPATH workaround."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_depth_auditor_direct_python_cli_imports_local_src():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_native_depth_capture.py"), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "native" in result.stdout.lower() or "bundle" in result.stdout.lower()
