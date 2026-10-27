"""CLI entrypoint for independently audited RoboTwin RGB-D evidence bundles."""

from __future__ import annotations

import sys
from pathlib import Path

# Direct CLI execution on pinned Ubuntu does not imply editable src install.
# Import the matching local project package, never an unrelated environment copy.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from vla_robot_sorting.perception.native_depth_audit import (  # noqa: E402
    DepthEvidenceError,
    audit_native_depth_bundle,
    main,
)

__all__ = ["DepthEvidenceError", "audit_native_depth_bundle", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
