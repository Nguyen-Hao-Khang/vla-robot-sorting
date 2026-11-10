"""Fail-closed tests for the offline native evidence replay builder."""
from __future__ import annotations

import importlib.util
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "sprint1_native_evidence_demo",
    ROOT / "scripts" / "build_sprint1_native_evidence_demo.py",
)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def create_manifest(tmp_path: Path, files: dict[str, bytes]) -> None:
    lines = []
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        lines.append(f"{sha256(data).hexdigest()}  ./{name}")
    (tmp_path / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verified_sha_manifest(tmp_path: Path) -> None:
    create_manifest(tmp_path, {"probe.json": b"{}", "sub/a.log": b"AUDIT=PASS\n"})
    assert mod.verify_manifest(tmp_path) == 2


def test_modified_evidence_rejected(tmp_path: Path) -> None:
    create_manifest(tmp_path, {"probe.json": b"{}"})
    (tmp_path / "probe.json").write_bytes(b'{"passed":true}')
    with pytest.raises(ValueError, match="modified evidence"):
        mod.verify_manifest(tmp_path)


def test_missing_evidence_rejected(tmp_path: Path) -> None:
    create_manifest(tmp_path, {"probe.json": b"{}"})
    (tmp_path / "probe.json").unlink()
    with pytest.raises(ValueError, match="missing or modified"):
        mod.verify_manifest(tmp_path)


def test_path_traversal_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_bytes(b"secret")
    (tmp_path / "SHA256SUMS.txt").write_text(
        f"{sha256(b'secret').hexdigest()}  ../outside.txt\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe"):
        mod.verify_manifest(tmp_path)


def test_duplicate_evidence_row_rejected(tmp_path: Path) -> None:
    create_manifest(tmp_path, {"probe.json": b"{}"})
    with (tmp_path / "SHA256SUMS.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{sha256(b'{}').hexdigest()}  probe.json\n")
    with pytest.raises(ValueError, match="repeated"):
        mod.verify_manifest(tmp_path)


def test_bad_sha_line_rejected(tmp_path: Path) -> None:
    (tmp_path / "SHA256SUMS.txt").write_text("something else\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        mod.verify_manifest(tmp_path)


def test_empty_sha_manifest_rejected(tmp_path: Path) -> None:
    (tmp_path / "SHA256SUMS.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=r"empty|missing|oversized"):
        mod.verify_manifest(tmp_path)


def test_nonfinite_joint_values_not_exported() -> None:
    assert mod._joint_values([0.0, float("nan"), float("inf")]) == [0.0, None, None]


def test_demo_has_explicit_non_live_provenance() -> None:
    assert "KHÔNG PHẢI mô phỏng live" in mod.HTML
    assert "không Gemini live" in mod.HTML
    assert "Không đồng bộ hình ảnh" in mod.HTML
    assert "<script src=" not in mod.HTML
    assert "@@DATA@@" in mod.HTML
    assert "fetch(" not in mod.HTML
    assert "WebSocket" not in mod.HTML
