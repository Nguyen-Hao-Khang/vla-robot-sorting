from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from scripts.audit_native_manipulation_export_integrity import (
    IntegrityError,
    _expected_rows,
    verify_nested_evidence,
    verify_top_level,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_top_level_sha256_requires_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.md"
    path.write_bytes(b"immutable")
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text(f"{_sha(b'immutable')}  RESULT.md\n", encoding="ascii")
    assert verify_top_level(tmp_path) == 1
    path.write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="mismatch"):
        verify_top_level(tmp_path)


def test_checksum_manifest_forbids_traversal_and_duplicates() -> None:
    good = "a" * 64
    with pytest.raises(IntegrityError, match="unsafe"):
        _expected_rows(f"{good}  ../outside.json\n")
    with pytest.raises(IntegrityError, match="unsafe"):
        _expected_rows(f"{good}  /absolute.json\n")
    with pytest.raises(IntegrityError, match="duplicate"):
        _expected_rows(f"{good}  A.json\n{good}  A.json\n")
    with pytest.raises(IntegrityError, match="empty"):
        _expected_rows("")


def _make_nested(tmp_path: Path, corrupt: bool = False) -> Path:
    archive = tmp_path / "evidence.tar.gz"
    prefix = "sprint1_manipulation_campaign_mock"
    head = ("1" * 40 + "\n").encode()
    manifest = (
        f"{_sha(head)}  ./git-head.txt\n"
        f"{_sha(b'expected')}  ./probe.json\n"
    ).encode()
    sources = {
        f"{prefix}/git-head.txt": head,
        f"{prefix}/probe.json": b"altered" if corrupt else b"expected",
        f"{prefix}/SHA256SUMS.txt": manifest,
    }
    with tarfile.open(archive, "w:gz") as output:
        for name, content in sources.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
    return archive


def test_nested_sha_valid_and_exported_git_head(tmp_path: Path) -> None:
    assert verify_nested_evidence(_make_nested(tmp_path)) == (
        2,
        "sprint1_manipulation_campaign_mock",
        "1" * 40,
    )


def test_nested_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match="SHA mismatch"):
        verify_nested_evidence(_make_nested(tmp_path, corrupt=True))
