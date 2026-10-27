"""Regression coverage for cross-OS EOL equivalence without weakening source locks."""

import subprocess

import pytest

from vla_robot_sorting.planning.manipulation_recovery import (
    RecoveryContractError,
    create_source_lock,
    verify_source_lock,
)


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    (root / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    (root / "tên tệp.txt").write_text("one\ntwo\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    return root


def _force_linux_porcelain(monkeypatch):
    """Emulate Linux's observed tracked-M status for otherwise identical CRLF."""
    original = subprocess.check_output

    def intercepted(cmd, *args, **kwargs):
        if list(cmd)[-3:] == ["status", "--porcelain=v1", "-z"]:
            return " M tên tệp.txt\0".encode()
        return original(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", intercepted)


def test_eol_only_dirty_porcelain_is_verified_against_committed_head(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    lock = create_source_lock(root)
    (root / "tên tệp.txt").write_bytes(b"one\r\ntwo\r\n")
    _force_linux_porcelain(monkeypatch)
    assert verify_source_lock(root, lock)["verified_file_count"] == 2
    assert create_source_lock(root)["files"] == lock["files"]


def test_non_eol_changed_source_still_rejected_with_same_porcelain(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    lock = create_source_lock(root)
    (root / "tên tệp.txt").write_bytes(b"one\r\nCHANGED\r\n")
    _force_linux_porcelain(monkeypatch)
    with pytest.raises(RecoveryContractError, match="not clean"):
        verify_source_lock(root, lock)
    with pytest.raises(RecoveryContractError, match="not clean"):
        create_source_lock(root)


def test_untracked_and_staged_sources_remain_forbidden(tmp_path):
    root = _repo(tmp_path)
    lock = create_source_lock(root)
    (root / "untracked.py").write_text("surprise=1\n", encoding="utf-8")
    with pytest.raises(RecoveryContractError, match="not clean"):
        verify_source_lock(root, lock)
    (root / "untracked.py").unlink()
    (root / "tên tệp.txt").write_text("one\nthree\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", "tên tệp.txt"],
        check=True,
        capture_output=True,
    )
    with pytest.raises(RecoveryContractError, match="not clean"):
        verify_source_lock(root, lock)
    with pytest.raises(RecoveryContractError, match="not clean"):
        create_source_lock(root)
