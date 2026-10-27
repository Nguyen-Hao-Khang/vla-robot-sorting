from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "linux"
    / "materialize_git_lfs_pointers.py"
)
SPEC = importlib.util.spec_from_file_location("lfs_materializer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pointer(payload: bytes) -> bytes:
    digest = hashlib.sha256(payload).hexdigest()
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{digest}\n".encode()
        + f"size {len(payload)}\n".encode()
    )


def test_parse_lfs_pointer(tmp_path: Path) -> None:
    payload = b"<robot/>\n"
    pointer = tmp_path / "robot.urdf"
    pointer.write_bytes(_pointer(payload))

    parsed = MODULE.parse_lfs_pointer(pointer)

    assert parsed == (hashlib.sha256(payload).hexdigest(), len(payload))


def test_non_pointer_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "robot.urdf"
    path.write_text("<robot/>\n", encoding="utf-8")

    assert MODULE.parse_lfs_pointer(path) is None


def test_download_verified_accepts_matching_file_url(tmp_path: Path) -> None:
    payload = b"verified binary payload"
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(payload)

    MODULE._download_verified(
        source.as_uri(),
        destination,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    assert destination.read_bytes() == payload
