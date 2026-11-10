#!/usr/bin/env python3
"""Materialize Git LFS pointer files from a commit-pinned GitHub media endpoint.

This is used because the pinned Panthera-HT_RoboTwin tree contains LFS pointer
blobs without a .gitattributes file that would make normal git-lfs checkout
materialize them automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

POINTER_RE = re.compile(
    rb"\Aversion https://git-lfs\.github\.com/spec/v1\r?\n"
    rb"oid sha256:([0-9a-f]{64})\r?\n"
    rb"size ([0-9]+)\r?\n?\Z"
)


def parse_lfs_pointer(path: Path) -> tuple[str, int] | None:
    """Return (sha256, size) when path contains a canonical Git LFS pointer."""

    if not path.is_file():
        return None
    if path.stat().st_size > 1024:
        return None
    data = path.read_bytes()
    match = POINTER_RE.match(data)
    if match is None:
        return None
    return match.group(1).decode("ascii"), int(match.group(2))


def _download_verified(
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int,
) -> None:
    temp = destination.with_name(destination.name + ".lfs-download")
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DAMH-S1.1-native-gate/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)

        actual_sha256 = digest.hexdigest()
        if size != expected_size:
            raise RuntimeError(
                f"size mismatch for {destination}: expected {expected_size}, got {size}"
            )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"sha256 mismatch for {destination}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def materialize_subtree(
    *,
    repo_root: Path,
    subtree: Path,
    github_repo: str,
    commit: str,
) -> list[dict[str, object]]:
    """Replace all canonical LFS pointers below subtree with verified payloads."""

    repo_root = repo_root.resolve()
    subtree = subtree.resolve()
    try:
        subtree.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("subtree must be inside repo_root") from exc

    entries: list[dict[str, object]] = []
    candidates = sorted(path for path in subtree.rglob("*") if path.is_file())

    for path in candidates:
        pointer = parse_lfs_pointer(path)
        if pointer is None:
            continue

        expected_sha256, expected_size = pointer
        relative = path.relative_to(repo_root).as_posix()
        quoted_relative = urllib.parse.quote(relative, safe="/")
        url = (
            f"https://media.githubusercontent.com/media/"
            f"{github_repo}/{commit}/{quoted_relative}"
        )
        _download_verified(url, path, expected_sha256, expected_size)
        if parse_lfs_pointer(path) is not None:
            raise RuntimeError(f"materialization did not replace pointer: {relative}")

        entries.append(
            {
                "path": relative,
                "sha256": expected_sha256,
                "size": expected_size,
                "source": url,
            }
        )


    remaining = [
        path.relative_to(repo_root).as_posix()
        for path in subtree.rglob("*")
        if path.is_file() and parse_lfs_pointer(path) is not None
    ]
    if remaining:
        raise RuntimeError(f"unmaterialized LFS pointers remain: {remaining}")

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--subtree", type=Path, required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subtree = args.subtree
    if not subtree.is_absolute():
        subtree = args.repo_root / subtree

    entries = materialize_subtree(
        repo_root=args.repo_root,
        subtree=subtree,
        github_repo=args.github_repo,
        commit=args.commit,
    )
    payload = {
        "schema": "git-lfs-materialization-v1",
        "github_repo": args.github_repo,
        "commit": args.commit,
        "subtree": subtree.resolve().relative_to(args.repo_root.resolve()).as_posix(),
        "materialized_count": len(entries),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
