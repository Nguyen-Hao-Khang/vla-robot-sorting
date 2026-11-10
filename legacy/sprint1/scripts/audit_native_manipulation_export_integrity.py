"""Verify immutable Linux manipulation export and its nested evidence checksums.

Integrity-only, never grants native manipulation or Sprint1 acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

SHA_ROW = re.compile(r"^([0-9a-f]{64})  \*?(.+)$")
BUNDLE_HEAD = re.compile(r"^([0-9a-f]{40})\s+")


class IntegrityError(ValueError):
    pass


def _hash_path(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _expected_rows(text: str) -> dict[str, str]:
    checks: dict[str, str] = {}
    for line in text.splitlines():
        match = SHA_ROW.fullmatch(line)
        if match is None:
            raise IntegrityError("malformed SHA256SUMS entry")
        expected, name = match.groups()
        name = name.removeprefix("./")
        parts = PurePosixPath(name).parts
        if (
            not name
            or PurePosixPath(name).is_absolute()
            or ".." in parts
            or "\\" in name
            or name in checks
        ):
            raise IntegrityError("unsafe or duplicate checksum path")
        checks[name] = expected
    if not checks:
        raise IntegrityError("empty checksum manifest")
    return checks


def verify_top_level(folder: Path) -> int:
    manifest = folder / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise IntegrityError("top-level SHA256SUMS.txt missing")
    checks = _expected_rows(manifest.read_text(encoding="ascii"))
    for name, expected in checks.items():
        source = folder / name
        if not source.is_file() or _hash_path(source) != expected:
            raise IntegrityError(f"top-level mismatch or missing: {name}")
    return len(checks)


def verify_nested_evidence(archive: Path) -> tuple[int, str, str]:
    if not archive.is_file():
        raise IntegrityError("evidence.tar.gz missing")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        name_to_member = {member.name.removeprefix("./"): member for member in members}
        manifests = [
            name
            for name in name_to_member
            if name.endswith("/SHA256SUMS.txt")
        ]
        if len(manifests) != 1:
            raise IntegrityError("expected exactly one dedicated checksum manifest")
        prefix = manifests[0].removesuffix("/SHA256SUMS.txt")
        member = name_to_member[manifests[0]]
        if not member.isfile():
            raise IntegrityError("dedicated checksum manifest is not a regular file")
        file = source.extractfile(member)
        if file is None:
            raise IntegrityError("dedicated checksum manifest unreadable")
        checks = _expected_rows(file.read().decode("ascii"))
        for relative, expected in checks.items():
            inner = name_to_member.get(f"{prefix}/{relative}")
            if inner is None or not inner.isfile():
                raise IntegrityError(f"dedicated evidence missing/non-file: {relative}")
            stream = source.extractfile(inner)
            if stream is None:
                raise IntegrityError(f"dedicated evidence unreadable: {relative}")
            sha = hashlib.sha256()
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
            if sha.hexdigest() != expected:
                raise IntegrityError(f"dedicated evidence SHA mismatch: {relative}")
        head_member = name_to_member.get(f"{prefix}/git-head.txt")
        if head_member is None or not head_member.isfile():
            raise IntegrityError("dedicated git-head.txt missing")
        stream = source.extractfile(head_member)
        if stream is None:
            raise IntegrityError("dedicated git-head.txt unreadable")
        linux_head = stream.read().decode("ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", linux_head):
            raise IntegrityError("dedicated git-head.txt invalid")
        return len(checks), prefix, linux_head


def verify_git_bundle(
    archive: Path, repo: Path, linux_head: str
) -> tuple[str, str]:
    if not (repo / ".git").exists():
        raise IntegrityError("canonical local Git repository missing")
    verified = subprocess.run(
        ["git", "-C", str(repo), "bundle", "verify", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verified.returncode != 0 or "complete history" not in (
        verified.stdout + verified.stderr
    ):
        raise IntegrityError("Git bundle failed complete-history verification")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    refs = subprocess.run(
        ["git", "-C", str(repo), "bundle", "list-heads", str(archive)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    bundle_heads = [match.group(1) for line in refs.splitlines()
                    if (match := BUNDLE_HEAD.match(line))]
    if linux_head not in bundle_heads:
        raise IntegrityError("Git bundle does not contain exported Linux HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", linux_head, head],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise IntegrityError("exported Linux HEAD is not ancestor of Windows HEAD")
    return head, "complete-history"


def audit(folder: Path, repo: Path) -> dict[str, Any]:
    folder = folder.resolve()
    top = verify_top_level(folder)
    dedicated, prefix, linux_head = verify_nested_evidence(
        folder / "evidence.tar.gz"
    )
    head, bundle = verify_git_bundle(
        folder / "linux-project.bundle", repo.resolve(), linux_head
    )
    return {
        "schema": "linux-native-export-integrity-v1",
        "integrity": "PASS",
        "native_acceptance": "NOT_EVALUATED",
        "export_folder": str(folder),
        "top_level_sha256_pass_count": top,
        "dedicated_sha256_pass_count": dedicated,
        "dedicated_campaign": prefix,
        "git_bundle": bundle,
        "linux_exported_head": linux_head,
        "windows_current_head": head,
        "claim_boundary": (
            "SHA256 and complete Git history only; re-run independent native "
            "manipulation auditor and examine effective seed resolution before promotion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_folder", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.export_folder, args.repo)
    except (OSError, IntegrityError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"INTEGRITY=FAIL: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
