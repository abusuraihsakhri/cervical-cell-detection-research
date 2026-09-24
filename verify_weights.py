#!/usr/bin/env python3
"""
Model Weights Verification & Loss Prevention Utility.

Safeguards PyTorch (.pt) checkpoints against accidental deletion, omission from git,
or cryptographic corruption. Compares all model weights against the authoritative
manifest and verifies that gitignore rules do not suppress model files.

Usage:
    python verify_weights.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "publication/weights/WEIGHTS_MANIFEST.json"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def check_git_ignore(path: Path) -> bool:
    """Returns True if git ignores this file, False if git tracks it."""
    try:
        res = subprocess.run(
            ["git", "check-ignore", str(path.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 78)
    print("      RESEARCH MODEL WEIGHTS INTEGRITY & SAFEGUARD AUDIT")
    print("=" * 78)

    if not MANIFEST_PATH.exists():
        print(f"[-] ERROR: Manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    checkpoints = manifest.get("checkpoints", [])
    print(f"Auditing {len(checkpoints)} registered model checkpoints...\n")

    failures = []
    ignored_warnings = []

    print(f"{'Status':<8} {'Model Checkpoint':<32} {'Size (MB)':<10} {'Git Status':<12} {'SHA-256 (first 12)'}")
    print("-" * 78)

    for item in checkpoints:
        rel_path = item["path"]
        expected_hash = item["sha256"].lower()
        full_path = ROOT / rel_path

        if not full_path.is_file():
            print(f"MISSING  {rel_path:<32} {'--':<10} {'UNKNOWN':<12} [FILE NOT FOUND]")
            failures.append((rel_path, "File missing from disk"))
            continue

        size_mb = full_path.stat().st_size / (1024 * 1024)
        actual_hash = sha256_file(full_path)
        is_ignored = check_git_ignore(full_path)
        git_label = "IGNORED!" if is_ignored else "TRACKED"

        if is_ignored:
            ignored_warnings.append(rel_path)

        if actual_hash == expected_hash:
            print(f"[PASS]   {rel_path:<32} {size_mb:>6.2f} MB  {git_label:<12} {actual_hash[:12]}")
        else:
            print(f"[CORRUPT] {rel_path:<32} {size_mb:>6.2f} MB  {git_label:<12} MISMATCH!")
            print(f"         Expected: {expected_hash}")
            print(f"         Actual:   {actual_hash}")
            failures.append((rel_path, f"Hash mismatch: expected {expected_hash}, got {actual_hash}"))

    print("-" * 78)

    if ignored_warnings:
        print("\n[!] WARNING: The following checkpoints are IGNORED by .gitignore!")
        print("    If pushed to GitHub now, these weights WILL BE LOST:")
        for w in ignored_warnings:
            print(f"      - {w}")
        print("    Remediation: Add `!{path}` whitelist exceptions in .gitignore.")

    if failures:
        print(f"\n[FAIL] {len(failures)} checkpoint integrity failures detected:")
        for path, reason in failures:
            print(f"  * {path}: {reason}")
        sys.exit(1)

    if not ignored_warnings:
        print("\n[SUCCESS] All model checkpoints exist, match verified SHA-256 hashes,")
        print("          and are properly configured for Git tracking without loss.")
    else:
        print(f"\n[ATTENTION] Checksums valid, but {len(ignored_warnings)} files are gitignored.")
        sys.exit(2)


if __name__ == "__main__":
    main()
