#!/usr/bin/env python3
"""Verify that all required cutover v1 files exist (H.12).

Checks configs, source modules, scripts, tests, and docs against a
manifest of expected paths.  Exits 0 if all present, 1 if any missing.

Usage
-----
    python scripts/verify_cutover_closeout.py
    python scripts/verify_cutover_closeout.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Manifest of required files relative to repo root.
REQUIRED_FILES: list[str] = [
    # Config
    "configs/cutover/operating_profile_v1.yaml",
    # Source modules (ops-layer)
    "src/ctl/operating_profile.py",
    "src/ctl/run_orchestrator.py",
    "src/ctl/ops_notifier.py",
    # Scripts
    "scripts/check_operating_profile.py",
    "scripts/run_weekly_b1_portfolio.py",
    "scripts/run_weekly_ops.py",
    # Tests
    "tests/unit/test_check_operating_profile.py",
    "tests/unit/test_run_weekly_b1_portfolio.py",
    "tests/unit/test_ops_notifier.py",
    "tests/unit/test_run_weekly_ops.py",
    # Docs
    "docs/governance/cutover_h2_h3_decision_log.md",
    "docs/governance/cutover_v1_closeout.md",
    "docs/ops/weekly_ops_runbook.md",
]


def verify_files(
    repo_root: Path = REPO_ROOT,
    required: list[str] | None = None,
) -> dict:
    """Check that all required cutover files exist.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.
    required : list of str or None
        Override the default manifest (for testing).

    Returns
    -------
    dict
        ``{"passed": bool, "present": [...], "missing": [...]}``.
    """
    manifest = required if required is not None else REQUIRED_FILES
    present: list[str] = []
    missing: list[str] = []

    for rel_path in manifest:
        full = repo_root / rel_path
        if full.is_file():
            present.append(rel_path)
        else:
            missing.append(rel_path)

    return {
        "passed": len(missing) == 0,
        "total": len(manifest),
        "present_count": len(present),
        "missing_count": len(missing),
        "present": present,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify cutover v1 file inventory (H.12).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as JSON.",
    )
    args = parser.parse_args()

    result = verify_files()

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['present_count']}/{result['total']} files present.")
        if result["missing"]:
            print("Missing:")
            for f in result["missing"]:
                print(f"  - {f}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
