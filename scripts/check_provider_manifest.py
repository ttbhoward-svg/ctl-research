#!/usr/bin/env python3
"""Validate and report provider-manifest readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.provider_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    evaluate_manifest_availability,
    load_provider_manifest,
    validate_provider_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Check per-symbol provider manifest readiness.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    manifest = load_provider_manifest(args.manifest)
    errors = validate_provider_manifest(manifest)
    rows = evaluate_manifest_availability(manifest)

    n_primary_ok = sum(1 for r in rows if r.primary_available)
    payload = {
        "manifest": str(args.manifest),
        "cycle_id": manifest.cycle_id,
        "version": manifest.version,
        "validation_errors": errors,
        "symbols_total": len(rows),
        "primary_available": n_primary_ok,
        "primary_missing": len(rows) - n_primary_ok,
        "rows": [
            {
                "symbol": r.symbol,
                "primary": r.primary,
                "primary_available": r.primary_available,
                "fallback": r.fallback,
                "fallback_available": r.fallback_available,
                "reference": r.reference,
                "reference_available": r.reference_available,
            }
            for r in rows
        ],
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
    else:
        print("Provider Manifest Readiness")
        print(f"Manifest: {args.manifest}")
        print(f"Cycle: {manifest.cycle_id}  Version: {manifest.version}")
        print(f"Primary availability: {n_primary_ok}/{len(rows)}")
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f" - {e}")
        print()
        print("Symbol  Primary      P_OK  Fallback      F_OK  Reference    R_OK")
        print("------  -------      ----  --------      ----  ---------    ----")
        for r in rows:
            print(
                f"{r.symbol:6s}  {r.primary:11s}  "
                f"{'Y' if r.primary_available else 'N':4s}  "
                f"{r.fallback:12s}  {'Y' if r.fallback_available else 'N':4s}  "
                f"{r.reference:11s}  {'Y' if r.reference_available else 'N':4s}"
            )

    # Any validation error or missing primary fails the check.
    if errors or n_primary_ok < len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
