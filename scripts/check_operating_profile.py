#!/usr/bin/env python3
"""Operating profile gate check (H.7).

Re-derives canonical acceptance for each gating symbol and compares
against the locked expected statuses in the operating profile YAML.
Acts as a regression guard — if acceptance semantics, data, or
diagnostics change, mismatches surface immediately.

Usage
-----
    python scripts/check_operating_profile.py
    python scripts/check_operating_profile.py --json
    python scripts/check_operating_profile.py --profile configs/cutover/operating_profile_v1.yaml

Exit codes: 0 = all match, 2 = mismatch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.canonical_acceptance import acceptance_from_diagnostics  # noqa: E402
from ctl.cutover_diagnostics import run_diagnostics  # noqa: E402
from ctl.operating_profile import (  # noqa: E402
    PortfolioCheckResult,
    check_portfolio,
    check_symbol_status,
    discover_ts_custom_file,
    load_operating_profile,
)
from ctl.parity_prep import load_and_validate  # noqa: E402
from ctl.roll_reconciliation import load_roll_manifest  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("check_operating_profile")

DEFAULT_PROFILE = REPO_ROOT / "configs" / "cutover" / "operating_profile_v1.yaml"
DEFAULT_DB_CONTINUOUS_DIR = (
    REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
)
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"


def _run_gate(profile_path: Path) -> PortfolioCheckResult:
    """Execute the gate check and return a PortfolioCheckResult."""
    profile = load_operating_profile(profile_path)
    symbol_results = []

    for sym in profile.gating_universe:
        settings = profile.symbol_settings.get(sym)
        if settings is None:
            from ctl.operating_profile import SymbolCheckResult

            symbol_results.append(
                SymbolCheckResult(
                    symbol=sym,
                    expected="(missing)",
                    actual="ERROR",
                    passed=False,
                    detail=f"No symbol_settings entry for {sym}",
                )
            )
            continue

        # --- Load canonical continuous ---
        canonical_path = DEFAULT_DB_CONTINUOUS_DIR / f"{sym}_continuous.csv"
        canonical_df, can_errors = load_and_validate(canonical_path, f"DB {sym}")
        if can_errors:
            from ctl.operating_profile import SymbolCheckResult

            symbol_results.append(
                SymbolCheckResult(
                    symbol=sym,
                    expected=settings.expected_status,
                    actual="ERROR",
                    passed=False,
                    detail=f"Load error: {'; '.join(can_errors)}",
                )
            )
            continue

        # --- Load manifest ---
        manifest_path = DEFAULT_DB_CONTINUOUS_DIR / f"{sym}_roll_manifest.json"
        manifest_entries = load_roll_manifest(manifest_path)

        # --- Load TS adj/unadj ---
        ts_adj_path = discover_ts_custom_file(sym, DEFAULT_TS_DIR, "ADJ")
        ts_unadj_path = discover_ts_custom_file(sym, DEFAULT_TS_DIR, "UNADJ")

        ts_adj_df = None
        ts_unadj_df = None

        if ts_adj_path:
            ts_adj_df, adj_errors = load_and_validate(ts_adj_path, f"TS {sym} ADJ")
            if adj_errors:
                logger.warning("TS ADJ load issues for %s: %s", sym, adj_errors)
                ts_adj_df = None

        if ts_unadj_path:
            ts_unadj_df, unadj_errors = load_and_validate(
                ts_unadj_path, f"TS {sym} UNADJ"
            )
            if unadj_errors:
                logger.warning("TS UNADJ load issues for %s: %s", sym, unadj_errors)
                ts_unadj_df = None

        if ts_adj_df is None:
            from ctl.operating_profile import SymbolCheckResult

            symbol_results.append(
                SymbolCheckResult(
                    symbol=sym,
                    expected=settings.expected_status,
                    actual="ERROR",
                    passed=False,
                    detail=f"TS ADJ file not found for {sym}",
                )
            )
            continue

        # --- Run diagnostics ---
        diag = run_diagnostics(
            canonical_adj_df=canonical_df,
            ts_adj_df=ts_adj_df,
            manifest_entries=manifest_entries,
            ts_unadj_df=ts_unadj_df,
            symbol=sym,
            tick_size=settings.tick_size,
            max_day_delta=settings.max_day_delta,
        )

        # --- Acceptance ---
        acceptance = acceptance_from_diagnostics(diag)
        actual_decision = acceptance.decision

        # --- Compare ---
        result = check_symbol_status(sym, settings.expected_status, actual_decision)
        symbol_results.append(result)

    return check_portfolio(profile, symbol_results)


def _format_text(result: PortfolioCheckResult) -> str:
    """Format portfolio check result as a human-readable table."""
    lines = [
        f"Operating Profile Gate: {'PASS' if result.passed else 'MISMATCH'}",
        f"Portfolio Recommendation: {result.recommendation}",
        "",
        f"{'Symbol':<8} {'Expected':<10} {'Actual':<10} {'Status':<8} Detail",
        f"{'------':<8} {'--------':<10} {'------':<10} {'------':<8} ------",
    ]
    for r in result.symbol_results:
        status = "OK" if r.passed else "FAIL"
        lines.append(
            f"{r.symbol:<8} {r.expected:<10} {r.actual:<10} {status:<8} {r.detail}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check operating profile gate (H.7).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Path to operating profile YAML.",
    )
    args = parser.parse_args()

    result = _run_gate(args.profile)

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_format_text(result))

    sys.exit(0 if result.passed else 2)


if __name__ == "__main__":
    main()
