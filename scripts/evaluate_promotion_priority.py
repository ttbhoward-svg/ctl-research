#!/usr/bin/env python3
"""Evaluate promotion priority for ES/PL (or selected symbols).

Produces a concise ranking based on current acceptance blockers and the
latest MTFA audit rates from portfolio run summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.canonical_acceptance import acceptance_from_diagnostics  # noqa: E402
from ctl.cutover_diagnostics import run_diagnostics  # noqa: E402
from ctl.operating_profile import discover_ts_custom_file, load_operating_profile  # noqa: E402
from ctl.parity_prep import load_and_validate  # noqa: E402
from ctl.promotion_priority import (  # noqa: E402
    build_priority_row,
    extract_mtfa_rates,
    load_latest_run_summary,
    rank_priority,
)
from ctl.roll_reconciliation import load_roll_manifest  # noqa: E402

DEFAULT_PROFILE = REPO_ROOT / "configs" / "cutover" / "operating_profile_v1.yaml"
DEFAULT_DB_DIR = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "data" / "processed" / "cutover_v1" / "run_summaries"


def _evaluate_symbol(symbol: str, profile, db_dir: Path, ts_dir: Path, mtfa_rates: dict):
    cfg = profile.symbol_settings[symbol]

    can_df, can_err = load_and_validate(db_dir / f"{symbol}_continuous.csv", f"DB {symbol}")
    if can_err:
        raise RuntimeError(f"{symbol} canonical load error: {'; '.join(can_err)}")

    manifest = load_roll_manifest(db_dir / f"{symbol}_roll_manifest.json")
    ts_adj_path = discover_ts_custom_file(symbol, ts_dir, "ADJ")
    ts_unadj_path = discover_ts_custom_file(symbol, ts_dir, "UNADJ")
    ts_adj_df, a_err = load_and_validate(ts_adj_path, f"TS {symbol} ADJ")
    ts_unadj_df, u_err = load_and_validate(ts_unadj_path, f"TS {symbol} UNADJ")
    if a_err or u_err:
        raise RuntimeError(f"{symbol} TS load error: {'; '.join(a_err + u_err)}")

    diag = run_diagnostics(
        canonical_adj_df=can_df,
        ts_adj_df=ts_adj_df,
        manifest_entries=manifest,
        ts_unadj_df=ts_unadj_df,
        symbol=symbol,
        tick_size=cfg.tick_size,
        max_day_delta=cfg.max_day_delta,
    )
    acceptance = acceptance_from_diagnostics(diag)
    return build_priority_row(
        symbol=symbol,
        acceptance=acceptance,
        mtfa=mtfa_rates.get(symbol, {}),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate promotion priority for selected symbols")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    p.add_argument("--ts-dir", type=Path, default=DEFAULT_TS_DIR)
    p.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    p.add_argument("--symbols", type=str, default="ES,PL", help="Comma-separated symbols")
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    profile = load_operating_profile(args.profile)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    latest_summary = load_latest_run_summary(args.summary_dir)
    mtfa_rates = extract_mtfa_rates(latest_summary)

    rows = []
    for sym in symbols:
        if sym not in profile.symbol_settings:
            raise SystemExit(f"Symbol not found in profile.symbol_settings: {sym}")
        rows.append(_evaluate_symbol(sym, profile, args.db_dir, args.ts_dir, mtfa_rates))

    ranked = rank_priority(rows)

    if args.json_output:
        print(json.dumps({
            "symbols": symbols,
            "ranked": [r.to_dict() for r in ranked],
        }, indent=2))
        return

    print("Promotion Priority Ranking")
    print("Symbol  Score    Band    Decision  MeanDrift  MeanGap  MTFA_W  MTFA_M")
    print("------  -----    ----    --------  ---------  -------  ------  ------")
    for r in ranked:
        w = "-" if r.mtfa_weekly_rate is None else f"{r.mtfa_weekly_rate:.4f}"
        m = "-" if r.mtfa_monthly_rate is None else f"{r.mtfa_monthly_rate:.4f}"
        print(
            f"{r.symbol:<6}  {r.priority_score:<7.4f}  {r.priority_band:<6}"
            f"  {r.decision:<8}  {r.mean_drift:<9.4f}  {r.mean_gap_diff:<7.4f}  {w:<6}  {m:<6}"
        )
        if r.reasons:
            for reason in r.reasons:
                print(f"  - {reason}")


if __name__ == "__main__":
    main()
