#!/usr/bin/env python3
"""Analyze PL interval-level basis behavior from current diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.cutover_diagnostics import run_diagnostics  # noqa: E402
from ctl.operating_profile import discover_ts_custom_file, load_operating_profile  # noqa: E402
from ctl.parity_prep import load_and_validate  # noqa: E402
from ctl.pl_basis_analysis import build_interval_basis_report, save_interval_basis_report  # noqa: E402
from ctl.roll_reconciliation import load_roll_manifest  # noqa: E402

DEFAULT_PROFILE = REPO_ROOT / "configs" / "cutover" / "operating_profile_v1.yaml"
DEFAULT_DB_DIR = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "cutover_v1" / "diagnostics_h6" / "PL_interval_basis_report.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="PL interval-level basis analyzer")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    p.add_argument("--ts-dir", type=Path, default=DEFAULT_TS_DIR)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--roll-window-days", type=int, default=3)
    p.add_argument("--save-csv", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    profile = load_operating_profile(args.profile)
    cfg = profile.symbol_settings["PL"]

    can_df, can_err = load_and_validate(args.db_dir / "PL_continuous.csv", "DB PL")
    if can_err:
        raise SystemExit(f"PL canonical load error: {'; '.join(can_err)}")

    manifest = load_roll_manifest(args.db_dir / "PL_roll_manifest.json")
    ts_adj_df, a_err = load_and_validate(discover_ts_custom_file("PL", args.ts_dir, "ADJ"), "TS PL ADJ")
    ts_unadj_df, u_err = load_and_validate(discover_ts_custom_file("PL", args.ts_dir, "UNADJ"), "TS PL UNADJ")
    if a_err or u_err:
        raise SystemExit(f"PL TS load error: {'; '.join(a_err + u_err)}")

    diag = run_diagnostics(
        canonical_adj_df=can_df,
        ts_adj_df=ts_adj_df,
        manifest_entries=manifest,
        ts_unadj_df=ts_unadj_df,
        symbol="PL",
        tick_size=cfg.tick_size,
        max_day_delta=cfg.max_day_delta,
    )

    explanation = diag.l4.explanation.to_dict() if diag.l4.explanation else {}
    report = build_interval_basis_report(
        drift_df=diag.l4.drift_df,
        l2_detail_df=diag.l2.detail_df,
        explanation=explanation,
        top_n=args.top_n,
        roll_window_days=args.roll_window_days,
    )

    if report.empty:
        print("No interval report generated.")
        return

    out_path = save_interval_basis_report(report, args.save_csv)

    print("PL Interval Basis Report (top rows)")
    print(report.head(args.top_n).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
