#!/usr/bin/env python3
"""Re-rank promotion priority with ES walk-forward candidate applied offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.canonical_acceptance import acceptance_from_diagnostics  # noqa: E402
from ctl.cutover_diagnostics import run_diagnostics  # noqa: E402
from ctl.es_drift_walkforward import apply_walkforward_offset, derive_walkforward_offset  # noqa: E402
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


def _diag(symbol, can_df, ts_adj_df, ts_unadj_df, manifest_entries, tick_size, mdd):
    return run_diagnostics(
        canonical_adj_df=can_df,
        ts_adj_df=ts_adj_df,
        manifest_entries=manifest_entries,
        ts_unadj_df=ts_unadj_df,
        symbol=symbol,
        tick_size=tick_size,
        max_day_delta=mdd,
    )


def _load_symbol_inputs(symbol: str, profile, db_dir: Path, ts_dir: Path):
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
    return cfg, can_df, ts_adj_df, ts_unadj_df, manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Priority re-rank with ES walk-forward candidate")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    p.add_argument("--ts-dir", type=Path, default=DEFAULT_TS_DIR)
    p.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    profile = load_operating_profile(args.profile)
    latest_summary = load_latest_run_summary(args.summary_dir)
    mtfa_rates = extract_mtfa_rates(latest_summary)

    # Baseline ES + PL rows.
    base_rows = []
    symbol_diags = {}
    for sym in ["ES", "PL"]:
        cfg, can, ts_adj, ts_un, manifest = _load_symbol_inputs(sym, profile, args.db_dir, args.ts_dir)
        d = _diag(sym, can, ts_adj, ts_un, manifest, cfg.tick_size, cfg.max_day_delta)
        symbol_diags[sym] = (cfg, can, ts_adj, ts_un, manifest, d)
        base_rows.append(
            build_priority_row(
                symbol=sym,
                acceptance=acceptance_from_diagnostics(d),
                mtfa=mtfa_rates.get(sym, {}),
            )
        )
    baseline_ranked = rank_priority(base_rows)

    # ES walk-forward candidates.
    cfg, can, ts_adj, ts_un, manifest, es_base_diag = symbol_diags["ES"]
    splits = [
        ("wf_1", "2018-01-01", "2019-12-31", "2020-01-01", "2022-12-31"),
        ("wf_2", "2018-01-01", "2022-12-31", "2023-01-01", "2026-02-17"),
    ]
    candidate_rows = []
    for label, tr_s, tr_e, ap_s, ap_e in splits:
        off = derive_walkforward_offset(can, ts_adj, tr_s, tr_e, ap_s, ap_e)
        can_h = apply_walkforward_offset(can, off)
        d = _diag("ES", can_h, ts_adj, ts_un, manifest, cfg.tick_size, cfg.max_day_delta)
        candidate_rows.append((label, off, d))

    # Choose best candidate by lowest global mean_drift.
    best_label, best_off, best_diag = sorted(candidate_rows, key=lambda x: x[2].l4.mean_drift)[0]

    alt_rows = []
    alt_rows.append(
        build_priority_row(
            symbol="ES",
            acceptance=acceptance_from_diagnostics(best_diag),
            mtfa=mtfa_rates.get("ES", {}),
        )
    )
    # PL unchanged baseline for comparison.
    _, _, _, _, _, pl_diag = symbol_diags["PL"]
    alt_rows.append(
        build_priority_row(
            symbol="PL",
            acceptance=acceptance_from_diagnostics(pl_diag),
            mtfa=mtfa_rates.get("PL", {}),
        )
    )
    candidate_ranked = rank_priority(alt_rows)

    payload = {
        "baseline_ranked": [r.to_dict() for r in baseline_ranked],
        "es_candidate": {
            "label": best_label,
            "offset": best_off.to_dict(),
            "baseline_mean_drift": es_base_diag.l4.mean_drift,
            "candidate_mean_drift": best_diag.l4.mean_drift,
            "delta_mean_drift": best_diag.l4.mean_drift - es_base_diag.l4.mean_drift,
        },
        "candidate_ranked": [r.to_dict() for r in candidate_ranked],
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("Baseline Priority Ranking")
    for r in baseline_ranked:
        print(f"- {r.symbol}: score={r.priority_score:.4f} decision={r.decision} drift={r.mean_drift:.4f} gap={r.mean_gap_diff:.4f}")

    print("\nBest ES Walk-Forward Candidate")
    print(
        f"- {best_label}: offset={best_off.median_signed_diff:.4f} "
        f"drift {es_base_diag.l4.mean_drift:.4f} -> {best_diag.l4.mean_drift:.4f} "
        f"(delta={best_diag.l4.mean_drift - es_base_diag.l4.mean_drift:.4f})"
    )

    print("\nPriority Ranking With ES Candidate")
    for r in candidate_ranked:
        print(f"- {r.symbol}: score={r.priority_score:.4f} decision={r.decision} drift={r.mean_drift:.4f} gap={r.mean_gap_diff:.4f}")


if __name__ == "__main__":
    main()
