#!/usr/bin/env python3
"""Evaluate strict ES walk-forward drift harmonization candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.canonical_acceptance import acceptance_from_diagnostics  # noqa: E402
from ctl.cutover_diagnostics import run_diagnostics  # noqa: E402
from ctl.es_drift_walkforward import (  # noqa: E402
    apply_walkforward_offset,
    derive_walkforward_offset,
    window_abs_drift_mean,
)
from ctl.operating_profile import discover_ts_custom_file, load_operating_profile  # noqa: E402
from ctl.parity_prep import load_and_validate  # noqa: E402
from ctl.roll_reconciliation import load_roll_manifest  # noqa: E402

DEFAULT_PROFILE = REPO_ROOT / "configs" / "cutover" / "operating_profile_v1.yaml"
DEFAULT_DB_DIR = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"


def _diag(can_df, ts_adj_df, ts_unadj_df, manifest_entries, tick_size, mdd):
    return run_diagnostics(
        canonical_adj_df=can_df,
        ts_adj_df=ts_adj_df,
        manifest_entries=manifest_entries,
        ts_unadj_df=ts_unadj_df,
        symbol="ES",
        tick_size=tick_size,
        max_day_delta=mdd,
    )


def _snap(diag):
    acc = acceptance_from_diagnostics(diag)
    return {
        "strict": diag.strict_status,
        "policy": diag.policy_status,
        "decision": acc.decision,
        "mean_gap_diff": diag.l3.mean_gap_diff,
        "mean_drift": diag.l4.mean_drift,
        "reasons": acc.reasons,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate ES walk-forward drift harmonization (offline)")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    profile = load_operating_profile(args.profile)
    cfg = profile.symbol_settings["ES"]

    can_df, e0 = load_and_validate(DEFAULT_DB_DIR / "ES_continuous.csv", "DB ES")
    ts_adj, e1 = load_and_validate(discover_ts_custom_file("ES", DEFAULT_TS_DIR, "ADJ"), "TS ES ADJ")
    ts_unadj, e2 = load_and_validate(discover_ts_custom_file("ES", DEFAULT_TS_DIR, "UNADJ"), "TS ES UNADJ")
    if e0 or e1 or e2:
        raise SystemExit("; ".join(e0 + e1 + e2))

    manifest = load_roll_manifest(DEFAULT_DB_DIR / "ES_roll_manifest.json")
    baseline = _diag(can_df, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)

    # Strict forward splits (train earlier, apply later).
    splits = [
        ("wf_1", "2018-01-01", "2019-12-31", "2020-01-01", "2022-12-31"),
        ("wf_2", "2018-01-01", "2022-12-31", "2023-01-01", "2026-02-17"),
    ]

    rows = []
    for label, tr_s, tr_e, ap_s, ap_e in splits:
        off = derive_walkforward_offset(can_df, ts_adj, tr_s, tr_e, ap_s, ap_e)
        can_h = apply_walkforward_offset(can_df, off)
        d = _diag(can_h, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)
        pre_m, pre_n = window_abs_drift_mean(can_df, ts_adj, ap_s, ap_e)
        post_m, post_n = window_abs_drift_mean(can_h, ts_adj, ap_s, ap_e)
        rows.append(
            {
                "label": label,
                "offset": off.to_dict(),
                "window_mean_drift_before": pre_m,
                "window_mean_drift_after": post_m,
                "window_n": min(pre_n, post_n),
                "window_delta": post_m - pre_m,
                "global": _snap(d),
                "global_delta_vs_baseline": {
                    "mean_gap_diff": d.l3.mean_gap_diff - baseline.l3.mean_gap_diff,
                    "mean_drift": d.l4.mean_drift - baseline.l4.mean_drift,
                },
            }
        )

    payload = {
        "symbol": "ES",
        "baseline": _snap(baseline),
        "walkforward_candidates": rows,
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("ES Walk-Forward Drift Harmonization (Offline)")
    print(
        f"baseline: decision={payload['baseline']['decision']} "
        f"gap={payload['baseline']['mean_gap_diff']:.4f} "
        f"drift={payload['baseline']['mean_drift']:.4f}"
    )
    for r in rows:
        g = r["global"]
        d = r["global_delta_vs_baseline"]
        print(f"\n{r['label']}:")
        print(
            f"  train={r['offset']['train_start']}..{r['offset']['train_end']} "
            f"apply={r['offset']['apply_start']}..{r['offset']['apply_end']}"
        )
        print(
            f"  offset={r['offset']['median_signed_diff']:.4f} (n_train={r['offset']['n_train_rows']}) "
            f"window_drift: {r['window_mean_drift_before']:.4f} -> {r['window_mean_drift_after']:.4f} "
            f"(delta={r['window_delta']:.4f})"
        )
        print(
            f"  global: decision={g['decision']} gap={g['mean_gap_diff']:.4f} "
            f"drift={g['mean_drift']:.4f} delta_drift={d['mean_drift']:.4f}"
        )


if __name__ == "__main__":
    main()
