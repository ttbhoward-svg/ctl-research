#!/usr/bin/env python3
"""Evaluate ES drift-focused harmonization candidate (offline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.canonical_acceptance import acceptance_from_diagnostics  # noqa: E402
from ctl.cutover_diagnostics import run_diagnostics  # noqa: E402
from ctl.es_drift_harmonization import (  # noqa: E402
    apply_regime_offsets,
    derive_regime_offsets,
    summarize_top_drift_intervals,
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
    p = argparse.ArgumentParser(description="Evaluate ES drift-focused harmonization candidate")
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

    top_intervals = summarize_top_drift_intervals(baseline.l4.explanation, top_n=3)
    regimes = [
        ("pre2020", "2018-01-01", "2019-12-31"),
        ("covid_2020_2022", "2020-01-01", "2022-12-31"),
        ("post2023", "2023-01-01", "2026-12-31"),
    ]
    offsets = derive_regime_offsets(can_df, ts_adj, regimes)
    can_h = apply_regime_offsets(can_df, offsets)
    harmonized = _diag(can_h, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)

    payload = {
        "symbol": "ES",
        "top_drift_intervals": [x.to_dict() for x in top_intervals],
        "regime_offsets": [x.to_dict() for x in offsets],
        "baseline": _snap(baseline),
        "harmonized": _snap(harmonized),
        "delta_harmonized_vs_baseline": {
            "mean_gap_diff": harmonized.l3.mean_gap_diff - baseline.l3.mean_gap_diff,
            "mean_drift": harmonized.l4.mean_drift - baseline.l4.mean_drift,
        },
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("ES Drift Harmonization (Offline)")
    print("Top drift intervals:")
    for t in top_intervals:
        print(
            f"- {t.interval_start} -> {t.interval_end}: "
            f"mean={t.mean_drift:.4f}, contrib={t.drift_contribution_pct:.4f}% ({t.roll_status})"
        )
    print("Regime offsets:")
    for o in offsets:
        print(f"- {o.label} [{o.start}..{o.end}]: median_signed_diff={o.median_signed_diff:.4f}")

    for name in ["baseline", "harmonized"]:
        s = payload[name]
        print(f"\n{name}:")
        print(f"  decision={s['decision']} strict/policy={s['strict']}/{s['policy']} gap={s['mean_gap_diff']:.4f} drift={s['mean_drift']:.4f}")

    d = payload["delta_harmonized_vs_baseline"]
    print("\ndelta harmonized - baseline:")
    print(f"  mean_gap={d['mean_gap_diff']:.4f} mean_drift={d['mean_drift']:.4f}")


if __name__ == "__main__":
    main()
