#!/usr/bin/env python3
"""Evaluate forward-safe PL feature-model gap correction candidate."""

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
from ctl.pl_gap_feature_model import apply_feature_bias_model, estimate_feature_bias_model  # noqa: E402
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
        symbol="PL",
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
    p = argparse.ArgumentParser(description="Evaluate PL feature-model gap correction (offline)")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--train-end", default="2023-12-31")
    p.add_argument("--apply-start", default="2024-01-01")
    p.add_argument("--min-rows", type=int, default=2)
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    profile = load_operating_profile(args.profile)
    cfg = profile.symbol_settings["PL"]

    can_df, e0 = load_and_validate(DEFAULT_DB_DIR / "PL_continuous.csv", "DB PL")
    ts_adj, e1 = load_and_validate(discover_ts_custom_file("PL", DEFAULT_TS_DIR, "ADJ"), "TS PL ADJ")
    ts_unadj, e2 = load_and_validate(discover_ts_custom_file("PL", DEFAULT_TS_DIR, "UNADJ"), "TS PL UNADJ")
    if e0 or e1 or e2:
        raise SystemExit("; ".join(e0 + e1 + e2))

    manifest = load_roll_manifest(DEFAULT_DB_DIR / "PL_roll_manifest.json")
    base = _diag(can_df, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)

    model = estimate_feature_bias_model(
        base.l2.detail_df,
        train_end_date=args.train_end,
        min_rows=args.min_rows,
    )
    corrected_manifest = apply_feature_bias_model(
        manifest,
        model=model,
        apply_start_date=args.apply_start,
    )
    corrected = _diag(can_df, ts_adj, ts_unadj, corrected_manifest, cfg.tick_size, cfg.max_day_delta)

    payload = {
        "symbol": "PL",
        "train_end": args.train_end,
        "apply_start": args.apply_start,
        "min_rows": args.min_rows,
        "model": model.to_dict(),
        "baseline": _snap(base),
        "feature_gap_only": _snap(corrected),
        "delta_feature_vs_baseline": {
            "mean_gap_diff": corrected.l3.mean_gap_diff - base.l3.mean_gap_diff,
            "mean_drift": corrected.l4.mean_drift - base.l4.mean_drift,
        },
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("PL Feature-Model Gap Correction (Offline)")
    print(f"train_end={args.train_end} apply_start={args.apply_start} min_rows={args.min_rows}")
    print(
        f"model sizes: exact={payload['model']['n_exact']} "
        f"regime_month={payload['model']['n_regime_month']} month={payload['model']['n_month']}"
    )

    for name in ["baseline", "feature_gap_only"]:
        s = payload[name]
        print(f"\n{name}:")
        print(f"  decision={s['decision']} strict/policy={s['strict']}/{s['policy']} gap={s['mean_gap_diff']:.4f} drift={s['mean_drift']:.4f}")

    d = payload["delta_feature_vs_baseline"]
    print("\ndelta feature - baseline:")
    print(f"  mean_gap={d['mean_gap_diff']:.4f} mean_drift={d['mean_drift']:.4f}")


if __name__ == "__main__":
    main()
