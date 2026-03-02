#!/usr/bin/env python3
"""Evaluate COT fusion replacements vs legacy baseline on locked IS/OOS split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.oos_evaluation import evaluate_oos  # noqa: E402
from ctl.regression import train_model  # noqa: E402


def _latest_dataset(dataset_dir: Path) -> Path:
    files = sorted(dataset_dir.glob("phase1a_triggers_v1_full_universe_real*.csv"))
    if not files:
        raise FileNotFoundError(f"No strict dataset found under: {dataset_dir}")
    return files[-1]


def _split_is_oos(df: pd.DataFrame, is_end: str = "2024-12-31") -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    is_mask = data["Date"] <= pd.Timestamp(is_end)
    return (
        data.loc[is_mask].reset_index(drop=True),
        data.loc[~is_mask].reset_index(drop=True),
    )


def _variant_dataset(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    d = df.copy()
    if variant == "baseline_legacy":
        return d

    mapping: Dict[str, Tuple[str, str]] = {
        "fusion_mean": ("fusion_mean_delta", "fusion_mean_z"),
        "fusion_median": ("fusion_median_delta", "fusion_median_z"),
        "fusion_shrink": ("fusion_shrink_delta", "fusion_shrink_z"),
        "fusion_consensus": ("fusion_consensus_delta", "fusion_consensus_z"),
    }
    if variant not in mapping:
        raise ValueError(f"Unknown variant: {variant}")
    delta_col, z_col = mapping[variant]
    for c in [delta_col, z_col]:
        if c not in d.columns:
            raise ValueError(f"Fusion feature missing in merged dataset: {c}")
    d["COT_20D_Delta"] = d[delta_col]
    d["COT_ZScore_1Y"] = d[z_col]
    return d


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate fusion COT variants against legacy baseline.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Strict dataset path. Defaults to latest under datasets/",
    )
    p.add_argument(
        "--fusion-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_fusion_features_latest.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_fusion_ablation_latest.json",
    )
    p.add_argument(
        "--tracking-config",
        type=Path,
        default=REPO_ROOT / "configs" / "cutover" / "cot_tracking_v1.yaml",
    )
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    dataset_path = args.dataset or _latest_dataset(
        REPO_ROOT / "data" / "processed" / "cutover_v1" / "datasets"
    )
    base = pd.read_csv(dataset_path)
    fusion = pd.read_csv(args.fusion_csv)
    merge_cols = [
        "Date", "Ticker",
        "fusion_mean_delta", "fusion_mean_z",
        "fusion_median_delta", "fusion_median_z",
        "fusion_shrink_delta", "fusion_shrink_z",
        "fusion_consensus_delta", "fusion_consensus_z",
        "cot_agree_sign", "cot_disagreement_frac", "cot_dispersion_z",
    ]
    m = fusion[merge_cols].copy()
    merged = base.merge(m, how="left", on=["Date", "Ticker"], validate="one_to_one")

    variants = [
        "baseline_legacy",
        "fusion_mean",
        "fusion_median",
        "fusion_shrink",
        "fusion_consensus",
    ]

    rows = []
    baseline_metrics = None
    for v in variants:
        d = _variant_dataset(merged, v)
        is_df, oos_df = _split_is_oos(d)
        model = train_model(is_df)
        oos = evaluate_oos(oos_df, model)
        rec = {
            "variant": v,
            "is_r2": round(model.diagnostics.r_squared_is, 6),
            "oos_n_trades": int(oos.n_trades),
            "oos_corr": round(oos.score_r_correlation, 6),
            "oos_spread": round(oos.tercile_spread, 6),
            "oos_top_avg_r": round(
                next((b.avg_r for b in oos.tercile_table if b.label == "top"), 0.0), 6
            ),
            "oos_verdict": oos.verdict,
        }
        rows.append(rec)
        if v == "baseline_legacy":
            baseline_metrics = rec

    if baseline_metrics is None:
        raise RuntimeError("Baseline metrics missing")

    for r in rows:
        r["lift_corr_vs_baseline"] = round(r["oos_corr"] - baseline_metrics["oos_corr"], 6)
        r["lift_spread_vs_baseline"] = round(r["oos_spread"] - baseline_metrics["oos_spread"], 6)
        r["lift_top_r_vs_baseline"] = round(
            r["oos_top_avg_r"] - baseline_metrics["oos_top_avg_r"], 6
        )
        r["positive_lift"] = bool(
            r["variant"] != "baseline_legacy"
            and r["lift_spread_vs_baseline"] > 0
            and r["lift_top_r_vs_baseline"] >= 0
            and r["lift_corr_vs_baseline"] >= -0.01
        )

    positive = [r for r in rows if r["positive_lift"]]
    positive_sorted = sorted(
        positive,
        key=lambda x: (x["lift_spread_vs_baseline"], x["lift_top_r_vs_baseline"], x["lift_corr_vs_baseline"]),
        reverse=True,
    )
    secondary = positive_sorted[0]["variant"] if positive_sorted else None

    payload = {
        "dataset_path": str(dataset_path),
        "fusion_csv": str(args.fusion_csv),
        "variants": rows,
        "primary_variant": "baseline_legacy",
        "secondary_variant": secondary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    tracking = {
        "version": "cot_tracking_v1",
        "primary_source": "legacy",
        "primary_variant": "baseline_legacy",
        "secondary_variant": secondary,
        "secondary_enabled": secondary is not None,
        "selection_rule": (
            "positive_lift requires spread_lift>0, top_r_lift>=0, corr_lift>=-0.01"
        ),
    }
    args.tracking_config.parent.mkdir(parents=True, exist_ok=True)
    args.tracking_config.write_text(yaml.safe_dump(tracking, sort_keys=False), encoding="utf-8")

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print(f"Saved ablation: {args.out}")
    print(f"Saved tracking: {args.tracking_config}")
    print()
    print("Variant          Corr     Spread   TopR     dCorr    dSpread  dTopR   Positive")
    print("-------          ----     ------   ----     -----    -------  -----   --------")
    for r in rows:
        print(
            f"{r['variant']:15s} {r['oos_corr']:7.4f}  {r['oos_spread']:7.4f}  {r['oos_top_avg_r']:7.4f}  "
            f"{r['lift_corr_vs_baseline']:7.4f}  {r['lift_spread_vs_baseline']:7.4f}  "
            f"{r['lift_top_r_vs_baseline']:7.4f}  {str(r['positive_lift'])}"
        )
    print()
    print(f"Primary: baseline_legacy")
    print(f"Secondary: {secondary}")


if __name__ == "__main__":
    main()
