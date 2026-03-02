#!/usr/bin/env python3
"""Evaluate COT/VIX feature ablation on locked IS/OOS boundaries.

Runs a compact ablation study using the existing Phase 1a Elastic Net pipeline:
- full (all 9 pre-registered features)
- no_cot (drops COT_20D_Delta, COT_ZScore_1Y)
- no_vix (drops VIX_Regime)
- no_external (drops all external features above)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.cross_validation import PurgedTimeSeriesSplit  # noqa: E402
from ctl.oos_evaluation import evaluate_oos  # noqa: E402
from ctl.regression import ModelResult, check_signs, compute_diagnostics, load_pre_reg  # noqa: E402

DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "processed" / "cutover_v1" / "datasets"
DEFAULT_PRE_REG = REPO_ROOT / "configs" / "pre_registration_v1.yaml"
_L1_RATIO_GRID = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]


def _latest_dataset(path: Path) -> Path:
    files = sorted(path.glob("phase1a_triggers_v1_full_universe_real_*.csv"))
    if not files:
        raise FileNotFoundError(f"No strict dataset found under: {path}")
    return files[-1]


def _split_is_oos(df: pd.DataFrame, is_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    is_mask = data["Date"] <= pd.Timestamp(is_end)
    return (
        data.loc[is_mask].reset_index(drop=True),
        data.loc[~is_mask].reset_index(drop=True),
    )


def _train_variant(is_df: pd.DataFrame, feature_names: list[str], pre_reg_path: Path) -> ModelResult:
    cfg = load_pre_reg(pre_reg_path)
    cfg.feature_names = feature_names
    cfg.expected_signs = {k: v for k, v in cfg.expected_signs.items() if k in feature_names}

    missing = [f for f in feature_names if f not in is_df.columns]
    if missing:
        raise ValueError(f"Missing features for variant: {missing}")

    X_cand = is_df[feature_names].copy().astype(float).fillna(0.0)
    y = is_df[cfg.target].values.astype(float)
    dates = pd.to_datetime(is_df["Date"])
    cluster_dummies = pd.get_dummies(
        is_df[cfg.cluster_column], prefix="cluster", drop_first=True
    ).astype(float)
    cluster_columns = list(cluster_dummies.columns)

    scaler = StandardScaler()
    X_cand_scaled = pd.DataFrame(
        scaler.fit_transform(X_cand),
        columns=feature_names,
        index=X_cand.index,
    )
    X = pd.concat([X_cand_scaled, cluster_dummies], axis=1)

    cv = PurgedTimeSeriesSplit(
        n_splits=cfg.cv_folds,
        purge_gap_days=cfg.purge_gap_days,
    )
    splits = [(tr, te) for tr, te, _ in cv.split(dates)]
    if not splits:
        raise ValueError("No valid CV splits for variant training")

    model = ElasticNetCV(
        l1_ratio=_L1_RATIO_GRID,
        cv=splits,
        random_state=cfg.random_seed,
        max_iter=10000,
    )
    model.fit(X.values, y)

    scores = model.predict(X.values)
    third = len(scores) // 3
    sorted_scores = np.sort(scores)
    low_cutoff = float(sorted_scores[third]) if len(scores) else 0.0
    high_cutoff = float(sorted_scores[2 * third]) if len(scores) else 0.0
    buckets = np.where(
        scores >= high_cutoff, "top",
        np.where(scores >= low_cutoff, "mid", "bottom"),
    )

    coef_table = check_signs(model.coef_[:len(feature_names)], feature_names, cfg.expected_signs)
    coef_table.intercept = float(model.intercept_)
    coef_table.cluster_coefs = {
        col: float(model.coef_[len(feature_names) + i]) for i, col in enumerate(cluster_columns)
    }
    corr, flags = compute_diagnostics(X_cand, feature_names)

    diagnostics = type("Diag", (), {})()
    diagnostics.correlation_matrix = corr
    diagnostics.multicollinearity_flags = flags
    diagnostics.coef_table = coef_table
    diagnostics.alpha = float(model.alpha_)
    diagnostics.l1_ratio = float(model.l1_ratio_)
    diagnostics.r_squared_is = float(model.score(X.values, y))

    return ModelResult(
        scaler=scaler,
        model=model,
        coef_table=coef_table,
        diagnostics=diagnostics,
        scores=scores,
        buckets=buckets,
        tercile_thresholds=(low_cutoff, high_cutoff),
        feature_names=feature_names,
        cluster_columns=cluster_columns,
    )


def _summarize_variant(
    name: str,
    model_result: ModelResult,
    oos_result,
) -> dict:
    top = next((b for b in oos_result.tercile_table if b.label == "top"), None)
    mid = next((b for b in oos_result.tercile_table if b.label == "mid"), None)
    bot = next((b for b in oos_result.tercile_table if b.label == "bottom"), None)
    return {
        "variant": name,
        "n_features": len(model_result.feature_names),
        "features": model_result.feature_names,
        "is_r2": round(model_result.diagnostics.r_squared_is, 6),
        "oos_n_trades": oos_result.n_trades,
        "oos_score_r_corr": round(oos_result.score_r_correlation, 6),
        "oos_tercile_spread": round(oos_result.tercile_spread, 6),
        "oos_top_avg_r": round(top.avg_r if top else 0.0, 6),
        "oos_mid_avg_r": round(mid.avg_r if mid else 0.0, 6),
        "oos_bottom_avg_r": round(bot.avg_r if bot else 0.0, 6),
        "oos_verdict": oos_result.verdict,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate COT/VIX ablation on current strict dataset.")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--pre-reg", type=Path, default=DEFAULT_PRE_REG)
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    dataset_path = args.dataset or _latest_dataset(DEFAULT_DATASET_DIR)
    df = pd.read_csv(dataset_path)

    pre = load_pre_reg(args.pre_reg)
    is_df, oos_df = _split_is_oos(df, is_end="2024-12-31")

    full = list(pre.feature_names)
    no_cot = [f for f in full if f not in {"COT_20D_Delta", "COT_ZScore_1Y"}]
    no_vix = [f for f in full if f != "VIX_Regime"]
    no_external = [f for f in full if f not in {"COT_20D_Delta", "COT_ZScore_1Y", "VIX_Regime"}]

    variants = {
        "full": full,
        "no_cot": no_cot,
        "no_vix": no_vix,
        "no_external": no_external,
    }

    rows = []
    for name, feats in variants.items():
        model_result = _train_variant(is_df, feats, args.pre_reg)
        oos_result = evaluate_oos(oos_df, model_result)
        rows.append(_summarize_variant(name, model_result, oos_result))

    baseline = next(r for r in rows if r["variant"] == "full")
    deltas = []
    for r in rows:
        if r["variant"] == "full":
            continue
        deltas.append({
            "variant": r["variant"],
            "delta_is_r2": round(r["is_r2"] - baseline["is_r2"], 6),
            "delta_oos_score_r_corr": round(r["oos_score_r_corr"] - baseline["oos_score_r_corr"], 6),
            "delta_oos_tercile_spread": round(r["oos_tercile_spread"] - baseline["oos_tercile_spread"], 6),
            "delta_oos_top_avg_r": round(r["oos_top_avg_r"] - baseline["oos_top_avg_r"], 6),
        })

    payload = {
        "dataset_path": str(dataset_path),
        "is_rows": len(is_df),
        "oos_rows": len(oos_df),
        "variants": rows,
        "deltas_vs_full": deltas,
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Dataset: {dataset_path}")
        print(f"IS/OOS rows: {len(is_df)}/{len(oos_df)}")
        print()
        print("Variant       IS_R2     OOS_Corr  OOS_Spread  OOS_TopR  Verdict")
        print("-------       -----     --------  ----------  --------  -------")
        for r in rows:
            print(
                f"{r['variant']:11s} {r['is_r2']:8.4f} "
                f"{r['oos_score_r_corr']:9.4f} {r['oos_tercile_spread']:10.4f} "
                f"{r['oos_top_avg_r']:8.4f} {r['oos_verdict']}"
            )
        print("\nDeltas vs full:")
        for d in deltas:
            print(
                f"  {d['variant']:11s} "
                f"dR2={d['delta_is_r2']:+.4f} "
                f"dCorr={d['delta_oos_score_r_corr']:+.4f} "
                f"dSpread={d['delta_oos_tercile_spread']:+.4f} "
                f"dTopR={d['delta_oos_top_avg_r']:+.4f}"
            )


if __name__ == "__main__":
    main()
