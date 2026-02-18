"""OOS Test + Calibration (Task 13).

Scores out-of-sample triggers using locked IS-trained model artifacts
and produces gate-ready calibration outputs for Task 14 decision step.

See docs/notes/Task13_assumptions.md for design rationale.

Usage
-----
>>> from ctl.regression import train_model
>>> from ctl.oos_evaluation import evaluate_oos
>>>
>>> model_result = train_model(is_df, pre_reg_path)
>>> oos_result = evaluate_oos(oos_df, model_result)
>>> print(oos_result.summary())
>>> oos_result.save_json(Path("outputs/oos_gate_report.json"))
>>> oos_result.save_bucket_csv(Path("outputs/oos_bucket_table.csv"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ctl.regression import ModelResult, PreRegConfig, load_pre_reg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Gate 1 thresholds (from pre_registration_v1.yaml / Phase Gate Checklist).
MIN_OOS_TRADES = 30
MIN_TERCILE_SPREAD_R = 1.0
MIN_TOP_TERCILE_AVG_R = 0.5
MIN_SCORE_CORRELATION = 0.05


# ---------------------------------------------------------------------------
# Feature-lock scoring
# ---------------------------------------------------------------------------

def score_oos(
    oos_df: pd.DataFrame,
    model_result: ModelResult,
    pre_reg_path: Optional[Path] = None,
) -> np.ndarray:
    """Generate OOS scores using the locked IS-trained model.

    Enforces feature lock: only the 9 pre-registered candidate features
    + cluster dummies are used. Raises ValueError if any required feature
    is missing.

    Parameters
    ----------
    oos_df : pd.DataFrame
        OOS dataset with same schema as IS (SCHEMA_COLUMNS).
    model_result : ModelResult
        Output from ``regression.train_model`` on IS data.
    pre_reg_path : Path, optional
        Pre-registration YAML. Defaults to repo default.

    Returns
    -------
    np.ndarray
        Raw predicted scores (one per OOS trade).
    """
    feature_names = model_result.feature_names

    # --- Feature lock enforcement ---
    missing = [f for f in feature_names if f not in oos_df.columns]
    if missing:
        raise ValueError(
            f"Feature lock violation: required features missing from OOS data: {missing}"
        )
    if "AssetCluster" not in oos_df.columns:
        raise ValueError("Feature lock violation: 'AssetCluster' missing from OOS data")

    # Extract and prepare candidate features (same pipeline as IS).
    X_cand = oos_df[feature_names].copy()
    X_cand = X_cand.astype(float).fillna(0.0)

    # Scale using IS-fitted scaler (transform only, no re-fit).
    X_cand_scaled = pd.DataFrame(
        model_result.scaler.transform(X_cand),
        columns=feature_names,
        index=X_cand.index,
    )

    # Cluster dummies aligned to IS columns.
    cluster_dummies = pd.get_dummies(
        oos_df["AssetCluster"], prefix="cluster", drop_first=False,
    ).astype(float)
    cluster_dummies = cluster_dummies.reindex(
        columns=model_result.cluster_columns, fill_value=0.0,
    )

    # Combine and predict.
    X_oos = pd.concat([X_cand_scaled, cluster_dummies], axis=1)
    scores = model_result.model.predict(X_oos.values)
    return scores


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------

def assign_oos_terciles(
    scores: np.ndarray,
    is_thresholds: Tuple[float, float],
) -> np.ndarray:
    """Assign OOS scores to terciles using IS-derived thresholds.

    Parameters
    ----------
    scores : np.ndarray
        OOS predicted scores.
    is_thresholds : (low_cutoff, high_cutoff)
        Tercile boundaries from IS training.

    Returns
    -------
    np.ndarray of str
        "top", "mid", or "bottom" per trade.
    """
    low, high = is_thresholds
    return np.where(
        scores >= high, "top",
        np.where(scores >= low, "mid", "bottom"),
    )


def assign_quintiles(scores: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """Assign OOS scores to quintiles using OOS-internal quantiles.

    Returns
    -------
    labels : np.ndarray of str
        "Q1" (lowest) through "Q5" (highest).
    boundaries : list of 4 floats
        The 20th/40th/60th/80th percentile cutoffs.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=object), []

    boundaries = [float(np.percentile(scores, p)) for p in (20, 40, 60, 80)]
    p20, p40, p60, p80 = boundaries

    labels = np.where(
        scores >= p80, "Q5",
        np.where(
            scores >= p60, "Q4",
            np.where(
                scores >= p40, "Q3",
                np.where(scores >= p20, "Q2", "Q1"),
            ),
        ),
    )
    return labels, boundaries


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class BucketMetrics:
    """Metrics for a single score bucket (tercile or quintile)."""

    label: str
    n_trades: int
    avg_r: float
    win_rate: float
    total_r: float


def _compute_bucket_metrics(
    r_values: np.ndarray,
    label: str,
) -> BucketMetrics:
    """Compute metrics for a single bucket."""
    n = len(r_values)
    if n == 0:
        return BucketMetrics(label=label, n_trades=0, avg_r=0.0, win_rate=0.0, total_r=0.0)
    return BucketMetrics(
        label=label,
        n_trades=n,
        avg_r=float(np.mean(r_values)),
        win_rate=float(np.sum(r_values > 0) / n),
        total_r=float(np.sum(r_values)),
    )


def compute_bucket_table(
    r_values: np.ndarray,
    buckets: np.ndarray,
    bucket_order: List[str],
) -> List[BucketMetrics]:
    """Compute per-bucket metrics in specified order."""
    table: List[BucketMetrics] = []
    for label in bucket_order:
        mask = buckets == label
        table.append(_compute_bucket_metrics(r_values[mask], label))
    return table


# ---------------------------------------------------------------------------
# Gate criteria evaluation
# ---------------------------------------------------------------------------

@dataclass
class CriterionResult:
    """Pass/fail result for a single gate criterion."""

    criterion_id: str
    name: str
    passed: bool
    value: float
    threshold: float
    reason: str


def _check_trade_count(n: int) -> CriterionResult:
    passed = n >= MIN_OOS_TRADES
    return CriterionResult(
        criterion_id="G1.1",
        name="oos_trade_count",
        passed=passed,
        value=float(n),
        threshold=float(MIN_OOS_TRADES),
        reason=f"OOS trades = {n} ({'>='}{'<'[passed:]} {MIN_OOS_TRADES})"
        if not passed else f"OOS trades = {n} >= {MIN_OOS_TRADES}",
    )


def _check_tercile_spread(
    tercile_table: List[BucketMetrics],
) -> CriterionResult:
    top = next((b for b in tercile_table if b.label == "top"), None)
    bottom = next((b for b in tercile_table if b.label == "bottom"), None)
    top_r = top.avg_r if top else 0.0
    bot_r = bottom.avg_r if bottom else 0.0
    spread = top_r - bot_r
    passed = spread >= MIN_TERCILE_SPREAD_R
    return CriterionResult(
        criterion_id="G1.2",
        name="oos_tercile_spread",
        passed=passed,
        value=round(spread, 4),
        threshold=MIN_TERCILE_SPREAD_R,
        reason=f"Spread = {spread:.4f}R ({'>='}{'<'[passed:]} {MIN_TERCILE_SPREAD_R}R)"
        if not passed else f"Spread = {spread:.4f}R >= {MIN_TERCILE_SPREAD_R}R",
    )


def _check_monotonicity(
    tercile_table: List[BucketMetrics],
) -> CriterionResult:
    lookup = {b.label: b.avg_r for b in tercile_table}
    top_r = lookup.get("top", 0.0)
    mid_r = lookup.get("mid", 0.0)
    bot_r = lookup.get("bottom", 0.0)
    passed = top_r > mid_r > bot_r
    return CriterionResult(
        criterion_id="G1.3",
        name="score_monotonicity",
        passed=passed,
        value=0.0,
        threshold=0.0,
        reason=(
            f"top={top_r:.4f} > mid={mid_r:.4f} > bottom={bot_r:.4f}"
            if passed
            else f"Monotonicity broken: top={top_r:.4f}, mid={mid_r:.4f}, bottom={bot_r:.4f}"
        ),
    )


def _check_quintile_calibration(
    quintile_table: List[BucketMetrics],
) -> CriterionResult:
    avg_rs = [b.avg_r for b in quintile_table]
    # Directionally calibrated = monotonically non-decreasing from Q1 to Q5.
    monotonic = all(avg_rs[i] <= avg_rs[i + 1] for i in range(len(avg_rs) - 1))
    return CriterionResult(
        criterion_id="G1.9",
        name="quintile_calibration",
        passed=monotonic,
        value=0.0,
        threshold=0.0,
        reason=(
            "Quintile avg R monotonically increasing"
            if monotonic
            else f"Calibration broken: quintile avg Rs = {[round(r, 4) for r in avg_rs]}"
        ),
    )


def _check_top_tercile_kill(
    tercile_table: List[BucketMetrics],
) -> CriterionResult:
    top = next((b for b in tercile_table if b.label == "top"), None)
    top_r = top.avg_r if top else 0.0
    passed = top_r >= MIN_TOP_TERCILE_AVG_R
    return CriterionResult(
        criterion_id="K.1",
        name="top_tercile_min_r",
        passed=passed,
        value=round(top_r, 4),
        threshold=MIN_TOP_TERCILE_AVG_R,
        reason=(
            f"Top tercile avg R = {top_r:.4f} >= {MIN_TOP_TERCILE_AVG_R}R"
            if passed
            else f"KILL: Top tercile avg R = {top_r:.4f} < {MIN_TOP_TERCILE_AVG_R}R"
        ),
    )


def _check_score_correlation(
    scores: np.ndarray,
    r_values: np.ndarray,
) -> CriterionResult:
    if len(scores) < 2:
        corr = 0.0
    else:
        corr = float(np.corrcoef(scores, r_values)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    passed = corr >= MIN_SCORE_CORRELATION
    return CriterionResult(
        criterion_id="K.2",
        name="score_outcome_correlation",
        passed=passed,
        value=round(corr, 4),
        threshold=MIN_SCORE_CORRELATION,
        reason=(
            f"Score-R correlation = {corr:.4f} >= {MIN_SCORE_CORRELATION}"
            if passed
            else f"KILL: Score-R correlation = {corr:.4f} < {MIN_SCORE_CORRELATION}"
        ),
    )


# ---------------------------------------------------------------------------
# Calibration reliability table
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRow:
    """One row in the calibration reliability table."""

    quintile: str
    score_low: float
    score_high: float
    n_trades: int
    avg_predicted_score: float
    avg_realized_r: float
    win_rate: float


def build_calibration_table(
    scores: np.ndarray,
    r_values: np.ndarray,
    quintile_labels: np.ndarray,
    quintile_boundaries: List[float],
) -> List[CalibrationRow]:
    """Build the calibration reliability table (quintile bands vs realized R).

    Parameters
    ----------
    scores : raw OOS predicted scores
    r_values : realized TheoreticalR
    quintile_labels : Q1-Q5 assignments
    quintile_boundaries : 4 boundary values (20th/40th/60th/80th percentiles)

    Returns
    -------
    List of CalibrationRow, one per quintile Q1 through Q5.
    """
    # Build boundary ranges.
    if len(quintile_boundaries) == 4:
        lo_hi = [
            (float("-inf"), quintile_boundaries[0]),
            (quintile_boundaries[0], quintile_boundaries[1]),
            (quintile_boundaries[1], quintile_boundaries[2]),
            (quintile_boundaries[2], quintile_boundaries[3]),
            (quintile_boundaries[3], float("inf")),
        ]
    else:
        lo_hi = [(float("-inf"), float("inf"))] * 5

    rows: List[CalibrationRow] = []
    for i, qlabel in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"]):
        mask = quintile_labels == qlabel
        s = scores[mask]
        r = r_values[mask]
        n = len(r)
        rows.append(CalibrationRow(
            quintile=qlabel,
            score_low=lo_hi[i][0],
            score_high=lo_hi[i][1],
            n_trades=n,
            avg_predicted_score=float(np.mean(s)) if n > 0 else 0.0,
            avg_realized_r=float(np.mean(r)) if n > 0 else 0.0,
            win_rate=float(np.sum(r > 0) / n) if n > 0 else 0.0,
        ))
    return rows


# ---------------------------------------------------------------------------
# OOS evaluation result
# ---------------------------------------------------------------------------

@dataclass
class OOSResult:
    """Complete OOS evaluation output, consumable by Task 14."""

    n_trades: int
    scores: np.ndarray
    tercile_buckets: np.ndarray
    quintile_labels: np.ndarray

    # Metric tables.
    tercile_table: List[BucketMetrics]
    quintile_table: List[BucketMetrics]
    calibration_table: List[CalibrationRow]

    # Summary metrics.
    tercile_spread: float
    score_r_correlation: float

    # Gate criteria results.
    criteria: List[CriterionResult]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    @property
    def failed_criteria(self) -> List[CriterionResult]:
        return [c for c in self.criteria if not c.passed]

    @property
    def verdict(self) -> str:
        """Human-readable verdict (NOT the final gate decision — that is Task 14)."""
        if self.all_passed:
            return "ALL_CRITERIA_MET"
        return "CRITERIA_NOT_MET"

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "OOS Evaluation Report",
            "=" * 60,
            f"Total OOS trades: {self.n_trades}",
            f"Score-R correlation: {self.score_r_correlation:.4f}",
            f"Tercile spread (top - bottom avg R): {self.tercile_spread:.4f}R",
            "",
            "--- Tercile Performance ---",
        ]
        for b in self.tercile_table:
            lines.append(
                f"  {b.label:>8s}: n={b.n_trades:3d}  avg_R={b.avg_r:+.4f}  "
                f"win_rate={b.win_rate:.1%}  total_R={b.total_r:+.2f}"
            )
        lines.append("")
        lines.append("--- Quintile Calibration ---")
        for row in self.calibration_table:
            lines.append(
                f"  {row.quintile}: n={row.n_trades:3d}  "
                f"avg_pred={row.avg_predicted_score:+.4f}  "
                f"avg_real={row.avg_realized_r:+.4f}  "
                f"win_rate={row.win_rate:.1%}"
            )
        lines.append("")
        lines.append("--- Gate Criteria ---")
        for c in self.criteria:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.criterion_id} {c.name}: {c.reason}")
        lines.append("")
        lines.append(f"Verdict: {self.verdict}")
        lines.append("(Final gate decision is Task 14's responsibility)")
        lines.append("=" * 60)
        return "\n".join(lines)

    # --- Export ---

    def to_dict(self) -> Dict:
        """JSON-serializable summary dict."""
        return {
            "n_trades": self.n_trades,
            "tercile_spread": round(self.tercile_spread, 4),
            "score_r_correlation": round(self.score_r_correlation, 4),
            "verdict": self.verdict,
            "tercile_table": [
                {
                    "label": b.label, "n_trades": b.n_trades,
                    "avg_r": round(b.avg_r, 4), "win_rate": round(b.win_rate, 4),
                    "total_r": round(b.total_r, 4),
                }
                for b in self.tercile_table
            ],
            "quintile_table": [
                {
                    "label": b.label, "n_trades": b.n_trades,
                    "avg_r": round(b.avg_r, 4), "win_rate": round(b.win_rate, 4),
                    "total_r": round(b.total_r, 4),
                }
                for b in self.quintile_table
            ],
            "calibration_table": [
                {
                    "quintile": r.quintile, "n_trades": r.n_trades,
                    "avg_predicted_score": round(r.avg_predicted_score, 4),
                    "avg_realized_r": round(r.avg_realized_r, 4),
                    "win_rate": round(r.win_rate, 4),
                }
                for r in self.calibration_table
            ],
            "criteria": [
                {
                    "id": c.criterion_id, "name": c.name,
                    "passed": c.passed, "value": c.value,
                    "threshold": c.threshold, "reason": c.reason,
                }
                for c in self.criteria
            ],
        }

    def save_json(self, path: Path) -> Path:
        """Write gate-facing JSON summary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def save_bucket_csv(self, path: Path) -> Path:
        """Write combined tercile + quintile bucket table as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for b in self.tercile_table:
            rows.append({
                "group_type": "tercile", "label": b.label,
                "n_trades": b.n_trades, "avg_r": b.avg_r,
                "win_rate": b.win_rate, "total_r": b.total_r,
            })
        for b in self.quintile_table:
            rows.append({
                "group_type": "quintile", "label": b.label,
                "n_trades": b.n_trades, "avg_r": b.avg_r,
                "win_rate": b.win_rate, "total_r": b.total_r,
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_oos(
    oos_df: pd.DataFrame,
    model_result: ModelResult,
    target_col: str = "TheoreticalR",
) -> OOSResult:
    """Run the full OOS evaluation pipeline.

    Parameters
    ----------
    oos_df : pd.DataFrame
        Out-of-sample dataset (same SCHEMA_COLUMNS as IS).
    model_result : ModelResult
        Fitted model artifacts from ``regression.train_model``.
    target_col : str
        Column containing realized R-multiples for evaluation.

    Returns
    -------
    OOSResult
        Complete evaluation output with metrics, criteria, and export methods.
    """
    n = len(oos_df)

    if n == 0:
        empty_scores = np.array([])
        empty_buckets = np.array([], dtype=object)
        return OOSResult(
            n_trades=0,
            scores=empty_scores,
            tercile_buckets=empty_buckets,
            quintile_labels=empty_buckets,
            tercile_table=[],
            quintile_table=[],
            calibration_table=[],
            tercile_spread=0.0,
            score_r_correlation=0.0,
            criteria=[
                _check_trade_count(0),
            ],
        )

    if target_col not in oos_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from OOS data")

    # Score OOS trades.
    scores = score_oos(oos_df, model_result)
    r_values = oos_df[target_col].values.astype(float)

    # Assign buckets.
    tercile_buckets = assign_oos_terciles(scores, model_result.tercile_thresholds)
    quintile_labels, quintile_boundaries = assign_quintiles(scores)

    # Compute metric tables.
    tercile_table = compute_bucket_table(
        r_values, tercile_buckets, ["top", "mid", "bottom"],
    )
    quintile_table = compute_bucket_table(
        r_values, quintile_labels, ["Q1", "Q2", "Q3", "Q4", "Q5"],
    )

    # Calibration reliability table.
    calibration_table = build_calibration_table(
        scores, r_values, quintile_labels, quintile_boundaries,
    )

    # Summary metrics.
    top_avg = next((b.avg_r for b in tercile_table if b.label == "top"), 0.0)
    bot_avg = next((b.avg_r for b in tercile_table if b.label == "bottom"), 0.0)
    tercile_spread = top_avg - bot_avg

    if n >= 2:
        corr = float(np.corrcoef(scores, r_values)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    # Gate criteria.
    criteria = [
        _check_trade_count(n),
        _check_tercile_spread(tercile_table),
        _check_monotonicity(tercile_table),
        _check_quintile_calibration(quintile_table),
        _check_top_tercile_kill(tercile_table),
        _check_score_correlation(scores, r_values),
    ]

    return OOSResult(
        n_trades=n,
        scores=scores,
        tercile_buckets=tercile_buckets,
        quintile_labels=quintile_labels,
        tercile_table=tercile_table,
        quintile_table=quintile_table,
        calibration_table=calibration_table,
        tercile_spread=tercile_spread,
        score_r_correlation=corr,
        criteria=criteria,
    )
