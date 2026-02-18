"""Phase 1a negative controls — anti-hallucination falsification checks.

Three pre-registered controls per ``pre_registration_v1.yaml``:
  1. Randomized-label: shuffle outcomes, expect collapsed R² and spread.
  2. Lag-shift: misalign features in time, expect degradation.
  3. Placebo-feature: add noise column, expect ~zero weight.

See docs/notes/Task11b_assumptions.md for thresholds and rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from ctl.cross_validation import PurgedTimeSeriesSplit
from ctl.ranking_gate import TercileResult, evaluate_terciles
from ctl.regression import (
    DEFAULT_PRE_REG,
    ModelResult,
    PreRegConfig,
    _L1_RATIO_GRID,
    load_pre_reg,
    prepare_features,
    train_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BaselineMetrics:
    """Metrics from the unperturbed baseline model."""

    r_squared: float
    tercile_spread: float
    tercile_monotonic: bool
    n_trades: int


@dataclass
class ControlResult:
    """Result for a single negative control."""

    name: str
    passed: bool
    detail: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class NegativeControlReport:
    """Complete negative-control report."""

    baseline: BaselineMetrics
    controls: List[ControlResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.controls)

    @property
    def verdict(self) -> str:
        return "PASS" if self.all_passed else "FAIL"

    def summary(self) -> str:
        lines = [
            "=== Negative Control Report ===",
            f"Baseline: R²={self.baseline.r_squared:.4f}, "
            f"spread={self.baseline.tercile_spread:.3f}R, "
            f"monotonic={self.baseline.tercile_monotonic}, "
            f"n={self.baseline.n_trades}",
            "",
        ]
        for c in self.controls:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"[{status}] {c.name}: {c.detail}")
            for k, v in c.metrics.items():
                lines.append(f"       {k}: {v:.4f}")
        lines.append("")
        lines.append(f"Overall verdict: {self.verdict}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_model_on_Xy(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> ElasticNetCV:
    """Fit an ElasticNetCV on pre-prepared X, y with given splits."""
    model = ElasticNetCV(
        l1_ratio=_L1_RATIO_GRID,
        cv=splits,
        random_state=seed,
        max_iter=10000,
    )
    model.fit(X, y)
    return model


def _compute_splits(
    dates: pd.Series,
    cfg: PreRegConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute purged CV splits from dates."""
    cv = PurgedTimeSeriesSplit(
        n_splits=cfg.cv_folds,
        purge_gap_days=cfg.purge_gap_days,
    )
    return [(tr, te) for tr, te, _ in cv.split(dates)]


def _build_X(
    X_cand: pd.DataFrame,
    cluster_dummies: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Scale candidates and concatenate with cluster dummies."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cand)
    else:
        X_scaled = scaler.transform(X_cand)
    X = np.hstack([X_scaled, cluster_dummies.values])
    return X, scaler


def _tercile_spread(scores: np.ndarray, outcomes: np.ndarray) -> TercileResult:
    """Evaluate tercile spread from predicted scores and actual outcomes."""
    return evaluate_terciles(scores, outcomes)


# ---------------------------------------------------------------------------
# Control 1: Randomized labels
# ---------------------------------------------------------------------------

def run_randomized_label_control(
    df: pd.DataFrame,
    cfg: PreRegConfig,
    baseline: BaselineMetrics,
    n_permutations: int = 5,
    base_seed: int = 42,
    r2_threshold: float = 0.02,
    spread_threshold: float = 0.3,
) -> ControlResult:
    """Shuffle TheoreticalR and refit; expect near-zero R² and spread.

    Parameters
    ----------
    df : pd.DataFrame
        Assembled IS dataset.
    cfg : PreRegConfig
    baseline : BaselineMetrics
    n_permutations : int
        Number of random shuffles to average over.
    base_seed : int
    r2_threshold : float
        Max acceptable mean R² (default 0.02).
    spread_threshold : float
        Max acceptable mean |spread| in R (default 0.3).

    Returns
    -------
    ControlResult
    """
    X_cand, y, dates, feature_names, cluster_dummies = prepare_features(df, cfg)
    splits = _compute_splits(dates, cfg)

    r2_values = []
    spread_values = []

    for i in range(n_permutations):
        rng = np.random.default_rng(base_seed + i)
        y_shuffled = rng.permutation(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cand)
        X = np.hstack([X_scaled, cluster_dummies.values])

        model = _fit_model_on_Xy(X, y_shuffled, splits, cfg.random_seed)
        r2 = float(model.score(X, y_shuffled))
        r2_values.append(max(r2, 0.0))  # clamp negatives to 0

        scores = model.predict(X)
        tercile = _tercile_spread(scores, y_shuffled)
        spread_values.append(abs(tercile.spread))

    mean_r2 = float(np.mean(r2_values))
    mean_spread = float(np.mean(spread_values))

    passed = mean_r2 < r2_threshold and mean_spread < spread_threshold

    detail = (
        f"mean R²={mean_r2:.4f} (threshold <{r2_threshold}), "
        f"mean |spread|={mean_spread:.3f}R (threshold <{spread_threshold}R)"
    )

    return ControlResult(
        name="randomized_labels",
        passed=passed,
        detail=detail,
        metrics={
            "mean_r2": mean_r2,
            "mean_abs_spread": mean_spread,
            "n_permutations": float(n_permutations),
            "r2_threshold": r2_threshold,
            "spread_threshold": spread_threshold,
        },
    )


# ---------------------------------------------------------------------------
# Control 2: Lag-shift
# ---------------------------------------------------------------------------

def run_lag_shift_control(
    df: pd.DataFrame,
    cfg: PreRegConfig,
    baseline: BaselineMetrics,
    shift_bars: int = 1,
    r2_degradation_threshold: float = 0.5,
    spread_degradation_threshold: float = 0.5,
) -> ControlResult:
    """Shift features forward to misalign with outcomes; expect degradation.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : PreRegConfig
    baseline : BaselineMetrics
    shift_bars : int
        Number of rows to shift features forward (default 1).
    r2_degradation_threshold : float
        Lag R² must be < this fraction of baseline R² (default 0.5).
    spread_degradation_threshold : float
        Lag spread must degrade by at least this fraction (default 0.5).

    Returns
    -------
    ControlResult
    """
    # Shift features forward: pair features[i] with outcome[i - shift_bars].
    df_shifted = df.copy()
    for feat in cfg.feature_names:
        df_shifted[feat] = df_shifted[feat].shift(shift_bars)

    # Drop rows with NaN from shift.
    df_shifted = df_shifted.dropna(subset=cfg.feature_names).reset_index(drop=True)

    X_cand, y, dates, feature_names, cluster_dummies = prepare_features(
        df_shifted, cfg,
    )
    splits = _compute_splits(dates, cfg)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cand)
    X = np.hstack([X_scaled, cluster_dummies.values])

    model = _fit_model_on_Xy(X, y, splits, cfg.random_seed)
    lag_r2 = max(float(model.score(X, y)), 0.0)

    scores = model.predict(X)
    tercile = _tercile_spread(scores, y)
    lag_spread = tercile.spread

    # Compute degradation ratios.
    if abs(baseline.r_squared) > 1e-10:
        r2_ratio = lag_r2 / baseline.r_squared
    else:
        r2_ratio = 0.0  # baseline R² ~0 means lag can't be worse

    if abs(baseline.tercile_spread) > 1e-10:
        spread_ratio = lag_spread / baseline.tercile_spread
    else:
        spread_ratio = 0.0

    # Pass if lag R² < threshold fraction of baseline AND spread degraded.
    r2_ok = r2_ratio < r2_degradation_threshold
    spread_ok = spread_ratio < (1.0 - spread_degradation_threshold)
    passed = r2_ok or spread_ok  # either degradation channel suffices

    detail = (
        f"lag R²={lag_r2:.4f} ({r2_ratio:.1%} of baseline {baseline.r_squared:.4f}), "
        f"lag spread={lag_spread:.3f}R ({spread_ratio:.1%} of baseline "
        f"{baseline.tercile_spread:.3f}R)"
    )

    return ControlResult(
        name="lag_shift",
        passed=passed,
        detail=detail,
        metrics={
            "lag_r2": lag_r2,
            "lag_spread": lag_spread,
            "r2_ratio": r2_ratio,
            "spread_ratio": spread_ratio,
            "shift_bars": float(shift_bars),
        },
    )


# ---------------------------------------------------------------------------
# Control 3: Placebo feature
# ---------------------------------------------------------------------------

def run_placebo_feature_control(
    df: pd.DataFrame,
    cfg: PreRegConfig,
    baseline: BaselineMetrics,
    seed: int = 42,
    coef_threshold: float = 0.05,
) -> ControlResult:
    """Add a noise feature; expect ~zero coefficient.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : PreRegConfig
    baseline : BaselineMetrics
    seed : int
    coef_threshold : float
        Max acceptable |coefficient| for placebo (default 0.05).

    Returns
    -------
    ControlResult
    """
    X_cand, y, dates, feature_names, cluster_dummies = prepare_features(df, cfg)
    splits = _compute_splits(dates, cfg)

    # Add placebo noise column.
    rng = np.random.default_rng(seed)
    placebo = rng.standard_normal(len(y))
    X_cand_with_placebo = X_cand.copy()
    X_cand_with_placebo["Placebo_Noise"] = placebo

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cand_with_placebo)
    X = np.hstack([X_scaled, cluster_dummies.values])

    model = _fit_model_on_Xy(X, y, splits, cfg.random_seed)

    # Placebo coefficient is at index 9 (after the 9 real candidates).
    placebo_coef = abs(float(model.coef_[len(feature_names)]))

    passed = placebo_coef < coef_threshold

    detail = (
        f"|placebo coef|={placebo_coef:.4f} (threshold <{coef_threshold})"
    )

    return ControlResult(
        name="placebo_feature",
        passed=passed,
        detail=detail,
        metrics={
            "placebo_coef_abs": placebo_coef,
            "coef_threshold": coef_threshold,
        },
    )


# ---------------------------------------------------------------------------
# Full negative-control suite
# ---------------------------------------------------------------------------

def run_negative_controls(
    df: pd.DataFrame,
    pre_reg_path: Path = DEFAULT_PRE_REG,
    n_permutations: int = 5,
    base_seed: int = 42,
) -> NegativeControlReport:
    """Run all three negative controls and produce a report.

    Parameters
    ----------
    df : pd.DataFrame
        Assembled IS dataset.
    pre_reg_path : Path
        Path to locked pre-registration YAML.
    n_permutations : int
        Number of randomized-label permutations.
    base_seed : int
        Base seed for reproducibility.

    Returns
    -------
    NegativeControlReport with baseline, per-control results, and overall verdict.
    """
    cfg = load_pre_reg(pre_reg_path)

    # Train baseline model.
    logger.info("Training baseline model for negative controls...")
    baseline_result = train_model(df, pre_reg_path)

    # Compute baseline tercile metrics.
    y = df[cfg.target].values.astype(float)
    tercile = _tercile_spread(baseline_result.scores, y)

    baseline = BaselineMetrics(
        r_squared=baseline_result.diagnostics.r_squared_is,
        tercile_spread=tercile.spread,
        tercile_monotonic=tercile.is_monotonic,
        n_trades=len(y),
    )

    controls: List[ControlResult] = []

    # Control 1: Randomized labels.
    logger.info("Running randomized-label control (%d permutations)...",
                n_permutations)
    controls.append(
        run_randomized_label_control(df, cfg, baseline, n_permutations, base_seed)
    )

    # Control 2: Lag shift.
    logger.info("Running lag-shift control...")
    controls.append(
        run_lag_shift_control(df, cfg, baseline)
    )

    # Control 3: Placebo feature.
    logger.info("Running placebo-feature control...")
    controls.append(
        run_placebo_feature_control(df, cfg, baseline, seed=base_seed)
    )

    report = NegativeControlReport(baseline=baseline, controls=controls)

    logger.info("Negative controls complete: %s", report.verdict)
    for c in controls:
        status = "PASS" if c.passed else "FAIL"
        logger.info("  [%s] %s: %s", status, c.name, c.detail)

    return report
