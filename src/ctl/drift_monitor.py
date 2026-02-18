"""Drift Monitoring Infrastructure (Task 15).

Provides baseline profiling and drift detection for Phase 1a
post-Gate monitoring. Uses Population Stability Index (PSI) for
distribution drift and rolling metrics for outcome drift.

See docs/notes/Task15_assumptions.md for design rationale.

Usage
-----
>>> from ctl.drift_monitor import build_baseline, check_drift
>>> baseline = build_baseline(is_features_df, is_scores, is_r_values)
>>> snapshot = check_drift(new_features_df, new_scores, new_r_values, baseline)
>>> print(snapshot.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: PSI thresholds (standard model monitoring convention).
PSI_OK = 0.10
PSI_WATCH = 0.25

#: Default number of bins for PSI computation.
DEFAULT_N_BINS = 10

#: Rolling outcome defaults.
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_AVG_R_ALERT = 0.0
DEFAULT_WIN_RATE_ALERT = 0.35

Status = str  # "OK", "WATCH", "ALERT"


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    bin_edges: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Compute Population Stability Index between two distributions.

    Parameters
    ----------
    baseline : np.ndarray
        Reference distribution values.
    current : np.ndarray
        New distribution values to compare.
    n_bins : int
        Number of equal-frequency bins (from baseline quantiles).
    bin_edges : np.ndarray, optional
        Pre-computed bin edges. If None, computed from baseline.

    Returns
    -------
    psi : float
        Total PSI value.
    bin_edges : np.ndarray
        The bin edges used (for reuse in future checks).
    """
    eps = 1e-4

    if len(baseline) == 0 or len(current) == 0:
        return 0.0, np.array([])

    if bin_edges is None:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(baseline, quantiles)
        # Ensure unique edges.
        bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0, bin_edges

    # Count observations in each bin.
    baseline_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
    current_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    # Convert to proportions.
    baseline_pct = baseline_counts / max(baseline_counts.sum(), 1.0)
    current_pct = current_counts / max(current_counts.sum(), 1.0)

    # Floor zeros.
    baseline_pct = np.maximum(baseline_pct, eps)
    current_pct = np.maximum(current_pct, eps)

    # PSI per bin.
    psi_bins = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    psi = float(np.sum(psi_bins))

    return psi, bin_edges


def classify_psi(psi: float) -> Status:
    """Classify PSI value into OK / WATCH / ALERT."""
    if psi < PSI_OK:
        return "OK"
    elif psi < PSI_WATCH:
        return "WATCH"
    return "ALERT"


# ---------------------------------------------------------------------------
# Rolling outcome metrics
# ---------------------------------------------------------------------------

def rolling_avg_r(
    r_values: np.ndarray,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> np.ndarray:
    """Compute rolling average R over a trade window."""
    if len(r_values) < window:
        return np.array([float(np.mean(r_values))]) if len(r_values) > 0 else np.array([])
    return pd.Series(r_values).rolling(window).mean().dropna().values


def rolling_win_rate(
    r_values: np.ndarray,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> np.ndarray:
    """Compute rolling win rate (fraction > 0) over a trade window."""
    wins = (np.asarray(r_values) > 0).astype(float)
    if len(wins) < window:
        return np.array([float(np.mean(wins))]) if len(wins) > 0 else np.array([])
    return pd.Series(wins).rolling(window).mean().dropna().values


# ---------------------------------------------------------------------------
# Baseline profile
# ---------------------------------------------------------------------------

@dataclass
class BaselineProfile:
    """Reference profile from Phase 1a IS data."""

    # Per-feature bin edges for PSI (feature_name -> bin_edges array).
    feature_bin_edges: Dict[str, np.ndarray] = field(default_factory=dict)

    # Score bin edges.
    score_bin_edges: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-feature summary stats.
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Performance baselines.
    avg_r: float = 0.0
    win_rate: float = 0.0
    total_r: float = 0.0
    n_trades: int = 0

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            "feature_bin_edges": {
                k: v.tolist() for k, v in self.feature_bin_edges.items()
            },
            "score_bin_edges": self.score_bin_edges.tolist(),
            "feature_stats": self.feature_stats,
            "avg_r": self.avg_r,
            "win_rate": self.win_rate,
            "total_r": self.total_r,
            "n_trades": self.n_trades,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BaselineProfile":
        return cls(
            feature_bin_edges={
                k: np.array(v) for k, v in d.get("feature_bin_edges", {}).items()
            },
            score_bin_edges=np.array(d.get("score_bin_edges", [])),
            feature_stats=d.get("feature_stats", {}),
            avg_r=d.get("avg_r", 0.0),
            win_rate=d.get("win_rate", 0.0),
            total_r=d.get("total_r", 0.0),
            n_trades=d.get("n_trades", 0),
        )


def build_baseline(
    features_df: pd.DataFrame,
    scores: np.ndarray,
    r_values: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> BaselineProfile:
    """Build a baseline reference profile from IS data.

    Parameters
    ----------
    features_df : pd.DataFrame
        IS feature columns (9 candidates).
    scores : np.ndarray
        IS predicted scores from the model.
    r_values : np.ndarray
        IS TheoreticalR values.
    n_bins : int
        Number of bins for PSI quantiles.

    Returns
    -------
    BaselineProfile
    """
    profile = BaselineProfile()
    profile.n_trades = len(r_values)

    # Feature bin edges and stats.
    for col in features_df.columns:
        vals = features_df[col].dropna().values.astype(float)
        if len(vals) > 0:
            _, edges = compute_psi(vals, vals, n_bins=n_bins)
            profile.feature_bin_edges[col] = edges
            profile.feature_stats[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    # Score bin edges.
    if len(scores) > 0:
        _, profile.score_bin_edges = compute_psi(scores, scores, n_bins=n_bins)

    # Performance baselines.
    if len(r_values) > 0:
        profile.avg_r = float(np.mean(r_values))
        profile.win_rate = float(np.sum(r_values > 0) / len(r_values))
        profile.total_r = float(np.sum(r_values))

    return profile


# ---------------------------------------------------------------------------
# Drift check result
# ---------------------------------------------------------------------------

@dataclass
class MetricDrift:
    """Drift status for a single metric."""

    name: str
    value: float
    status: Status
    threshold_watch: Optional[float] = None
    threshold_alert: Optional[float] = None
    reason: str = ""


@dataclass
class DriftSnapshot:
    """Timestamped drift check result."""

    timestamp: str
    metrics: List[MetricDrift] = field(default_factory=list)

    @property
    def overall_status(self) -> Status:
        """Worst status across all metrics."""
        if any(m.status == "ALERT" for m in self.metrics):
            return "ALERT"
        if any(m.status == "WATCH" for m in self.metrics):
            return "WATCH"
        return "OK"

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "Drift Monitor Snapshot",
            "=" * 55,
            f"Timestamp: {self.timestamp}",
            f"Overall:   {self.overall_status}",
            "",
        ]
        for m in self.metrics:
            lines.append(
                f"  [{m.status:5s}] {m.name}: {m.value:.4f}"
                + (f"  — {m.reason}" if m.reason else "")
            )
        lines.append("=" * 55)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "metrics": [
                {
                    "name": m.name, "value": round(m.value, 6),
                    "status": m.status, "reason": m.reason,
                }
                for m in self.metrics
            ],
        }


# ---------------------------------------------------------------------------
# Main drift check
# ---------------------------------------------------------------------------

def check_drift(
    features_df: Optional[pd.DataFrame],
    scores: Optional[np.ndarray],
    r_values: Optional[np.ndarray],
    baseline: BaselineProfile,
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    avg_r_alert: float = DEFAULT_AVG_R_ALERT,
    win_rate_alert: float = DEFAULT_WIN_RATE_ALERT,
) -> DriftSnapshot:
    """Run drift checks against the baseline profile.

    Parameters
    ----------
    features_df : pd.DataFrame or None
        New feature values to check. None to skip feature drift.
    scores : np.ndarray or None
        New predicted scores. None to skip score drift.
    r_values : np.ndarray or None
        New realized R-multiples. None to skip outcome drift.
    baseline : BaselineProfile
        Reference profile from ``build_baseline``.
    rolling_window : int
        Window for rolling outcome metrics.
    avg_r_alert : float
        Alert threshold for rolling avg R.
    win_rate_alert : float
        Alert threshold for rolling win rate.

    Returns
    -------
    DriftSnapshot
    """
    snapshot = DriftSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Feature distribution drift.
    if features_df is not None:
        for col in features_df.columns:
            if col in baseline.feature_bin_edges:
                vals = features_df[col].dropna().values.astype(float)
                if len(vals) == 0:
                    continue
                edges = baseline.feature_bin_edges[col]
                psi, _ = compute_psi(np.array([]), vals, bin_edges=edges)
                # Re-compute properly: we need baseline values from edges.
                # PSI needs baseline counts — use uniform expected from edges.
                psi, _ = compute_psi(vals, vals, bin_edges=edges)
                # Actually: recompute using the stored edges against new data.
                # Baseline was equal-frequency → each bin ~= 1/n_bins.
                psi = _psi_from_edges(edges, vals)
                status = classify_psi(psi)
                reason = ""
                if status != "OK":
                    reason = f"PSI={psi:.4f} exceeds threshold"
                snapshot.metrics.append(MetricDrift(
                    name=f"feature_psi_{col}",
                    value=psi,
                    status=status,
                    threshold_watch=PSI_OK,
                    threshold_alert=PSI_WATCH,
                    reason=reason,
                ))

    # Score distribution drift.
    if scores is not None and len(baseline.score_bin_edges) > 1:
        psi = _psi_from_edges(baseline.score_bin_edges, scores)
        status = classify_psi(psi)
        reason = f"PSI={psi:.4f} exceeds threshold" if status != "OK" else ""
        snapshot.metrics.append(MetricDrift(
            name="score_psi",
            value=psi,
            status=status,
            threshold_watch=PSI_OK,
            threshold_alert=PSI_WATCH,
            reason=reason,
        ))

    # Outcome drift (rolling metrics).
    if r_values is not None and len(r_values) > 0:
        # Rolling avg R.
        roll_r = rolling_avg_r(r_values, window=rolling_window)
        if len(roll_r) > 0:
            latest_r = float(roll_r[-1])
            r_status = "ALERT" if latest_r < avg_r_alert else "OK"
            snapshot.metrics.append(MetricDrift(
                name="rolling_avg_r",
                value=latest_r,
                status=r_status,
                threshold_alert=avg_r_alert,
                reason=f"Rolling avg R = {latest_r:.4f} < {avg_r_alert}" if r_status == "ALERT" else "",
            ))

        # Rolling win rate.
        roll_wr = rolling_win_rate(r_values, window=rolling_window)
        if len(roll_wr) > 0:
            latest_wr = float(roll_wr[-1])
            wr_status = "ALERT" if latest_wr < win_rate_alert else "OK"
            snapshot.metrics.append(MetricDrift(
                name="rolling_win_rate",
                value=latest_wr,
                status=wr_status,
                threshold_alert=win_rate_alert,
                reason=f"Rolling win rate = {latest_wr:.4f} < {win_rate_alert}" if wr_status == "ALERT" else "",
            ))

    return snapshot


def _psi_from_edges(
    bin_edges: np.ndarray,
    current: np.ndarray,
    eps: float = 1e-4,
) -> float:
    """Compute PSI using pre-computed baseline bin edges.

    Assumes baseline was equal-frequency, so expected proportion
    per bin is 1 / n_bins.
    """
    if len(bin_edges) < 2 or len(current) == 0:
        return 0.0

    n_bins = len(bin_edges) - 1
    expected_pct = 1.0 / n_bins

    current_counts = np.histogram(current, bins=bin_edges)[0].astype(float)
    current_pct = current_counts / max(current_counts.sum(), 1.0)

    # Floor zeros.
    current_pct = np.maximum(current_pct, eps)
    expected_pct_safe = max(expected_pct, eps)

    psi_bins = (current_pct - expected_pct_safe) * np.log(current_pct / expected_pct_safe)
    return float(np.sum(psi_bins))
