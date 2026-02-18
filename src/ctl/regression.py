"""Phase 1a Elastic Net regression pipeline.

Trains a StandardScaler + ElasticNetCV pipeline on the assembled dataset
using purged time-series CV for alpha selection.  All features, expected
signs, and thresholds are loaded from the locked pre-registration artifact.

See docs/notes/Task10_assumptions.md for NaN handling and scaling rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from ctl.cross_validation import PurgedTimeSeriesSplit

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRE_REG = REPO_ROOT / "configs" / "pre_registration_v1.yaml"

# l1_ratio grid for ElasticNetCV (Ridge ← → Lasso).
_L1_RATIO_GRID = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]


# ---------------------------------------------------------------------------
# Pre-registration config
# ---------------------------------------------------------------------------

@dataclass
class PreRegConfig:
    """Parsed pre-registration parameters needed for training."""

    feature_names: List[str]
    expected_signs: Dict[str, str]
    target: str
    cluster_column: str
    random_seed: int
    cv_folds: int
    purge_gap_days: int


def load_pre_reg(path: Path = DEFAULT_PRE_REG) -> PreRegConfig:
    """Load and validate the pre-registration YAML.

    Raises ValueError if feature cap is violated or required keys are missing.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    candidates = raw["features"]["candidates"]
    if len(candidates) != 9:
        raise ValueError(
            f"Feature cap violated: expected 9 candidates, got {len(candidates)}"
        )

    feature_names = [c["name"] for c in candidates]
    expected_signs = {c["name"]: c["expected_sign"] for c in candidates}

    model = raw["model"]
    return PreRegConfig(
        feature_names=feature_names,
        expected_signs=expected_signs,
        target=model["target"],
        cluster_column=raw["features"]["control"][0]["name"],
        random_seed=model["random_seed"],
        cv_folds=model["cv_folds"],
        purge_gap_days=model["purge_gap_days"],
    )


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    cfg: PreRegConfig,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, List[str], List[str]]:
    """Extract feature matrix, target, and dates from assembled dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``dataset_assembler.assemble_dataset``.
    cfg : PreRegConfig

    Returns
    -------
    X_candidates : pd.DataFrame
        9 candidate feature columns (NaN filled, booleans as float).
    y : np.ndarray
        Target values (TheoreticalR).
    dates : pd.Series
        Trigger dates for CV splitting.
    candidate_cols : list of str
        Candidate column names (in order).
    cluster_dummies : pd.DataFrame
        One-hot encoded cluster columns (drop_first=True).

    Raises
    ------
    ValueError
        If any required feature column is missing from the dataset.
    """
    missing = [f for f in cfg.feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Required features missing from dataset: {missing}")
    if cfg.target not in df.columns:
        raise ValueError(f"Target column '{cfg.target}' missing from dataset")
    if cfg.cluster_column not in df.columns:
        raise ValueError(f"Cluster column '{cfg.cluster_column}' missing from dataset")

    # Extract candidate features.
    X_cand = df[cfg.feature_names].copy()

    # Fill NaN with 0.0 (neutral imputation — see Task10_assumptions.md #1-2).
    X_cand = X_cand.fillna(0.0).astype(float)

    # Target.
    y = df[cfg.target].values.astype(float)

    # Dates for CV.
    dates = df["Date"]

    # Cluster one-hot (drop first to avoid multicollinearity).
    cluster_dummies = pd.get_dummies(
        df[cfg.cluster_column], prefix="cluster", drop_first=True,
    ).astype(float)

    return X_cand, y, dates, cfg.feature_names, cluster_dummies


# ---------------------------------------------------------------------------
# Sign check
# ---------------------------------------------------------------------------

@dataclass
class SignCheck:
    """Coefficient sign vs pre-registered expectation for one feature."""

    feature: str
    coefficient: float
    expected_sign: str
    actual_sign: str   # "positive", "negative", or "zero"
    matches: bool


@dataclass
class CoefficientTable:
    """Complete coefficient report."""

    entries: List[SignCheck] = field(default_factory=list)
    intercept: float = 0.0
    cluster_coefs: Dict[str, float] = field(default_factory=dict)

    @property
    def n_matches(self) -> int:
        return sum(1 for e in self.entries if e.matches)

    @property
    def n_mismatches(self) -> int:
        return sum(1 for e in self.entries if not e.matches and e.actual_sign != "zero")

    @property
    def n_zeroed(self) -> int:
        return sum(1 for e in self.entries if e.actual_sign == "zero")

    @property
    def sign_flip_pause(self) -> bool:
        """Two or more sign flips triggers a PAUSE per kill criteria."""
        return self.n_mismatches >= 2


def check_signs(
    coefs: np.ndarray,
    feature_names: List[str],
    expected_signs: Dict[str, str],
) -> CoefficientTable:
    """Compare fitted coefficients against pre-registered expected signs.

    Parameters
    ----------
    coefs : np.ndarray
        Coefficient values for the 9 candidate features (same order as feature_names).
    feature_names : list of str
    expected_signs : dict of feature -> "positive" or "negative"

    Returns
    -------
    CoefficientTable with per-feature sign checks.
    """
    table = CoefficientTable()
    for name, coef in zip(feature_names, coefs):
        expected = expected_signs.get(name, "positive")
        if abs(coef) < 1e-10:
            actual = "zero"
            matches = True  # zeroed out by regularization, not a mismatch
        elif coef > 0:
            actual = "positive"
            matches = expected == "positive"
        else:
            actual = "negative"
            matches = expected == "negative"

        table.entries.append(SignCheck(
            feature=name,
            coefficient=float(coef),
            expected_sign=expected,
            actual_sign=actual,
            matches=matches,
        ))
    return table


# ---------------------------------------------------------------------------
# Tercile assignment
# ---------------------------------------------------------------------------

def assign_terciles(
    scores: np.ndarray,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Assign IS trades to top/mid/bottom terciles by predicted score.

    Remainder trades (when n % 3 != 0) are assigned to the top group.

    Returns
    -------
    buckets : np.ndarray of str
        "top", "mid", or "bottom" per trade.
    thresholds : (low_cutoff, high_cutoff)
        Score boundaries.  bottom < low_cutoff <= mid < high_cutoff <= top.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=object), (0.0, 0.0)

    third = n // 3
    sorted_scores = np.sort(scores)

    low_cutoff = float(sorted_scores[third])
    high_cutoff = float(sorted_scores[2 * third])

    buckets = np.where(
        scores >= high_cutoff, "top",
        np.where(scores >= low_cutoff, "mid", "bottom"),
    )
    return buckets, (low_cutoff, high_cutoff)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class Diagnostics:
    """Model diagnostics output."""

    correlation_matrix: pd.DataFrame
    multicollinearity_flags: List[str]
    coef_table: CoefficientTable
    alpha: float
    l1_ratio: float
    r_squared_is: float


def compute_diagnostics(
    X_candidates: pd.DataFrame,
    feature_names: List[str],
    threshold: float = 0.70,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute correlation matrix and flag high-correlation pairs.

    Parameters
    ----------
    X_candidates : pd.DataFrame
        Candidate features (9 columns).
    feature_names : list of str
    threshold : float
        |r| above this triggers a flag.

    Returns
    -------
    (correlation_matrix, list_of_flag_strings)
    """
    corr = X_candidates[feature_names].corr()
    flags = []
    for i, f1 in enumerate(feature_names):
        for j, f2 in enumerate(feature_names):
            if j <= i:
                continue
            r = corr.loc[f1, f2]
            if abs(r) > threshold:
                flags.append(f"{f1} x {f2}: r={r:.3f}")
    return corr, flags


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Complete output from a training run."""

    scaler: StandardScaler
    model: ElasticNetCV
    coef_table: CoefficientTable
    diagnostics: Diagnostics
    scores: np.ndarray
    buckets: np.ndarray
    tercile_thresholds: Tuple[float, float]
    feature_names: List[str]
    cluster_columns: List[str]


def train_model(
    df: pd.DataFrame,
    pre_reg_path: Path = DEFAULT_PRE_REG,
) -> ModelResult:
    """Train the Phase 1a Elastic Net model on the assembled IS dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Assembled dataset (output of ``dataset_assembler.assemble_dataset``).
        Should contain only IS data (2018-01-01 to 2024-12-31).
    pre_reg_path : Path
        Path to the locked pre-registration YAML.

    Returns
    -------
    ModelResult with fitted model, coefficients, scores, and diagnostics.
    """
    cfg = load_pre_reg(pre_reg_path)

    # Prepare features.
    X_cand, y, dates, feature_names, cluster_dummies = prepare_features(df, cfg)
    cluster_columns = list(cluster_dummies.columns)

    # Scale candidate features only.
    scaler = StandardScaler()
    X_cand_scaled = pd.DataFrame(
        scaler.fit_transform(X_cand),
        columns=feature_names,
        index=X_cand.index,
    )

    # Combine scaled candidates + unscaled cluster dummies.
    X = pd.concat([X_cand_scaled, cluster_dummies], axis=1)

    # Compute purged CV splits from dates.
    cv = PurgedTimeSeriesSplit(
        n_splits=cfg.cv_folds,
        purge_gap_days=cfg.purge_gap_days,
    )
    splits = [(tr, te) for tr, te, _ in cv.split(dates)]

    if not splits:
        raise ValueError(
            f"No valid CV folds (need at least {cfg.cv_folds + 1} blocks of data)"
        )

    # Fit ElasticNetCV.
    model = ElasticNetCV(
        l1_ratio=_L1_RATIO_GRID,
        cv=splits,
        random_state=cfg.random_seed,
        max_iter=10000,
    )
    model.fit(X.values, y)

    logger.info(
        "ElasticNetCV: alpha=%.6f, l1_ratio=%.3f, n_features=%d",
        model.alpha_, model.l1_ratio_, X.shape[1],
    )

    # Coefficient analysis (first 9 = candidates, rest = cluster dummies).
    candidate_coefs = model.coef_[:len(feature_names)]
    coef_table = check_signs(candidate_coefs, feature_names, cfg.expected_signs)
    coef_table.intercept = float(model.intercept_)
    coef_table.cluster_coefs = {
        col: float(model.coef_[len(feature_names) + i])
        for i, col in enumerate(cluster_columns)
    }

    # Predict IS scores.
    scores = model.predict(X.values)

    # Assign terciles.
    buckets, thresholds = assign_terciles(scores)

    # Diagnostics.
    corr_matrix, mc_flags = compute_diagnostics(X_cand, feature_names)
    r_squared = float(model.score(X.values, y))

    diagnostics = Diagnostics(
        correlation_matrix=corr_matrix,
        multicollinearity_flags=mc_flags,
        coef_table=coef_table,
        alpha=float(model.alpha_),
        l1_ratio=float(model.l1_ratio_),
        r_squared_is=r_squared,
    )

    if coef_table.sign_flip_pause:
        logger.warning(
            "PAUSE: %d coefficient sign flips detected (kill criteria threshold: 2)",
            coef_table.n_mismatches,
        )

    logger.info(
        "Training complete: R²=%.4f, %d/%d signs match, %d zeroed",
        r_squared, coef_table.n_matches, len(feature_names), coef_table.n_zeroed,
    )

    return ModelResult(
        scaler=scaler,
        model=model,
        coef_table=coef_table,
        diagnostics=diagnostics,
        scores=scores,
        buckets=buckets,
        tercile_thresholds=thresholds,
        feature_names=feature_names,
        cluster_columns=cluster_columns,
    )
