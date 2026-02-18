# Task 15 Assumptions — Archive + Drift Monitoring Setup

## Scope

1. **Phase 1a closeout only.** No Phase 2 features. No threshold changes
   to prior gates. This task finalizes the infrastructure for post-Gate
   monitoring and creates a reproducible archive.

## Drift Monitoring

2. **Population Stability Index (PSI)** for feature and score distribution
   drift. PSI is chosen over KS because it gives a single interpretable
   number with established thresholds (< 0.1 OK, 0.1–0.25 WATCH, > 0.25
   ALERT) and is standard in model monitoring.

3. **PSI computation.** Divide baseline and current distributions into
   equal-frequency bins (default 10 bins from baseline quantiles). For each
   bin: PSI_i = (actual_pct - expected_pct) * ln(actual_pct / expected_pct).
   Total PSI = sum of PSI_i. Zero-frequency bins are floored to a small
   epsilon (1e-4) to avoid log(0).

4. **Outcome drift.** Rolling average R and rolling win rate over a
   configurable window (default 20 trades). Drift is flagged when rolling
   metrics fall below configurable thresholds.

5. **Score distribution drift.** Same PSI metric applied to model predicted
   scores (baseline IS distribution vs new OOS/forward scores).

6. **Status levels.** Three levels per metric:
   - `OK`: within normal range
   - `WATCH`: approaching threshold, warrants attention
   - `ALERT`: threshold breached, action required

7. **Thresholds are configurable** but ship with Phase Gate Checklist
   defaults:
   - PSI: OK < 0.10, WATCH 0.10–0.25, ALERT > 0.25
   - Rolling avg R: ALERT if < 0.0 over window
   - Rolling win rate: ALERT if < 0.35 over window
   - Rolling spread proxy not implemented (requires ongoing tercile
     scoring which is out of scope for baseline setup).

8. **Baseline reference profile.** Stored as a dict of:
   - per-feature quantile bin edges (from IS data)
   - score quantile bin edges
   - summary stats (mean, std, min, max per feature)
   - performance baselines (avg R, win rate, total R from IS)

## Archive

9. **Archive manifest.** A JSON file listing all archived artifacts with
   their SHA-256 hashes, paths, and timestamps. The manifest itself is
   deterministic given the same input files.

10. **Archived artifacts include:**
    - `configs/pre_registration_v1.yaml`
    - `configs/phase1a.yaml`
    - `docs/governance/model_card_v1.md`
    - `docs/governance/phase_gate_decision_v1.md`
    - Gate decision JSON (if exists)
    - Dataset manifest JSON (if exists)

11. **No file copying.** The archive manifest records hashes and paths
    of files in place. It does not duplicate files into a separate
    directory (avoids file bloat per project constraints).

## Implementation

12. **Two modules.** `drift_monitor.py` for drift logic, `archive.py` for
    archive manifest generation. Keeps concerns separate.

13. **Deterministic.** PSI computation and archive hashing are fully
    deterministic given the same inputs. No randomness.
