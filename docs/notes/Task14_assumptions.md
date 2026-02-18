# Task 14 Assumptions — Gate 1 Pass/Fail Decision Package

## Scope

1. **Decision only, no threshold changes.** This module evaluates Gate 1
   criteria against locked thresholds from the Phase Gate Checklist and
   `pre_registration_v1.yaml`. It does not modify any thresholds.

2. **No model retraining or data processing.** Consumes pre-computed outputs
   from Tasks 4/4b/11b/11c/13.

## Inputs

3. **Typed upstream results.** The decision function accepts:
   - `OOSResult` (Task 13) — OOS trade count, tercile spread, monotonicity,
     quintile calibration, kill criteria (top-tercile min R, score-R corr)
   - `StressReport` (Task 4b) — `gate_pass` (profitable at 2 ticks)
   - `NegativeControlReport` (Task 11b) — `all_passed`
   - `DegradationReport` (Task 11c) — `all_passed`
   - `feature_cap_respected: bool` — manual attestation (Item 4)
   - `model_card_complete: bool` — manual attestation (Item 5)

4. **Optional inputs.** Any upstream result may be `None` (not yet computed).
   Missing inputs are marked `INCOMPLETE` rather than `PASS` or `FAIL`.

## Gate 1 Criteria (9 Items)

5. **Criteria map to Phase Gate Checklist Gate 1 items 1-9:**
   | Item | Source | Field |
   |------|--------|-------|
   | 1. OOS trades >= 30 | OOSResult.criteria G1.1 | |
   | 2. Tercile spread >= 1.0R | OOSResult.criteria G1.2 | |
   | 3. Monotonicity (top > mid > bottom) | OOSResult.criteria G1.3 | |
   | 4. Feature cap respected | Manual bool | |
   | 5. Model card complete | Manual bool | |
   | 6. Negative controls passed | NegativeControlReport.all_passed | |
   | 7. Entry degradation within tolerances | DegradationReport.all_passed | |
   | 8. Slippage stress: profitable at 2 ticks | StressReport.gate_pass | |
   | 9. Quintile calibration | OOSResult.criteria G1.9 | |

## Kill Criteria

6. **Kill criteria from OOSResult.** Two kill-level checks are evaluated
   in Task 13 (`K.1` top-tercile avg R < 0.5R, `K.2` score-R corr < 0.05).
   If either triggers, overall verdict is `REJECT`.

## Verdict Logic

7. **Severity hierarchy:** `REJECT` > `PAUSE` > `ITERATE` > `INCOMPLETE` > `PASS`.
   - `REJECT`: any kill criterion triggered
   - `PAUSE`: monotonicity failure (mid > top) without kill trigger
   - `ITERATE`: one or more criteria failed but no kill/pause
   - `INCOMPLETE`: one or more inputs missing (None)
   - `PASS`: all 9 criteria pass and no kill/pause triggered

8. **CONDITIONAL not used.** The spec mentions CONDITIONAL but the Phase Gate
   Checklist uses binary PASS/ITERATE. We use `INCOMPLETE` for missing inputs
   instead, which is more precise.

## Artifact Output

9. **JSON artifact** contains: timestamp, dataset_hash, config_hash,
   code_commit_hash (placeholder), criteria table, kill checks, verdict,
   remediation actions.

10. **Markdown summary** is a human-readable gate decision document with
    criteria table, evidence references, and remediation guidance.

11. **Hash placeholders.** `dataset_hash` and `code_commit_hash` are
    string fields populated by the caller. This module does not compute
    hashes itself (it has no access to raw data files or git).

## Remediation

12. **Per-criterion remediation text.** Each failed criterion includes a
    pre-written remediation action from the Phase Gate Checklist fallback
    rules (e.g., "iterate features/ranges/universe per pre-registered
    fallback only").
