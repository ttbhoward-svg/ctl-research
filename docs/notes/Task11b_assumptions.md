# Task 11b Assumptions — Negative Controls

## Scope

1. **Three pre-registered controls.** Per `pre_registration_v1.yaml`:
   - Randomized-label: shuffle TheoreticalR, refit, expect near-zero R² and no spread.
   - Lag-shift: shift features forward 1 bar, expect degradation or failure.
   - Placebo-feature: add random noise column, expect ~zero weight.

2. **Controls operate on the regression pipeline.** They call the same
   `prepare_features` + `ElasticNetCV` flow from `regression.py`, with
   controlled modifications to the data before fitting. This isolates the
   falsification to the scoring model, not the detection pipeline.

## Randomized-Label Control

3. **Multiple shuffles.** The target (TheoreticalR) is shuffled with different
   seeds. Default: 5 permutations. The mean R² and mean tercile spread across
   permutations are compared to baseline.

4. **Pass criteria.** Mean R² across shuffled runs must be < 0.02 (near zero).
   Mean absolute tercile spread must be < 0.3R. If the model finds meaningful
   signal in random labels, data leakage is likely.

## Lag-Shift Control

5. **Feature shift.** All 9 candidate features are shifted forward by
   `shift_bars` (default 1) within the DataFrame. This misaligns features with
   outcomes — the model sees features from trade N paired with the outcome from
   trade N-1. The first `shift_bars` rows are dropped (no valid pairing).

6. **Pass criteria.** The lag-shifted model's R² must be < 50% of baseline R².
   The tercile spread must degrade by at least 50% vs baseline. If lag-shifting
   features barely hurts performance, the model may be fitting noise or
   autocorrelation rather than genuine predictive relationships.

## Placebo-Feature Control

7. **Noise column.** A single random noise column (standard normal, fixed seed)
   is appended to the 9 candidate features before scaling. The model is refit
   with 10 features + cluster dummies.

8. **Pass criteria.** The absolute coefficient on the placebo feature must be
   < 0.05 (near zero after scaling). If the placebo gets a large coefficient,
   the model is not effectively regularizing noise.

9. **Feature cap exception.** The placebo control intentionally violates the
   9-feature cap for testing purposes only. This is documented and does not
   affect the production model.

## Overall Verdict

10. **All three must pass.** Per Gate 1 item 6 and kill criteria: "negative
    controls fail → REJECT." The overall verdict is PASS only if all three
    individual controls pass. Any single failure triggers FAIL.

## Implementation

11. **Self-contained module.** `negative_controls.py` accepts a DataFrame
    (assembled IS dataset) and pre-registration path. It runs baseline +
    controls internally and returns a structured report.

12. **Deterministic.** All random operations use explicit seeds for
    reproducibility. The baseline model is trained once and reused for
    comparison (not re-trained).

13. **No OOS data.** Controls are applied to IS data only. OOS validation
    is separate (Task 13).
