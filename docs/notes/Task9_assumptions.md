# Task 9 Assumptions — Pre-Registration Package

## Scope

1. **No model fitting.** This task produces the pre-registration document and
   machine-readable artifact. Actual regression (Task 10) is deferred.

2. **Feature naming follows implementation, not original spec.** The spec lists
   `COT_Commercial_Pctile_3yr` and `COT_Commercial_Zscore_1yr`, but the
   implemented features (Task 7) are `COT_20D_Delta` and `COT_ZScore_1Y`.
   The pre-registration uses the implemented names since those are what the
   code produces and the model will consume.

3. **Expected signs are directional hypotheses, not constraints.** The Elastic Net
   is free to zero out or assign negative coefficients. If a coefficient sign
   flips from expectation, it triggers investigation (not automatic rejection).
   Two or more sign flips triggers a PAUSE per the kill criteria.

4. **Dataset hash is a placeholder.** The SHA-256 hash field in the pre-registration
   is populated with the code commit hash at time of writing. The actual dataset
   hash will be recorded when the dataset is produced from real data (Task 8
   infrastructure is ready; real data has not yet been processed).

5. **Machine-readable artifact** is YAML to match the existing config convention
   (`model.yaml`, `phase1a.yaml`). It contains the same locked parameters as the
   markdown document, parseable by downstream code.

6. **Kill criteria are copied verbatim** from the locked Phase Gate Checklist v2.
   No modifications or additions.
