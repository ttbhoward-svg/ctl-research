# Task E Assumptions — Provider Connectivity & Setup Verification

1. **No network calls required.** The connectivity check validates local
   configuration (env vars, config file loading, sample file schema) without
   making API calls. If credentials are absent, the check marks the provider
   as SKIPPED with a clear message rather than FAIL.

2. **Environment variables.** Two optional env vars govern provider
   credentials:
   - `DATABENTO_API_KEY` — Databento REST API key.
   - `NORGATE_DATABASE_PATH` — Filesystem path to Norgate local database.
   If either is unset/empty, the corresponding provider is SKIPPED.

3. **Provider stubs.** Both `DatabentoProvider` and `NorgateProvider` are
   stubs that raise `NotImplementedError` on `get_ohlcv()`. The connectivity
   check does NOT call `get_ohlcv()`; it only verifies that the class can be
   instantiated and metadata is valid.

4. **Config loading.** The check verifies that `configs/symbol_map_v1.yaml`
   loads without error and passes validation (29 symbols, all providers
   mapped).

5. **Schema smoke check.** If sample `.csv.zst` files exist under
   `data/raw/databento/cutover_v1/outrights_only/`, the check reads one file
   per root symbol (ES, CL, PA) and verifies expected columns are present:
   `ts_event, open, high, low, close, volume, symbol`.

6. **Status values.** Each check reports one of three statuses:
   - `PASS` — check succeeded.
   - `FAIL` — check ran but found an error.
   - `SKIP` — check could not run (missing credentials, missing files).

7. **Exit codes.** The CLI exits 0 if all checks are PASS or SKIP, exits 1
   if any check is FAIL.

8. **No strategy/parity changes.** This task touches only infrastructure
   scripts, documentation, and tests. No changes to detection, simulation,
   reconciliation, or parity modules.

9. **Provider metadata validation.** The check verifies that each provider
   stub's `ProviderMeta` passes `meta.validate()` — i.e. session_type,
   roll_method, and close_type are within allowed enumerations.

10. **Sample file discovery.** The smoke check uses the first `.csv.zst`
    file found per root symbol directory. If no files exist, the check is
    SKIPPED for that symbol.

11. **Security guidance.** `docs/governance/data_provider_setup.md`
    documents that API keys must not be committed to the repo and should
    be set via environment variables or a `.env` file (which is
    gitignored).

12. **Deterministic output.** The CLI outputs a JSON-serialisable status
    dict suitable for CI integration. Tests verify deterministic output
    for identical inputs.
