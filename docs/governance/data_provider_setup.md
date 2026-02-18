# Data Provider Setup Guide

This document describes how to configure CTL Research data providers for
local development and CI environments.

## Providers

| Provider     | Type        | Auth mechanism          | Status     |
|-------------|-------------|-------------------------|------------|
| Databento    | REST API    | API key (env var)       | Stub       |
| Norgate      | Local DB    | Database path (env var) | Stub       |
| TradeStation | Archived CSV| No auth (local files)   | Active     |

Both Databento and Norgate providers are stubs pending API integration.
The connectivity check script verifies that local configuration is correct
so that integration can proceed without friction.

## Required Environment Variables

### Databento

```bash
export DATABENTO_API_KEY="db-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

- Obtain from: https://databento.com/portal/keys
- Required for: live data fetches via `DatabentoProvider`
- If absent: provider checks are SKIPPED (not FAIL)

### Norgate

```bash
export NORGATE_DATABASE_PATH="/path/to/NorgateData"
```

- Obtain from: Norgate Data Updater installation path
- Required for: local database access via `NorgateProvider`
- If absent: provider checks are SKIPPED (not FAIL)
- macOS default: `~/Library/Application Support/Norgate Data/`
- Windows default: `C:\Users\<user>\AppData\Local\Norgate Data\`

## Security Guidance

1. **Never commit credentials.** API keys and database paths must not
   appear in source code, YAML configs, or commit messages.

2. **Use environment variables.** Set credentials via shell profile
   (`~/.zshrc`, `~/.bashrc`) or a `.env` file in the repo root.

3. **`.env` is gitignored.** The `.gitignore` already excludes `.env`
   files. Verify with: `git check-ignore .env`

4. **Rotate keys after exposure.** If a key is accidentally committed,
   rotate it immediately via the provider's portal and force-push a
   cleaned history (or use `git filter-branch`/BFG Repo Cleaner).

5. **CI secrets.** In CI pipelines, store credentials as encrypted
   secrets (e.g. GitHub Actions secrets) and inject them as env vars
   at runtime.

## Expected Folder Paths

### Raw Data

```
data/
├── raw/
│   ├── databento/
│   │   └── cutover_v1/
│   │       ├── glbx_extract/         # Full GLBX OHLCV-1D extracts
│   │       ├── xnas_extract/         # XNAS equity extracts
│   │       └── outrights_only/       # Per-symbol outright contracts
│   │           ├── ES/               # E-mini S&P 500
│   │           ├── CL/               # Crude Oil
│   │           └── PA/               # Palladium
│   └── tradestation/
│       └── cutover_v1/              # Archived TradeStation CSVs
```

### Processed Data

```
data/
├── processed/
│   ├── databento/
│   │   └── cutover_v1/
│   │       └── continuous/           # Continuous back-adjusted series
│   │           ├── ES_continuous.csv
│   │           ├── CL_continuous.csv
│   │           ├── PA_continuous.csv
│   │           └── roll_log.csv
│   └── *.parquet                    # Ingested OHLCV (from ingest_data.py)
```

### Config Files

```
configs/
├── symbol_map_v1.yaml    # 29-symbol cross-provider mapping
├── phase1a.yaml          # Phase 1a configuration
├── symbols_phase1a.yaml  # Universe definition
└── model.yaml            # Model parameters
```

## Running the Connectivity Check

```bash
# Full check (attempts all providers):
python scripts/check_provider_connectivity.py

# Verbose output:
python scripts/check_provider_connectivity.py --verbose

# JSON output (for CI):
python scripts/check_provider_connectivity.py --json
```

### Status Codes

| Status | Meaning |
|--------|---------|
| PASS   | Check succeeded |
| FAIL   | Check ran but found an error |
| SKIP   | Check could not run (missing credentials/files) |

**Exit code 0:** All checks PASS or SKIP.
**Exit code 1:** One or more checks FAIL.

## Smoke Check: Schema Validation

The connectivity script performs a lightweight schema check on sample
`.csv.zst` files in the outrights directory. It reads the first file
for each of ES, CL, PA and verifies expected columns:

```
ts_event, rtype, publisher_id, instrument_id, open, high, low, close, volume, symbol
```

This confirms that:
- `zstandard` decompression works (package installed)
- CSV column schema matches expectations
- File paths are accessible

No actual data analysis or price validation is performed during this check.
