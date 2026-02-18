# Task B Assumptions — Symbol Mapping + Hash Tracking

1. **29-symbol universe.** The map covers the exact 29 symbols from
   `configs/symbols_phase1a.yaml`: 15 futures, 7 ETFs, 6 equities (tradable),
   1 equity (research_only = SBSW).

2. **Four providers.** TradeStation, Databento, Norgate, IBKR. Each CTL
   canonical symbol has one entry per provider.

3. **CTL canonical names are the keys.** Futures use `/` prefix (/ES),
   equities/ETFs have no prefix (XOM, XLE). This matches `universe.py`.

4. **Provider symbol conventions.**
   - **TradeStation**: `@` prefix for futures (@ES), `$` prefix for equities
     ($XOM), plain for ETFs (XLE).
   - **Databento**: Root symbol for futures (ES.FUT), plain for equities/ETFs.
     Exact stype/dataset parameters are deferred to provider implementation.
   - **Norgate**: `&` prefix for continuous adjusted futures (&ES), plain for
     equities/ETFs.
   - **IBKR**: Plain root for futures (ES), plain for equities/ETFs.

5. **Hash is file-level SHA-256.** Computed on the raw bytes of
   `configs/symbol_map_v1.yaml`. Any edit (even whitespace) changes the hash.

6. **Deterministic hash.** Because the YAML file is checked into git, its hash
   is stable across machines and sessions.

7. **Archive integration.** `configs/symbol_map_v1.yaml` is added to
   `archive.DEFAULT_ARTIFACTS` so the Phase 1a archive manifest includes it.

8. **Cross-validation.** `validate_symbol_map` checks that all 29 symbols are
   present and every entry has all 4 provider mappings. It does not import or
   depend on `universe.py` — both files are derived from the same spec.

9. **No API calls.** All provider symbols are conventional/documented; no
   runtime resolution against live APIs.

10. **Versioned file name.** `symbol_map_v1.yaml` — version suffix enables
    future revisions without breaking references.

11. **Reverse lookup is O(n).** Acceptable for 29 symbols. No index built.

12. **ETF symbols are identical across all providers.** TradeStation, Databento,
    Norgate, and IBKR all use the same ticker for US-listed ETFs.
