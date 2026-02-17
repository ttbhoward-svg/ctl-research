# Task 3 (Trade Simulator + Chart Inspector): Assumptions & Ambiguities

---

## Stop Fill Timing

- **Spec §4.3 vs §4.2 wording conflict.** §4.2 shows `EXIT at Close[bar]` in a comment, but §4.3 explicitly overrides this: "Exit occurs at the NEXT bar's open, plus slippage." We implement §4.3 (next-bar open). The §4.2 comment is treated as a placeholder that §4.3 supersedes.
- **Gap through stop.** If bar N closes below stop, the exit is at Open[bar N+1]. If bar N+1 gaps down significantly, exit price can be much worse than the stop level. This is correct and intentional — it models realistic slippage on gap opens. We do NOT clamp exit price to the stop level.
- **Stop breach on last bar of data.** If the final bar breaches the stop, the next-bar open doesn't exist. We mark the trade as `exit_reason='stop'` with exit price = Close of the breaching bar. This is flagged with `exit_on_last_bar=True` for filtering.

## TP Fill

- **TP fill at level exactly.** Spec §Execution Conventions: "TP fill: At TP level exactly (resting limit order assumption)." If High >= TP_level, fill at TP_level (not at High). This is standard for resting limit orders.
- **Multi-TP same bar.** Spec §5.4: If High exceeds multiple TPs on the same bar, exit at the HIGHEST qualifying TP. All lower TPs are considered simultaneously hit. We implement this as: scan TP3 first, then TP2, then TP1 — the first match wins and covers all remaining position.

## TP/Stop Same-Bar Collision

- **Default: TP wins.** Spec §4.4 and §Execution Conventions are consistent. If `High[bar] >= TP_level AND Close[bar] < StopPrice` on the same bar, TP takes priority. Rationale: resting limit order at TP would have filled intrabar before the close breached the stop.
- **Configurable.** We expose `collision_rule='tp_wins'` (default) and `collision_rule='stop_wins'`.
- **Collision flag.** `same_bar_collision` is logged as a boolean on the trade record for monitoring (spec says flag if >5% of trades).

## Theoretical R Calculation

- **Partial exits in TheoreticalR.** Spec §7: "always compute as if the position can be split into perfect thirds." For TheoreticalR:
  - If stopped before TP1: TheoreticalR = full stop R (all three thirds stopped).
  - If TP1 hit, then stopped before TP2: 1/3 at TP1 R + 2/3 at stop R.
  - If TP1+TP2 hit, then stopped before TP3: 1/3 at TP1 R + 1/3 at TP2 R + 1/3 at stop R.
  - If all three TPs hit: 1/3 at each TP R.
- **Stop R for remaining position.** After a partial TP exit, if the remainder is stopped out, the stop exit uses the same next-bar-open fill as a regular stop. The partial R for the stopped portion uses that fill price.

## MFE / MAE

- **Window.** MFE and MAE are computed over the life of the trade, from the entry bar through the final exit bar (inclusive). Entry bar itself is included.
- **R-normalisation.** MFE_R and MAE_R are in R-units: `(price - entry) / risk_per_unit`.

## Day1Fail

- **Spec §7: `Day1Fail = (Low[entry_bar] < StopPrice)`.** The entry bar is the bar whose Open we entered at, not the confirmation bar. Trade still runs — this is a research flag only.

## Hold Days

- **Not explicitly in the spec.** We compute `hold_bars = exit_bar_idx - entry_bar_idx` as a useful research column. This counts calendar trading bars, not calendar days.

## Chart Inspector

- **Scope in Task 3.** We build a data payload per trade (entry/exit markers, TP/stop levels, EMA overlay data) and a Plotly chart generation function. The payload is a dict that can be serialised to JSON. The Plotly function produces a standalone HTML file.
- **No dashboard integration yet.** That's Phase 1a dashboard work (Task 3 of the tracker says "Chart Inspector" but not Streamlit wiring).

## Trade Outcome Classification

- **Win:** At least TP1 hit (regardless of whether remainder was stopped).
- **Loss:** Stopped out before TP1.
- **Expired:** Never explicitly defined in spec for simulator context. We do NOT generate "Expired" outcomes from the simulator — that label applies only to triggers that never confirmed (handled by the detector). All confirmed trades resolve as Win or Loss.
