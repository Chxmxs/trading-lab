# Strategy Base — ML Hooks (Phase 3.5)

This update adds **optional ML hooks** while staying compatible with purely rule-based strategies.

## Contract

**Time & Index**
- All time series must use a UTC `DatetimeIndex` (tz-aware).
- `equity` must be a `pd.Series` named `equity`.

**Trades schema (exact columns)**
["symbol","entry_time","exit_time","side","qty",
"entry_price","exit_price","fees","slippage",
"pnl","pnl_pct","mae","mfe","bars_held",
"entry_reason","exit_reason","stop_triggered"]


## Hooks

### `propose_entries(df) -> Series | DataFrame | None`
Purpose: emit **candidate entries**.

Accepted shapes:
- `pd.Series` (bool/int), UTC index → True/1 marks a candidate
- `pd.DataFrame`, UTC index with optional cols:
  `["is_candidate","meta_features","label","prob"]`

If not overridden, returns `None` (no ML usage).

### `filter_entries(entries, meta_probs=None) -> Series | DataFrame | None`
Purpose: meta-label filtering. Input is whatever `propose_entries` returned.
Return the filtered set; returning an empty frame/series is allowed.

Default: returns `entries` unchanged.

## Coexistence with `generate_signals`
- If **only** `generate_signals(...)` is overridden, ML hooks are **ignored**.
- If ML hooks are overridden but result in **zero** entries after filtering,
  the run remains **valid** (flat `equity`, empty `trades`).

## Logging
Every run logs counts:
- `proposed_entries=...`
- `filtered_entries=...`
- `placed_orders=...` (base class keeps at 0; execution happens elsewhere)

Logs are written to `.\logs\run_*.log` in UTC.

## Sanity: `scripts/sanity_ml_hooks.py`
Creates a 3-row dummy DF, a dummy strategy that:
- proposes one fake candidate
- filters it out (no trades)
Then prints:
- Equity series shape
- Trades columns
- Confirm log lines contain `proposed_entries` and `filtered_entries`.
