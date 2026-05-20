# PRD — Phase 1: Data Layer (fetch + §5 gatekeeper + cache)

**Phase:** 1 of 10 · **Status:** approved-to-build · **Owner decisions:** human

Covers `REQUIREMENTS.md` rows **B1, B2** and **C5** (config), and §5 of the
booklet. Lowest-judgment layer → built first to prove the TDD loop.

## Goal
Provide reproducible, offline-capable OHLCV data to the rest of the pipeline,
without ever hammering Yahoo Finance.

## Scope (this phase only)
1. **Config loader** — `src/tradedqn/config.py`: parse `config/config.yaml`
   into an attribute-access namespace; resolve relative paths from the project
   root; fail loudly on a malformed root.
2. **Rate-limit gatekeeper** — `src/tradedqn/data/gatekeeper.py`
   (`RateLimitGatekeeper`): enforce a minimum interval between calls **and** a
   max-calls-per-rolling-window cap. Injectable clock/sleep so tests use no real
   time. `acquire(wait=False)` raises `RateLimitError`; `wait=True` blocks.
3. **DataClient** — `src/tradedqn/data/client.py`: `get_ohlcv(...)` is
   **cache-first** — returns the local CSV if present; on a miss it calls the
   live fetcher *through the gatekeeper*, validates OHLCV columns, writes the
   cache, returns the frame. The fetcher (`yfinance`) is injected so tests never
   touch the network.

## Out of scope (later phases)
Feature engineering / indicators (Phase 2), windowing & the env (Phase 3).

## Public API (the SDK will call these)
```
load_config(path="config/config.yaml") -> Config
RateLimitGatekeeper(min_interval_seconds, max_calls_per_window, window_seconds,
                    clock=time.monotonic, sleep=time.sleep).acquire(wait=True) -> float
DataClient(cache_dir, gatekeeper=None, fetch_fn=_yf_download)
    .get_ohlcv(ticker, start, end, interval="1d", force_refresh=False) -> DataFrame
```

## Acceptance criteria (tests must assert)
- Config: `window_size==30`, `features_count==10`, `gamma==0.95`,
  `actions.sell/hold/buy == 0/1/2`; nested attribute access; `to_dict()`
  round-trips; bad path raises; non-mapping root raises.
- Gatekeeper: min-interval throttle delays the 2nd immediate call by the right
  amount; window cap blocks the (N+1)th call; `wait=False` raises
  `RateLimitError`; `acquire` returns total time waited; uses injected clock/sleep
  (no real sleeping in tests).
- DataClient: cache miss → fetcher called once, CSV written, gatekeeper acquired;
  cache hit → fetcher **not** called; empty frame raises; missing OHLCV column
  raises; `force_refresh=True` re-fetches.

## Gates
≤150 code lines/file · TDD (RED→GREEN) · coverage ≥85% · ruff clean · no secrets.

## Risks / notes
- yfinance occasionally returns a multi-index column frame; `_validate` selects
  the canonical OHLCV columns and drops NaNs to normalise that.
- Network access is never exercised in unit tests (injected fetcher); a separate
  manual/integration check can hit the real API behind the gatekeeper.
