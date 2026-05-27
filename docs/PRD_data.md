# PRD — Phase 1: Data Layer (fetch + §5 gatekeeper + cache)

**Phase:** 1 of 10 · **Status:** approved-to-build · **Owner decisions:** human

Covers `REQUIREMENTS.md` rows **B1, B2** and **C5** (config), and §5 of the
booklet. Lowest-judgment layer → built first to prove the TDD loop.

## Goal
Provide reproducible, offline-capable OHLCV data to the rest of the pipeline,
without ever hammering Yahoo Finance. Raw pulls are cached as **Parquet** (with a
per-ticker **CSV fallback**) under `data/raw` and committed/pinned to the repo, so
a grader can run the project fully offline and deterministically — no live network
call and no API key required.

## Scope (this phase only)
1. **Config loader** — `src/tradedqn/config.py`: parse `config/config.yaml`
   into an attribute-access namespace; resolve relative paths from the project
   root; fail loudly on a malformed root.
2. **Rate-limit gatekeeper** — `src/tradedqn/data/gatekeeper.py`
   (`RateLimitGatekeeper`): enforce a minimum interval between calls **and** a
   max-calls-per-rolling-window cap, with a bounded retry on transient failures.
   Injectable clock/sleep so tests use no real time. The public entry point is
   `execute(api_call, *args, **kwargs)` — it throttles, runs the call, retries up
   to `max_retries` times, and logs each attempt. The underlying permit primitive
   `acquire(wait=False)` raises `RateLimitError`; `acquire(wait=True)` blocks.
3. **DataClient** — `src/tradedqn/data/client.py`: `get_ohlcv(...)` is
   **cache-first** — returns the local **parquet** if present; on a miss it calls
   the live fetcher *through the gatekeeper*, validates OHLCV columns, writes the
   parquet cache (with a `{ticker}.csv` fallback if a live fetch fails), returns
   the frame. The fetcher (`yfinance`) is injected so tests never touch the network.

## Out of scope (later phases)
Feature engineering / indicators (Phase 2), windowing & the env (Phase 3).

## Public API (the SDK will call these)
```
load_config(path="config/config.yaml") -> Config
RateLimitGatekeeper(min_interval_seconds, max_calls_per_window, window_seconds,
                    max_retries=3, clock=time.monotonic, sleep=time.sleep)
    .execute(api_call, *args, **kwargs) -> object   # throttle + retry + log (primary)
    .acquire(wait=True) -> float                     # underlying permit; returns seconds waited
DataClient(cache_dir, gatekeeper=None, fetch_fn=_yf_download)
    .get_ohlcv(ticker, start, end, interval="1d", force_refresh=False) -> DataFrame
```

### Reproduce the exact raw pull (§4)
The `yf.download` call is **wrapped in the data class** (never pasted into training).
To reproduce the binding AAPL 2020-2023 raw frame from scratch — the same cache-first /
gatekept / CSV-fallback path the SDK uses:

```python
from tradedqn.data.client import DataClient
from tradedqn.data.gatekeeper import RateLimitGatekeeper

# cache-first: returns data/raw/AAPL_2020-01-01_2023-01-01.parquet if present,
# else one gatekept yf.download(interval="1d"), else the {ticker}.csv fallback.
client = DataClient("data/raw", RateLimitGatekeeper())
raw = client.get_ohlcv("AAPL", "2020-01-01", "2023-01-01")   # Open/High/Low/Close/Volume
print(len(raw), "rows"); print(raw.head())
```

The default fetcher (`tradedqn.data.client._yf_download`, injected as `fetch_fn` and stored as `self._fetch_fn`) is literally `yf.download(ticker, start=start,
end=end, interval=interval, progress=False)` with the single-ticker MultiIndex flattened —
identical to the brief's snippet, but behind the class so it is cached, rate-limited, and
testable (the fetcher is injectable, so tests never hit the network).

### Rate-limit config key mapping (§5.2)
The guidelines' example names a single `requests_per_minute`-style throttle; our
gatekeeper enforces **two** independent limits, so the `config.yaml`
`data.rate_limit` keys are named for what they control (ADR-001 keeps
behaviour-precise names over the example literals):

| Guideline example | This project (`data.rate_limit`) | Meaning |
|---|---|---|
| `requests_per_minute` | `max_calls_per_window` + `window_seconds` | ≤ N calls per rolling window (default 5 / 60s) |
| (implicit spacing) | `min_interval_seconds` | minimum gap between two calls (default 2.0s) |
| `max_retries` | `max_retries` | bounded retry on transient failure |

`requests_per_minute == max_calls_per_window` when `window_seconds == 60`. The
keys are read by attribute in `sdk.py::_gatekeeper` and `RateLimitGatekeeper.__init__`;
renaming them is a behaviour-equivalent cosmetic swap, deferred to avoid a
two-source-of-truth config.

## Acceptance criteria (tests must assert)
- Config: `window_size==30`, `features_count==10`, `gamma==0.95`,
  `actions.sell/hold/buy == 0/1/2`; nested attribute access; `to_dict()`
  round-trips; bad path raises; non-mapping root raises.
- Gatekeeper: min-interval throttle delays the 2nd immediate call by the right
  amount; window cap blocks the (N+1)th call; `wait=False` raises
  `RateLimitError`; `acquire` returns total time waited; uses injected clock/sleep
  (no real sleeping in tests).
- DataClient: cache miss → fetcher called once, parquet written, gatekeeper acquired;
  cache hit → fetcher **not** called; empty frame raises; missing OHLCV column
  raises; `force_refresh=True` re-fetches.

## Gates
≤150 code lines/file · TDD (RED→GREEN) · coverage ≥85% · ruff clean · no secrets.

## Risks / notes
- yfinance occasionally returns a multi-index column frame; `_validate` selects
  the canonical OHLCV columns and drops NaNs to normalise that.
- Network access is never exercised in unit tests (injected fetcher); a separate
  manual/integration check can hit the real API behind the gatekeeper.
