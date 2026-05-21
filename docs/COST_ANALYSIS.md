# Cost & Resource Analysis (§11)

Two costs matter for a Vibe Coding project: the **runtime** cost of the model,
and the **development** cost of building it with an AI assistant. The second is
the bigger and more honest story.

> Disclaimer: this is a teaching tool, not a trading system. "Cost" here is
> compute + AI-tooling, not money risked in markets.

## 1. Runtime cost

The model is deliberately small — a Conv1D Dueling DQN over a 30×10 window.

| Item | Value | Notes |
|---|---|---|
| Network params | ~tens of thousands | Conv1D 10→32→64 + Dense(128) + two heads (3 / 1) |
| State tensor | 30 × 10 float32 = 1.2 KB | one window |
| Replay buffer | `replay_capacity` × (2 states + scalars) | 10 000 × ~2.5 KB ≈ 25 MB cap |
| Device | CPU / Apple MPS | the net is tiny; GPU is not required |
| Training | one episode ≈ one pass over the train slice (~1.7 k steps for 10 y daily) | see measured run below |
| Inference | one forward pass on a 30×10 window | sub-millisecond on CPU |

**Where it stops being free:** the cost scales with `window_size × features ×
replay_capacity × episodes`, and with the number of tickers if you train a
portfolio. A single-ticker daily model trains on a laptop CPU in minutes;
intraday data (×100 the rows) or many tickers is where a GPU / batched data
loader starts to matter. Pulling data is the other limit — Yahoo Finance
rate-limits, which is exactly why the §5 gatekeeper + CSV cache exist (fetch
once, train offline).

_Measured numbers from the README's real AAPL run are filled in by
`scripts/generate_results.py` → `results/analysis/backtest_metrics.json`._

## 2. Development cost (the Vibe Coding story)

Built with Claude Code over a **10-phase, PRD-first** workflow (see
[`PROMPTS.md`](PROMPTS.md)). The honest picture of AI-assisted development:

- **Token usage** is dominated by *context*, not the prompt you type: every
  turn re-sends the growing conversation + files in scope. Order-of-magnitude
  for a project this size (10 phases, ~700 lines of source + ~700 of tests +
  docs, with re-reads and gate runs): **single-digit millions of tokens**,
  reduced substantially by prompt caching. Not instrumented precisely — stated
  as an estimate, not a measurement.
- **The AI-rework tax is real.** AI writes code fast but its first answer takes
  the "obvious shape" and misses non-obvious traps. Concrete examples this
  project paid for in review, not luck: precomputing path-dependent portfolio
  features (leak), fitting the normalizer on the whole series (look-ahead),
  loading checkpoints without `weights_only` (code-exec risk). Catching these
  is *verification* work — and verification, not typing, is the cost centre of
  AI-assisted development.
- **The guardrails are cost controls.** The ≤150-line file cap and the 85%
  coverage gate exist precisely because AI code is cheap to write and expensive
  to verify; small modules + tight tests bound the verification cost. Holding
  100% coverage across 10 phases is what kept each phase's review cheap.

**Takeaway:** the runtime cost is negligible; the real cost is the human
attention spent reading and verifying generated code. The project's structure
(SDK facade, single-purpose modules, TDD) is an explicit bet to minimise it.
