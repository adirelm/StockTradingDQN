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
| Training | one episode ≈ one pass over the train slice (~515 steps); headline run is 300 episodes | ~minutes on a laptop CPU (estimate, not instrumented) |
| Inference | one forward pass on a 30×10 window | sub-millisecond on CPU |

**Where it stops being free:** the cost scales with `window_size × features ×
replay_capacity × episodes`, and with the number of tickers if you train a
portfolio. A single-ticker daily model trains on a laptop CPU in minutes;
intraday data (×100 the rows) or many tickers is where a GPU / batched data
loader starts to matter. Pulling data is the other limit — Yahoo Finance
rate-limits, which is exactly why the §5 gatekeeper + parquet cache exist (fetch
once, train offline).

_The README's **result** metrics (return / Sharpe / drawdown / trades) come from
`scripts/generate_results.py` → `results/analysis/backtest_metrics.json`. Wall-clock
runtime is **not** instrumented — training takes ~minutes on a laptop CPU (an
estimate, not a committed measurement)._

## 2. Development cost (the Vibe Coding story)

Built with Claude Code over a **10-phase, PRD-first** workflow (see
[`PROMPTS.md`](PROMPTS.md)). The honest picture of AI-assisted development:

- **Token usage** is dominated by *context*, not the prompt you type: every
  turn re-sends the growing conversation + files in scope. Order-of-magnitude
  for a project this size (10 phases, ~700 lines of source + ~700 of tests +
  docs, with re-reads and gate runs): **single-digit millions of tokens**,
  reduced substantially by prompt caching. Not instrumented precisely — stated
  as an estimate, not a measurement.

### Token-cost breakdown (estimated)

⚠️ **Estimate, not a measurement** — the sessions were not token-instrumented;
figures are reconstructed from typical Claude Code session sizes and prompt
caching. Prices are Anthropic list prices (per million tokens, "MTok").

| Model | Role | Input (est.) | Output (est.) | $/MTok (in / out) | Est. cost |
|---|---|---:|---:|---:|---:|
| Claude Sonnet | bulk implementation, tests, edits | ~3.0 M | ~0.5 M | $3 / $15 | ~$16.5 |
| Claude Opus | architecture, reviews, audits | ~1.0 M | ~0.2 M | $15 / $75 | ~$30.0 |
| **Total** | | **~4.0 M** | **~0.7 M** | — | **~$45** (≈ **$25–70** band) |

With **prompt caching**, repeated context (the file tree, prior turns) bills at
~10% of the input rate, so the *effective* cost sits at the low end of the band.
Under a Claude **Max** subscription the marginal cost per session is **$0** within
rate limits — the table is the *if-billed-pay-as-you-go* equivalent.

**Optimization strategies used:** (1) **prompt caching** of the stable file tree
+ standards; (2) the **≤150-line file cap** bounds how much code enters context
per turn; (3) **parallel sub-agents** for the §-by-§ audit (each gets a fresh,
small context instead of one giant one); (4) reading only the needed file ranges,
not whole files.
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

## 3. Budget management (§11.2)

- **Budget envelope (human-decided).** A Claude Max subscription (~$200/mo);
  the pay-as-you-go equivalent for this project (~$25–70) stays well inside it.
- **Monitoring.** Claude Code reports per-session token usage; the cadence to
  watch is *tokens-per-merged-phase* (a phase that balloons signals scope creep
  or a file that's too big for its context budget — the 150-line cap is the
  early-warning here).
- **Overrun alert / threshold.** Rule of thumb: if a single phase exceeds
  ~1 MTok of *output*, stop and split it — that's the signal the task wasn't
  decomposed enough (the PRD should have been smaller). For pay-as-you-go, a
  hard monthly cap with an 80%-of-cap alert is the standard control.
- **Cost forecast for scale.** Development cost scales with *number of phases /
  features*, **not** with training scale. Adding a feature ≈ one more PRD-sized
  phase (~0.3–0.5 MTok). Runtime cost scales separately (more tickers/episodes →
  more compute, not more tokens) — see §1.

