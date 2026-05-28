# Experiments summary (§11 deliverable)

Consolidated record of every controlled experiment — hypothesis → setup → result →
verdict. All are **seeded + offline-reproducible** off the committed cache; raw numbers
live in [`results/analysis/*.json`](../results/analysis) and the narrative + charts are in
the [README "Comparative experiments"](../README.md#comparative-experiments-4-cross-ticker--6-double-dqn--7-reward-design--9-seed-robustness)
section and [`notebooks/analysis.ipynb`](../notebooks/analysis.ipynb).

| # | Experiment | Reproduce |
|---|---|---|
| 1 | §4 cross-ticker (NVDA) | `uv run python scripts/compare_experiments.py --episodes 300` |
| 2 | §7 reward design (basic vs full) | `uv run python scripts/compare_experiments.py --episodes 300` |
| 3 | §6 Double-DQN vs vanilla | `uv run python scripts/ablations.py --episodes 300` |
| 4 | §9 seed robustness (5 seeds + CI) | `uv run python scripts/ablations.py --episodes 300` |
| 5 | §9 reward decomposition (waterfall) | `uv run python scripts/reward_waterfall.py --episodes 300` |
| 6 | §9 hyperparameter sweep (OAT, on validation) | `uv run python scripts/parameter_sweep.py` |

---

## 1 — §4 cross-ticker generalisation (AAPL vs NVDA)
- **Hypothesis (H):** if the DQN learned a *general* trading skill, the held-out verdict
  should not flip between symbols.
- **Setup:** identical pipeline/hyperparameters, only the ticker changes; held-out test slice.
- **Result** (`cross_ticker.json`): AAPL −17.5 % vs Buy & Hold −16.5 % (Sharpe −1.66);
  **NVDA +26.6 % vs +7.1 % (Sharpe +1.73)**.
- **Verdict:** **H rejected** — same method, *opposite* verdicts. A single-ticker backtest
  cannot establish skill; the NVDA "win" is as likely regime luck as edge.

## 2 — §7 reward design (basic ΔV vs full risk/cost-adjusted)
- **H:** the cost/slippage/Sharpe terms change *what* the agent optimises, not just the number.
- **Setup:** same data/agent; basic reward = ΔV only vs full `ΔV − C − S + λ·Sharpe`.
  *Honest caveat:* the basic variant also zeroes simulator costs, so it contrasts a
  basic-reward/cost-free regime against the full one.
- **Result** (`reward_comparison.json`): full −17.5 % (Sharpe −1.66, 21 trades, 20 % win);
  basic −17.6 % (Sharpe −1.64, 19 trades, 67 % win) — end-return parity, **different policy**.
- **Verdict:** **H supported** — reward design reshapes the trade pattern / win-rate / drawdown
  profile even when the end return is similar on this losing slice.

## 3 — §6 Double-DQN vs vanilla target
- **H:** decoupling action *selection* from *evaluation* (Double-DQN) curbs value
  over-estimation and should help out-of-sample.
- **Setup:** same trunk; only the Bellman target differs (`config.training.double_q`).
- **Result** (`double_q.json`): vanilla −17.5 % (Sharpe −1.66) vs **Double −23.0 % (Sharpe −2.06)**.
- **Verdict:** **H rejected here** — the right fix for the wrong disease. Over-estimation was
  not the binding failure; **overfitting** is, so the more conservative target is slightly worse,
  not better. Honest negative result.

## 4 — §9 seed robustness (is the headline a fluke?)
- **H:** the single-seed headline could be a lucky/unlucky draw.
- **Setup:** full train→test repeated over seeds `[42, 1, 7, 13, 100]`; mean ± std + 95 % t-CI.
- **Result** (`seed_variance.json`): return mean **−13.2 % ± 7.5 %**, 95 % CI **[−23.5 %, −2.8 %]**
  (entirely below 0); Sharpe mean **−1.22 ± 0.95**, 95 % CI **[−2.53, +0.10]** (crosses 0).
  Every seed loses; spread is 22 points; seed 42 sits on the pessimistic tail.
- **Verdict:** **robust verdict, fragile number** — "loses money, no risk-adjusted edge" holds
  inferentially across seeds; the exact percentage does not. At n=5 the Sharpe is not
  statistically distinguishable from 0 — an honest refusal to over-claim.

## 5 — §9 reward decomposition (waterfall)
- **H:** which reward term actually drives the learning signal?
- **Setup:** sum each component over the headline test backtest (the four bars sum to Σ reward).
- **Result** (`reward_waterfall.json`): ΔV −0.136, −cost −0.020, −slippage −0.020,
  **+λ·Sharpe −8.34**, net −8.52 — the **λ·Sharpe term is ~98 % of the reward magnitude**.
- **Verdict:** with λ=1.0 the agent overwhelmingly optimises *risk-adjustment*, not raw PnL —
  concrete evidence that the Sharpe-heavy reward is worth re-weighting (see Conclusions).

## 6 — §9 hyperparameter sweep (OAT, on validation)
- **H:** are the configured (deck-default) `lr=0.001`, `γ=0.95` the validation-optimal choices?
- **Setup:** one-at-a-time sweep; **validation** split only (no test peeking — §15 leakage guard).
- **Result** (`sweep.json`): the configured values are **not** the validation argmax — on validation
  Sharpe, `lr=0.005` (+0.05) edges `lr=0.001` (−0.10), and `γ=0.99` (+0.38) edges `γ=0.95` (−0.10).
  But every validation Sharpe is near zero and noisy at this scale, so the "optimum" isn't robust.
- **Verdict:** **H rejected (honestly).** We **kept the deck reference defaults** for fidelity to the
  course's `DQN-Trader-SDK` rather than chasing a noisy validation argmax. The sweep is reported as-is
  (it didn't favour our defaults), and crucially **selection happened on validation, never on test** —
  which is the §15 point that actually matters.

---

**Overall takeaway.** The deliverable is a correct, honestly-reported RL *system*: the headline
is a truthful "no demonstrable out-of-sample edge" on AAPL's 2022 drawdown, the verdict flips on
NVDA, and the experiments surface *why* (overfitting + a Sharpe-dominated reward) rather than
hiding it. **Past ≠ future.**
