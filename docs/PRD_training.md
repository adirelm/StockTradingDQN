# PRD — Phase 5: Training stack (replay + target net + DQN loop)

**Phase:** 5 of 10 · **Status:** approved-to-build · Covers requirement rows
**B9, B12, B13, B14, B15, B26**. This is where the agent actually learns.

## Goal
Train the Dueling DQN on the `TradingEnvironment` using experience replay and a
target network, with an ε-greedy schedule, the Bellman target + MSE loss, and
checkpointing.

## Modules
- `model/replay_buffer.py` — `ReplayBuffer(capacity, rng)`: store
  `(s, a, r, s', done)`; `sample(batch_size)` returns stacked tensors; seeded
  RNG so sampling is reproducible.
- `model/agent.py` — `DQNAgent`: owns **policy** + **target** `DuelingDQN`
  (target = frozen copy synced every `target_update_frequency` learn-steps);
  Adam optimizer; ε-greedy `act`; `learn` = sample → Bellman target via target
  net → MSE → backprop; `decay_epsilon`; `save/load` checkpoints with
  `weights_only=True` (tensors only, no arbitrary-code execution on load —
  carries the §7 security lesson from Assignment 1). `learn` is **gated by
  `train_frequency`** (config `4`): it counts env-steps and no-ops (returns
  `None`) until the replay holds `min_replay_before_train` (config `500`)
  transitions *and* the step index is a multiple of `train_frequency` —
  optimising every N env-steps is standard DQN and far faster than every step.
  `save` also persists a `metadata` dict (carries the §6 best-checkpoint
  record); `load` restores it onto `agent.metadata`.
- `services/training.py` — `TrainingService.train(episodes, on_episode=None)`:
  episode loop (reset → act → step → remember → learn), epsilon decay per
  episode, returns a history of per-episode reward / final value / epsilon /
  mean loss. The optional `on_episode(record)` callback fires as each episode
  completes so a UI (GUI / `generate_results.py`) can stream live progress
  instead of freezing until the whole run finishes. `TradingSDK.train` and
  `.compare` relay the same callback through.

## Overfitting guard — best-by-validation-Sharpe checkpoint (B16 / §6)
`scripts/generate_results.py` does not just keep the final policy. Every
`--validate-every` episodes (default `20`) it runs a **greedy** backtest on the
`validation` split (`sdk.backtest("validation")`, which does not perturb the
training RNG/weights) and, when the validation Sharpe improves on the running
best, saves that checkpoint to `config.paths.checkpoint` with its
`{episode, sharpe, return}` recorded as `metadata`. The chosen episode + metric
are surfaced as `backtest_metrics.json → best_checkpoint` (validation Sharpe
peaked at episode 59). The headline **test** metrics report the final
(episode-300) policy; the best-by-validation checkpoint is saved alongside it
with its metadata for reload/inspection. (This best-by-validation checkpoint does *not*
improve the held-out test result — validation 2021 and test 2022 are different
regimes — which is why the final-policy number stands as the honest headline;
see the Conclusions' regime-shift point.)

## Key formulas (deck)
- Bellman target: `y = r + γ · max_a' Q_target(s', a')·(1 − done)`
- Loss: `L(θ) = mean((y − Q_policy(s,a))²)` (MSE)
- ε-greedy: `random action w.p. ε, else argmax_a Q_policy(s,a)`; ε decays
  `max(epsilon_min, ε·epsilon_decay)` per episode.
- **Double-DQN ablation (`double_q` toggle, §6, config default `false`):** when
  enabled, the Bellman target decouples action *selection* from *evaluation* to
  curb max-operator overestimation — the **online** net selects the next action
  `a* = argmax_a' Q_policy(s', a')` while the **target** net evaluates it:
  `y = r + γ · Q_target(s', a*)·(1 − done)`. With `double_q: false` the vanilla
  target net both selects and evaluates (`max_a' Q_target`). Implemented in
  `DQNAgent.learn` (single `if self.double_q` branch; same MSE loss).

## Public API
```
ReplayBuffer(capacity, rng).push(s,a,r,s2,done); .sample(n) -> (S,A,R,S2,D); len()
DQNAgent(cfg, device="cpu")        # reads train_frequency + double_q from cfg.training
  .act(state, greedy=False) -> int
  .remember(s,a,r,s2,done); .learn() -> float | None; .decay_epsilon()
  .sync_target(); .save(path, metadata=None); .load(path)
TrainingService(env, agent).train(episodes, on_episode=None) -> list[dict]
```

## Acceptance criteria (tests assert)
- ReplayBuffer: push grows len to capacity then evicts (FIFO); `sample`
  returns correct shapes & dtypes (states float32 (n,window,feat), actions int64,
  rewards/dones float32); seeded rng → reproducible sample.
- Agent: `act` returns a valid action ∈ {0,1,2}; `greedy=True` is deterministic
  argmax; ε=0 → never random; `learn` returns `None` before `min_replay`, a
  finite float after; `sync_target` makes target params equal policy; `save`
  then `load` into a fresh agent reproduces identical greedy actions and ε/γ.
- TrainingService: `train(episodes=2)` on a tiny env returns history length 2;
  epsilon is lower at the end; no exceptions; replay grows.

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean. CPU device (small net),
seeded for determinism.
