# PRD — Phase 5: Training stack (replay + target net + DQN loop)

**Phase:** 5 of 10 · **Status:** approved-to-build · Covers `REQUIREMENTS.md`
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
  carries the §7 security lesson from Assignment 1).
- `services/training.py` — `TrainingService.train(episodes)`: episode loop
  (reset → act → step → remember → learn), epsilon decay per episode, returns a
  history of per-episode reward / final value / epsilon / mean loss.

## Key formulas (deck)
- Bellman target: `y = r + γ · max_a' Q_target(s', a')·(1 − done)`
- Loss: `L(θ) = mean((y − Q_policy(s,a))²)` (MSE)
- ε-greedy: `random action w.p. ε, else argmax_a Q_policy(s,a)`; ε decays
  `max(epsilon_min, ε·epsilon_decay)` per episode.

## Public API
```
ReplayBuffer(capacity, rng).push(s,a,r,s2,done); .sample(n) -> (S,A,R,S2,D); len()
DQNAgent(cfg, device="cpu")
  .act(state, greedy=False) -> int
  .remember(s,a,r,s2,done); .learn() -> float | None; .decay_epsilon()
  .sync_target(); .save(path); .load(path)
TrainingService(env, agent).train(episodes) -> list[dict]
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
