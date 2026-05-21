"""TrainingService — runs the DQN training loop over the trading environment.

One episode = one pass over the data slice: reset → (act → step → remember →
learn) until done. Epsilon decays once per episode. Returns a per-episode
history the GUI / analysis can plot. Shares the rollout skeleton with the
backtest via :class:`RolloutService`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tradedqn.services.rollout import RolloutService


class TrainingService(RolloutService):
    """Runs the DQN training loop (remember → learn → decay ε) over the env."""

    def train(self, episodes: int, on_episode: Callable[[dict], None] | None = None) -> list[dict]:
        """Train for ``episodes`` passes; return the per-episode history.

        ``on_episode`` (optional) is called with each episode's record as it
        completes, so a UI can show live progress instead of freezing until the
        whole run finishes.
        """
        history = []
        for index in range(int(episodes)):
            record = self._train_episode(index)
            history.append(record)
            if on_episode is not None:
                on_episode(record)
        return history

    def _train_episode(self, index: int) -> dict:
        """Run one training episode (rollout + ε-decay); return its summary record."""
        stats = {"reward": 0.0, "losses": [], "value": self.env.portfolio.initial_capital}

        def on_step(state, action, reward, next_state, done, info):
            """Remember the transition, learn, and accumulate episode stats."""
            self.agent.remember(state, action, reward, next_state, done)
            loss = self.agent.learn()
            if loss is not None:
                stats["losses"].append(loss)
            stats["reward"] += reward
            stats["value"] = info["value"]

        self._rollout(greedy=False, on_step=on_step)
        self.agent.decay_epsilon()
        return {
            "episode": index,
            "reward": stats["reward"],
            "final_value": stats["value"],
            "epsilon": self.agent.epsilon,
            "mean_loss": float(np.mean(stats["losses"])) if stats["losses"] else None,
        }
