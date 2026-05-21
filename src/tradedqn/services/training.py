"""TrainingService — runs the DQN training loop over the trading environment.

One episode = one pass over the data slice: reset → (act → step → remember →
learn) until done. Epsilon decays once per episode. Returns a per-episode
history the GUI / analysis can plot. Shares the rollout skeleton with the
backtest via :class:`RolloutService`.
"""

from __future__ import annotations

import numpy as np

from tradedqn.services.rollout import RolloutService


class TrainingService(RolloutService):
    def train(self, episodes: int) -> list[dict]:
        """Train for ``episodes`` passes; return the per-episode history."""
        return [self._train_episode(index) for index in range(int(episodes))]

    def _train_episode(self, index: int) -> dict:
        stats = {"reward": 0.0, "losses": [], "value": self.env.portfolio.initial_capital}

        def on_step(state, action, reward, next_state, done, info):
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
