"""TrainingService — runs the DQN training loop over the trading environment.

One episode = one pass over the data slice: reset → (act → step → remember →
learn) until done. Epsilon decays once per episode. Returns a per-episode
history the GUI / analysis can plot.
"""

from __future__ import annotations

import numpy as np


class TrainingService:
    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    def _run_episode(self, index: int) -> dict:
        state = self.env.reset()
        done = False
        total_reward = 0.0
        final_value = self.agent_initial_value()
        losses: list[float] = []
        while not done:
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.remember(state, action, reward, next_state, done)
            loss = self.agent.learn()
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            final_value = info["value"]
            state = next_state
        self.agent.decay_epsilon()
        return {
            "episode": index,
            "reward": total_reward,
            "final_value": final_value,
            "epsilon": self.agent.epsilon,
            "mean_loss": float(np.mean(losses)) if losses else None,
        }

    def agent_initial_value(self) -> float:
        return float(self.env.portfolio.initial_capital)

    def train(self, episodes: int) -> list[dict]:
        """Train for ``episodes`` passes; return the per-episode history."""
        return [self._run_episode(i) for i in range(int(episodes))]
