"""RolloutService — shared base for one env rollout (Template Method).

`TrainingService` and `BacktestService` both reset the env and loop
`act → step` until done; only *what they do per step* and *whether they act
greedily* differ. That shared skeleton lives here once; subclasses pass a
per-step callback to `_rollout`.
"""

from __future__ import annotations

from collections.abc import Callable


class RolloutService:
    """Template-Method base: reset the env and play one episode, calling ``on_step``."""

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    def _rollout(self, greedy: bool, on_step: Callable[..., None]) -> None:
        """Reset and play one episode; call ``on_step(state, action, reward, next, done, info)``."""
        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.act(state, greedy=greedy)
            next_state, reward, done, info = self.env.step(action)
            on_step(state, action, reward, next_state, done, info)
            state = next_state
