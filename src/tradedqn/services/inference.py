"""InferenceService — single-step action recommendation from the trained policy.

Given the latest 30×10 market window, return the greedy action (argmax Q) with
its name and the raw Q-vector for display.
"""

from __future__ import annotations


def action_names(cfg) -> list[str]:
    """Action names ordered by their config index, e.g. ['sell', 'hold', 'buy']."""
    pairs = vars(cfg.actions).items()
    return [name for name, _ in sorted(pairs, key=lambda kv: kv[1])]


class InferenceService:
    def __init__(self, agent, names: list[str]) -> None:
        self.agent = agent
        self.names = names

    @classmethod
    def from_config(cls, agent, cfg) -> InferenceService:
        return cls(agent, action_names(cfg))

    def recommend(self, state) -> dict:
        """Return ``{action, action_index, q_values}`` for the given state."""
        q = self.agent.q_values(state)
        index = int(q.argmax())
        return {
            "action": self.names[index],
            "action_index": index,
            "q_values": [float(v) for v in q],
        }
