"""InferenceService — single-step action recommendation from the trained policy.

Given the latest 30×10 market window, return the greedy action (argmax Q) plus its
**confidence** (softmax over Q) and a short **feature-attribution** explanation —
the top channels by saliency, answering the brief's "which features contributed" (§8).
"""

from __future__ import annotations

import numpy as np


def action_names(cfg) -> list[str]:
    """Action names ordered by their config index, e.g. ['sell', 'hold', 'buy']."""
    pairs = vars(cfg.actions).items()
    return [name for name, _ in sorted(pairs, key=lambda kv: kv[1])]


def _softmax(values: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over a 1-D Q-vector → action probabilities."""
    exp = np.exp(values - values.max())
    return exp / exp.sum()


class InferenceService:
    """Latest window → Buy/Hold/Sell with confidence + a feature-attribution explanation."""

    def __init__(self, agent, names: list[str], feature_names: list[str]) -> None:
        self.agent = agent
        self.names = names
        self.feature_names = feature_names

    @classmethod
    def from_config(cls, agent, cfg) -> InferenceService:
        """Build with action names + feature names ordered by their config index."""
        return cls(agent, action_names(cfg), list(cfg.features.names))

    def recommend(self, state, top_k: int = 3) -> dict:
        """Greedy action + confidence (softmax) + top contributing features (saliency)."""
        q = self.agent.q_values(state)
        index = int(q.argmax())
        saliency = self.agent.q_saliency(state)
        order = sorted(range(len(self.feature_names)), key=lambda j: saliency[j], reverse=True)
        return {
            "action": self.names[index],
            "action_index": index,
            "q_values": [float(v) for v in q],
            "names": list(self.names),  # config-ordered labels (chart reads these, not a literal)
            "confidence": float(_softmax(q)[index]),
            "top_features": [self.feature_names[j] for j in order[:top_k]],
        }
