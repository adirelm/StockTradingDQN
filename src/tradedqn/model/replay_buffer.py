"""ReplayBuffer — experience replay for DQN.

Stores ``(state, action, reward, next_state, done)`` transitions and samples
uniform mini-batches, which breaks the temporal correlation of sequential
market steps. The sampling RNG is injected so batches are reproducible.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-capacity transition store with uniform mini-batch sampling."""

    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        self.capacity = int(capacity)
        self._buffer: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = rng or np.random.default_rng()

    def push(self, state, action, reward, next_state, done) -> None:
        """Append one (state, action, reward, next_state, done) transition."""
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        """Return a uniform mini-batch as stacked tensors (S, A, R, S2, D)."""
        idx = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[int(i)] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return (
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.stack(next_states)),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)
