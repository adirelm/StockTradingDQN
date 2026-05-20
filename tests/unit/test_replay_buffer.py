"""Tests for ReplayBuffer (B13 — store transitions, uniform mini-batch)."""

import numpy as np
import torch

from tradedqn.model.replay_buffer import ReplayBuffer

WINDOW, FEATURES = 5, 10


def push_n(buf, n):
    for i in range(n):
        s = np.full((WINDOW, FEATURES), float(i), dtype=np.float32)
        buf.push(s, i % 3, float(i), s, i == n - 1)


class TestStorage:
    def test_len_grows_then_caps_at_capacity(self):
        buf = ReplayBuffer(capacity=5)
        push_n(buf, 8)
        assert len(buf) == 5  # FIFO eviction at capacity

    def test_sample_shapes_and_dtypes(self):
        buf = ReplayBuffer(capacity=50, rng=np.random.default_rng(0))
        push_n(buf, 20)
        states, actions, rewards, next_states, dones = buf.sample(6)
        assert states.shape == (6, WINDOW, FEATURES)
        assert states.dtype == torch.float32
        assert actions.dtype == torch.int64
        assert rewards.dtype == torch.float32 and dones.dtype == torch.float32
        assert next_states.shape == (6, WINDOW, FEATURES)


class TestDeterminism:
    def test_seeded_sample_is_reproducible(self):
        a = ReplayBuffer(50, np.random.default_rng(123))
        b = ReplayBuffer(50, np.random.default_rng(123))
        push_n(a, 20)
        push_n(b, 20)
        sa, *_ = a.sample(5)
        sb, *_ = b.sample(5)
        assert torch.equal(sa, sb)
