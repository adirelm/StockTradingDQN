"""Tests for global RNG seeding (§6 — reproducible train-from-scratch)."""

import numpy as np
import torch

from tradedqn.seeding import seed_everything


class TestSeedEverything:
    def test_torch_init_is_reproducible(self):
        seed_everything(42)
        a = torch.rand(4)
        seed_everything(42)
        b = torch.rand(4)
        assert torch.equal(a, b)  # same seed → identical weight-init draws

    def test_numpy_is_reproducible(self):
        seed_everything(7)
        a = np.random.rand(4)
        seed_everything(7)
        b = np.random.rand(4)
        assert np.array_equal(a, b)

    def test_none_is_a_noop(self):
        seed_everything(None)  # must not raise and must not seed anything

    def test_two_agents_share_init_under_same_seed(self, tiny_cfg):
        from tradedqn.model.agent import DQNAgent

        a, b = DQNAgent(tiny_cfg), DQNAgent(tiny_cfg)  # both reseed to cfg.seed
        for pa, pb in zip(a.policy.parameters(), b.policy.parameters(), strict=True):
            assert torch.equal(pa, pb)  # reproducible init → fair Dueling-vs-plain ablation
