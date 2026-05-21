"""Tests for the Dueling Conv1D DQN (B8 — shape, aggregation, determinism)."""

import torch

from tradedqn.config import load_config
from tradedqn.model.network import DuelingDQN

WINDOW, FEATURES, ACTIONS = 30, 10, 3


def make_net():
    return DuelingDQN(WINDOW, FEATURES, ACTIONS, conv_channels=[32, 64],
                      kernel_size=3, dense_units=128)


class TestForwardShape:
    def test_batch_output_is_n_actions(self):
        out = make_net()(torch.randn(8, WINDOW, FEATURES))
        assert out.shape == (8, ACTIONS)

    def test_single_sample(self):
        out = make_net()(torch.randn(1, WINDOW, FEATURES))
        assert out.shape == (1, ACTIONS)

    def test_output_requires_grad(self):
        out = make_net()(torch.randn(2, WINDOW, FEATURES))
        assert out.requires_grad
        assert len(list(make_net().parameters())) > 0


class TestDuelingAggregation:
    def test_q_equals_value_plus_centered_advantage(self):
        net = make_net()
        x = torch.randn(4, WINDOW, FEATURES)
        value, advantage = net.value_advantage(x)
        expected = value + advantage - advantage.mean(dim=1, keepdim=True)
        assert torch.allclose(net(x), expected, atol=1e-6)

    def test_advantage_contribution_is_mean_zero(self):
        net = make_net()
        x = torch.randn(4, WINDOW, FEATURES)
        value, _ = net.value_advantage(x)
        # (Q − V) is the mean-centered advantage → mean over actions ≈ 0
        residual = net(x) - value
        assert torch.allclose(residual.mean(dim=1), torch.zeros(4), atol=1e-6)


class TestDeterminismAndConfig:
    def test_same_seed_same_output(self):
        x = torch.randn(2, WINDOW, FEATURES)
        torch.manual_seed(0)
        out_a = make_net()(x)
        torch.manual_seed(0)
        out_b = make_net()(x)
        assert torch.allclose(out_a, out_b)

    def test_from_config_dims(self):
        cfg = load_config("config/config.yaml")
        net = DuelingDQN.from_config(cfg)
        out = net(torch.randn(1, cfg.features.window_size, cfg.features.features_count))
        assert out.shape[1] == len(vars(cfg.actions)) == 3
        assert net.conv[0].in_channels == cfg.features.features_count


class TestAblation:
    def test_plain_mode_returns_advantage_as_q(self):
        net = DuelingDQN(WINDOW, FEATURES, ACTIONS, [32, 64], 3, 128, dueling=False)
        x = torch.randn(2, WINDOW, FEATURES)
        _, advantage = net.value_advantage(x)
        assert torch.allclose(net(x), advantage, atol=1e-6)  # plain DQN: Q = A, no V+A−mean

    def test_from_config_respects_dueling_flag(self):
        cfg = load_config("config/config.yaml")
        cfg.network.dueling = False
        assert DuelingDQN.from_config(cfg).dueling is False
