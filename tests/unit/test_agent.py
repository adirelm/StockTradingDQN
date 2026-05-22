"""Tests for DQNAgent (B9/B12/B14/B26 — ε-greedy, learn, target sync, checkpoint)."""

import numpy as np
import torch

WINDOW, FEATURES = 5, 10


def a_state(value=0.5):
    return np.full((WINDOW, FEATURES), value, dtype=np.float32)


def fill_replay(agent, n=8):
    for i in range(n):
        s = a_state(i / 10.0)
        agent.remember(s, i % 3, float(i), s, False)


class TestAct:
    def test_action_is_valid(self, dqn_agent):
        assert dqn_agent.act(a_state()) in (0, 1, 2)

    def test_greedy_is_deterministic_argmax(self, dqn_agent):
        s = a_state()
        assert dqn_agent.act(s, greedy=True) == dqn_agent.act(s, greedy=True)

    def test_zero_epsilon_never_random(self, dqn_agent):
        dqn_agent.epsilon = 0.0
        s = a_state()
        assert all(dqn_agent.act(s) == dqn_agent.act(s, greedy=True) for _ in range(10))


class TestSaliency:
    def test_per_feature_importance_shape_and_sign(self, dqn_agent):
        sal = dqn_agent.q_saliency(a_state())
        assert sal.shape == (FEATURES,)                 # one importance score per input channel
        assert (sal >= 0).all() and np.isfinite(sal).all()

    def test_saliency_leaves_no_grad_on_policy(self, dqn_agent):
        dqn_agent.q_saliency(a_state())
        assert all(p.grad is None for p in dqn_agent.policy.parameters())  # training grads untouched


class TestLearn:
    def test_learn_none_before_warmup(self, dqn_agent):
        dqn_agent.remember(a_state(), 1, 0.0, a_state(), False)
        assert dqn_agent.learn() is None

    def test_learn_returns_finite_loss_after_warmup(self, dqn_agent):
        fill_replay(dqn_agent, 8)
        loss = dqn_agent.learn()
        assert isinstance(loss, float) and np.isfinite(loss)

    def test_train_frequency_gates_optimisation(self, tiny_cfg):
        from tradedqn.model.agent import DQNAgent

        tiny_cfg.training.train_frequency = 3  # optimise only every 3rd step
        agent = DQNAgent(tiny_cfg)
        fill_replay(agent, 8)
        outcomes = [agent.learn() is not None for _ in range(6)]
        assert outcomes == [False, False, True, False, False, True]

    def test_epsilon_decays_with_floor(self, dqn_agent):
        dqn_agent.epsilon = 1.0
        dqn_agent.decay_epsilon()
        assert dqn_agent.epsilon == 0.5  # decay 0.5
        for _ in range(20):
            dqn_agent.decay_epsilon()
        assert dqn_agent.epsilon == dqn_agent.epsilon_min


class TestTargetAndCheckpoint:
    def test_sync_makes_target_equal_policy(self, dqn_agent):
        fill_replay(dqn_agent, 8)
        dqn_agent.learn()  # mutates policy
        dqn_agent.sync_target()
        for p, t in zip(dqn_agent.policy.parameters(), dqn_agent.target.parameters(), strict=True):
            assert torch.allclose(p, t)

    def test_save_load_reproduces(self, dqn_agent, tiny_cfg, tmp_path):
        from tradedqn.model.agent import DQNAgent

        fill_replay(dqn_agent, 8)
        dqn_agent.learn()
        dqn_agent.epsilon = 0.42
        path = str(tmp_path / "ckpt.pt")
        dqn_agent.save(path)
        fresh = DQNAgent(tiny_cfg)
        fresh.load(path)
        s = a_state(0.3)
        assert fresh.act(s, greedy=True) == dqn_agent.act(s, greedy=True)
        assert fresh.epsilon == 0.42 and fresh.gamma == dqn_agent.gamma

    def test_checkpoint_carries_metadata(self, dqn_agent, tiny_cfg, tmp_path):
        from tradedqn.model.agent import DQNAgent

        path = str(tmp_path / "best.pt")
        dqn_agent.save(path, metadata={"episode": 42, "val_sharpe": 1.5})  # §6 best-by-val record
        fresh = DQNAgent(tiny_cfg)
        fresh.load(path)
        assert fresh.metadata == {"episode": 42, "val_sharpe": 1.5}
