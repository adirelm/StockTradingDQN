"""Shared fixtures — a tiny config, a toy trading env, and a DQN agent.

The dims are deliberately small (window 5, conv 8→16, dense 16) so the
training tests run in well under a second.
"""

import pandas as pd
import pytest

from tradedqn.config import Config
from tradedqn.env.trading_env import TradingEnvironment
from tradedqn.model.agent import DQNAgent

WINDOW = 5


@pytest.fixture
def tiny_cfg():
    return Config(
        {
            "seed": 0,
            "features": {"window_size": WINDOW, "features_count": 10},
            "actions": {"sell": 0, "hold": 1, "buy": 2},
            "network": {"conv_channels": [8, 16], "kernel_size": 3, "dense_units": 16},
            "env": {
                "initial_capital": 1000.0,
                "transaction_cost": 0.001,
                "slippage": 0.0005,
                "risk_lambda": 0.1,
                "sharpe_window": 5,
            },
            "training": {
                "gamma": 0.95,
                "learning_rate": 0.01,
                "epsilon_start": 1.0,
                "epsilon_min": 0.1,
                "epsilon_decay": 0.5,
                "replay_capacity": 100,
                "batch_size": 4,
                "min_replay_before_train": 4,
                "train_frequency": 1,
                "target_update_frequency": 2,
            },
        }
    )


@pytest.fixture
def toy_env(tiny_cfg):
    n = 20
    feats = pd.DataFrame({f"f{j}": [float(i) for i in range(n)] for j in range(8)})
    prices = [10.0 + i for i in range(n)]  # strictly rising
    return TradingEnvironment(feats, prices, tiny_cfg)


@pytest.fixture
def dqn_agent(tiny_cfg):
    return DQNAgent(tiny_cfg)
