"""Global RNG seeding for reproducible train-from-scratch runs (§6).

The agent's ε-greedy and the replay buffer's sampling already draw from an
injected, seeded NumPy ``Generator``; the network's *weight initialisation* draws
from the global Torch RNG. Seeding Python / NumPy / Torch — **and pinning Torch to
a single CPU thread + deterministic kernels** — makes a fresh training run
reproducible bit-for-bit *across machines and environments*, not just within one
process. (Multi-threaded CPU BLAS reductions accumulate floats in a
thread-schedule-dependent order, so without single-threading the same seed yields
slightly different weights run-to-run — enough to drift the backtest.)
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int | None) -> None:
    """Seed Python/NumPy/Torch + force deterministic single-threaded Torch (no-op if None)."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # deterministic kernels where available
    torch.set_num_threads(1)  # single-threaded → reproducible float-accumulation order across envs
