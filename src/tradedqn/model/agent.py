"""DQNAgent — policy + target Dueling DQN with ε-greedy action and DQN learning.

The policy net is trained by gradient descent; the target net is a frozen copy
synced every ``target_update_frequency`` learn-steps to give a stable Bellman
target. Checkpoints save/load tensors only (``weights_only=True``) — no
arbitrary-code execution on load (§7 security).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from tradedqn.model.network import DuelingDQN
from tradedqn.model.replay_buffer import ReplayBuffer
from tradedqn.seeding import seed_everything


class DQNAgent:
    """Policy + target Dueling-DQN with ε-greedy action selection and DQN learning."""

    def __init__(self, cfg, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.n_actions = len(vars(cfg.actions))
        self.gamma = float(cfg.training.gamma)
        self.batch_size = int(cfg.training.batch_size)
        self.min_replay = int(cfg.training.min_replay_before_train)
        self.target_update_frequency = int(cfg.training.target_update_frequency)
        self.train_frequency = int(getattr(cfg.training, "train_frequency", 1))
        self.double_q = bool(getattr(cfg.training, "double_q", False))  # Double-DQN target (§6 extension)
        self.epsilon = float(cfg.training.epsilon_start)
        self.epsilon_min = float(cfg.training.epsilon_min)
        self.epsilon_decay = float(cfg.training.epsilon_decay)
        seed = getattr(cfg, "seed", None)
        seed_everything(seed)  # seed Torch *before* weight init so a fresh run reproduces
        self._rng = np.random.default_rng(seed)
        self.policy = DuelingDQN.from_config(cfg).to(self.device)
        self.target = DuelingDQN.from_config(cfg).to(self.device)
        self.sync_target()
        self.target.train(False)  # target net is never trained directly (eval mode)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(cfg.training.learning_rate))
        self.replay = ReplayBuffer(int(cfg.training.replay_capacity), self._rng)
        self._learn_steps = 0
        self._env_steps = 0

    def _batch(self, state) -> torch.Tensor:
        """Wrap a single state as a (1, …) float32 tensor on the device."""
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def act(self, state, greedy: bool = False) -> int:
        """ε-greedy action: random with prob ε (unless ``greedy``), else argmax Q."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        with torch.no_grad():
            q_values = self.policy(self._batch(state))
        return int(torch.argmax(q_values, dim=1).item())

    def q_values(self, state) -> np.ndarray:
        """Raw Q-vector for a single state (used by inference)."""
        with torch.no_grad():
            return self.policy(self._batch(state)).squeeze(0).cpu().numpy()

    def q_saliency(self, state) -> np.ndarray:
        """Per-feature attribution: |∂ maxₐ Q / ∂ input| summed over the time axis.

        A saliency map answering "which features drove this decision" (§8): the
        gradient of the chosen action's Q-value w.r.t. each input channel.
        """
        inputs = self._batch(state).requires_grad_(True)
        self.policy(inputs).max().backward()
        grad = inputs.grad.detach().abs().squeeze(0)  # (window, features)
        self.policy.zero_grad(set_to_none=True)        # don't perturb training grads
        return grad.sum(dim=0).cpu().numpy()           # (features,)

    def remember(self, state, action, reward, next_state, done) -> None:
        """Store a transition (s, a, r, s', done) in the replay buffer."""
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """Optimise on a replay mini-batch every ``train_frequency`` steps.

        Returns ``None`` until the replay is warmed up and on the steps that are
        skipped by the train-frequency gate (learning every env step is wasteful;
        every few steps is standard DQN and far faster).
        """
        self._env_steps += 1
        if len(self.replay) < self.min_replay or self._env_steps % self.train_frequency != 0:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states, next_states = states.to(self.device), next_states.to(self.device)
        actions, rewards, dones = actions.to(self.device), rewards.to(self.device), dones.to(self.device)
        q_sa = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_q:  # online net SELECTS the next action, target net EVALUATES it
                next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
                max_next = self.target(next_states).gather(1, next_actions).squeeze(1)
            else:  # vanilla DQN: target net both selects and evaluates (max)
                max_next = self.target(next_states).max(dim=1).values
            target = rewards + self.gamma * max_next * (1.0 - dones)
        loss = nn.functional.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._learn_steps += 1
        if self._learn_steps % self.target_update_frequency == 0:
            self.sync_target()
        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decay ε one step toward its floor (called once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sync_target(self) -> None:
        """Copy the policy weights into the frozen target network (θ⁻ ← θ)."""
        self.target.load_state_dict(self.policy.state_dict())

    def save(self, path: str, metadata: dict | None = None) -> None:
        """Checkpoint policy+target weights, ε/γ, and run ``metadata`` to ``path`` (tensors only).

        ``metadata`` carries the §6 reproducibility record (e.g. episode + validation
        metric of a best-by-validation checkpoint).
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target.state_dict(),
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "metadata": dict(metadata or {}),
            },
            target_path,
        )

    def load(self, path: str) -> None:
        """Restore weights, ε/γ, and ``metadata`` from a checkpoint (``weights_only=True``)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.metadata = checkpoint.get("metadata", {})
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint["target"])
        self.epsilon = float(checkpoint["epsilon"])
        self.gamma = float(checkpoint["gamma"])
