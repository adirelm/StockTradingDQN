"""Tests for RolloutService — the shared episode skeleton (Template Method, §4.2/§6.1)."""

from tradedqn.services.rollout import RolloutService


class _Env:
    """Two-step env: signals done on the second step."""

    def __init__(self) -> None:
        self.steps = 0

    def reset(self):
        return 0

    def step(self, action):
        self.steps += 1
        return self.steps, 1.0, self.steps >= 2, {"action": action}


class _Agent:
    def __init__(self) -> None:
        self.greedy_seen: list[bool] = []

    def act(self, state, greedy=False):
        self.greedy_seen.append(greedy)
        return 1


def test_rollout_calls_on_step_each_step_until_done():
    env, agent = _Env(), _Agent()
    seen: list[tuple] = []
    RolloutService(env, agent)._rollout(greedy=True, on_step=lambda *a: seen.append(a))
    assert len(seen) == 2                      # two steps, then the env reports done
    assert agent.greedy_seen == [True, True]   # the greedy flag is forwarded to the agent
