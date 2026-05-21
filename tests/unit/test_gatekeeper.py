"""Tests for the §5 RateLimitGatekeeper (no real time — injected clock/sleep)."""

import pytest

from tradedqn.data.gatekeeper import RateLimitError, RateLimitGatekeeper


class FakeClock:
    """Deterministic clock whose ``sleep`` advances the same virtual time."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make(clock, **kw):
    params = {"min_interval_seconds": 2.0, "max_calls_per_window": 3, "window_seconds": 60.0}
    params.update(kw)
    return RateLimitGatekeeper(clock=clock.time, sleep=clock.sleep, **params)


class TestMinInterval:
    def test_first_call_does_not_wait(self):
        clk = FakeClock()
        assert make(clk).acquire() == 0.0

    def test_immediate_second_call_waits_min_interval(self):
        clk = FakeClock()
        gk = make(clk)
        gk.acquire()
        assert gk.acquire() == pytest.approx(2.0)
        assert clk.sleeps == [2.0]

    def test_no_wait_raises_when_too_soon(self):
        clk = FakeClock()
        gk = make(clk)
        gk.acquire()
        with pytest.raises(RateLimitError, match="minimum interval"):
            gk.acquire(wait=False)


class TestWindowCap:
    def test_blocks_when_window_full(self):
        clk = FakeClock()
        gk = make(clk, min_interval_seconds=0.0, max_calls_per_window=3, window_seconds=60.0)
        for _ in range(3):
            gk.acquire()
        waited = gk.acquire()  # 4th call must wait out the window
        assert waited == pytest.approx(60.0)

    def test_no_wait_raises_when_window_full(self):
        clk = FakeClock()
        gk = make(clk, min_interval_seconds=0.0, max_calls_per_window=2, window_seconds=30.0)
        gk.acquire()
        gk.acquire()
        with pytest.raises(RateLimitError, match="per 30.0s window"):
            gk.acquire(wait=False)

    def test_window_prunes_old_calls(self):
        clk = FakeClock()
        gk = make(clk, min_interval_seconds=0.0, max_calls_per_window=2, window_seconds=10.0)
        gk.acquire()
        gk.acquire()
        clk.now = 11.0  # both prior calls are now outside the window
        assert gk.acquire() == 0.0


class TestExecute:
    def test_runs_call_through_throttle_and_returns(self):
        clk = FakeClock()
        gk = make(clk, min_interval_seconds=0.0, max_retries=3)
        assert gk.execute(lambda x: x * 2, 21) == 42

    def test_retries_transient_failure_then_succeeds(self):
        clk = FakeClock()
        gk = make(clk, min_interval_seconds=0.0, max_retries=3)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return "ok"

        assert gk.execute(flaky) == "ok"
        assert calls["n"] == 2

    def test_raises_after_max_retries(self):
        clk = FakeClock()
        gk = make(clk, min_interval_seconds=0.0, max_retries=2)

        def always_fails():
            raise ConnectionError("down")

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            gk.execute(always_fails)
