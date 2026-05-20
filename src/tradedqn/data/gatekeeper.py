"""§5 API gatekeeper — throttle outbound market-data calls.

Yahoo Finance is fine with one large historical pull but blocks rapid repeated
calls. ``RateLimitGatekeeper`` enforces two limits before any live fetch:
a minimum interval between calls, and a maximum number of calls per rolling
window. The clock and sleep functions are injected so tests run instantly with
no real waiting.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class RateLimitError(RuntimeError):
    """Raised when a call would exceed the limit and waiting is disabled."""


class RateLimitGatekeeper:
    """Sliding-window + minimum-interval rate limiter."""

    def __init__(
        self,
        min_interval_seconds: float = 2.0,
        max_calls_per_window: int = 5,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = float(min_interval_seconds)
        self.max_calls = int(max_calls_per_window)
        self.window = float(window_seconds)
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._last_call: float | None = None

    def _prune(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self.window:
            self._calls.popleft()

    def _throttle_min_interval(self, now: float, wait: bool) -> tuple[float, float]:
        if self._last_call is None:
            return 0.0, now
        gap = now - self._last_call
        if gap >= self.min_interval:
            return 0.0, now
        delay = self.min_interval - gap
        if not wait:
            raise RateLimitError(f"minimum interval of {self.min_interval}s has not elapsed")
        self._sleep(delay)
        return delay, self._clock()

    def _throttle_window(self, now: float, wait: bool) -> tuple[float, float]:
        self._prune(now)
        if len(self._calls) < self.max_calls:
            return 0.0, now
        delay = self.window - (now - self._calls[0])
        if not wait:
            raise RateLimitError(
                f"exceeded {self.max_calls} calls per {self.window}s window"
            )
        self._sleep(delay)
        now = self._clock()
        self._prune(now)
        return delay, now

    def acquire(self, wait: bool = True) -> float:
        """Block until a call is permitted; return total seconds waited.

        With ``wait=False`` raise :class:`RateLimitError` instead of sleeping.
        """
        now = self._clock()
        waited_a, now = self._throttle_min_interval(now, wait)
        waited_b, now = self._throttle_window(now, wait)
        self._calls.append(now)
        self._last_call = now
        return waited_a + waited_b
