"""§5 API gatekeeper — throttle outbound market-data calls.

Yahoo Finance is fine with one large historical pull but blocks rapid repeated
calls. ``RateLimitGatekeeper`` enforces two limits before any live fetch:
a minimum interval between calls, and a maximum number of calls per rolling
window. The clock and sleep functions are injected so tests run instantly with
no real waiting.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger("tradedqn.gatekeeper")


class RateLimitError(RuntimeError):
    """Raised when a call would exceed the limit and waiting is disabled."""


class RateLimitGatekeeper:
    """Sliding-window + minimum-interval rate limiter (§16 building block).

    Input:  a callable API call (via ``execute``) or a permit request (``acquire``).
    Output: the call's result (retried on failure) / the seconds waited.
    Setup:  ``min_interval_seconds``, ``max_calls_per_window``, ``window_seconds``,
            ``max_retries`` (from config); injectable ``clock``/``sleep`` for tests.
    """

    def __init__(
        self,
        min_interval_seconds: float = 2.0,
        max_calls_per_window: int = 5,
        window_seconds: float = 60.0,
        max_retries: int = 3,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = float(min_interval_seconds)
        self.max_calls = int(max_calls_per_window)
        self.window = float(window_seconds)
        self.max_retries = int(max_retries)
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

    def execute(self, api_call: Callable[..., object], *args, **kwargs) -> object:
        """Throttle, run ``api_call``, retry transient failures, and log every attempt.

        All external calls should go through here (§5): the gatekeeper enforces
        the rate limit *before* each attempt, retries up to ``max_retries`` on
        failure, and logs each attempt for monitoring. Overflow is handled by
        ``acquire`` blocking until a slot frees — a serialized FIFO for this
        single-threaded client (see the §15 concurrency note), not a dropped call.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            waited = self.acquire()
            logger.info("api call attempt %d/%d (throttled %.2fs)", attempt, self.max_retries, waited)
            try:
                result = api_call(*args, **kwargs)
            except Exception as error:  # transient (network/HTTP) — retry within the limit
                last_error = error
                logger.warning("api call attempt %d failed: %s", attempt, error)
                continue
            logger.info("api call succeeded on attempt %d", attempt)
            return result
        raise RuntimeError(f"api call failed after {self.max_retries} attempts") from last_error
