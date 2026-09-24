"""Pre-dispatch safety: budget reservation, rate limiting, cancellation.

DESIGN.md §12 requires *conservative reservation before dispatch* and
reconciliation afterwards, with the first-reached limit winning.  Because the
provider is the only authority on tokenisation, estimates are recorded
separately from server-reported usage and are deliberately over-generous, so the
guard trips earlier than the real bill would.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable


class BudgetExceeded(RuntimeError):
    """Raised *before* dispatch; no request should have left the process."""

    def __init__(self, limit: str, requested: float, available: float) -> None:
        super().__init__(f"budget limit {limit!r}: requested {requested} exceeds available {available}")
        self.limit = limit
        self.requested = requested
        self.available = available


class Cancelled(RuntimeError):
    pass


@dataclass
class Cancellation:
    """Cooperative cancellation: checked before every dispatch, never mid-billing."""

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled("run cancelled; no further requests dispatched")


@dataclass
class BudgetCaps:
    max_requests: int | None = None
    max_estimated_input_tokens: int | None = None
    max_cost_usd: float | None = None
    price_per_million_input_tokens: float = 0.042  # vendor quote, S8; not a guarantee

    def cost_for_tokens(self, tokens: int) -> float:
        return tokens * self.price_per_million_input_tokens / 1_000_000.0


@dataclass
class Reservation:
    tokens: int
    cost_usd: float
    ticket: int


class BudgetLedger:
    """Thread-safe reserve/commit accounting with hard pre-dispatch caps."""

    def __init__(self, caps: BudgetCaps, clock: Callable[[], float] = time.monotonic) -> None:
        self.caps = caps
        self.clock = clock
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self.reserved_tokens = 0
        self.reserved_cost = 0.0
        self.billed_tokens = 0  # server-reported, reconciled
        self.billed_cost = 0.0
        self.estimated_tokens = 0  # our over-generous pre-dispatch estimate
        self.n_requests = 0
        self.n_reservations = 0
        self.released_without_usage = 0
        self.denied: list[str] = []

    def cost_ceiling(self) -> float:
        """Conservative ceiling = billed + still-reserved (never under-counts)."""
        with self._lock:
            return self.billed_cost + self.reserved_cost

    def reserve(self, estimated_input_tokens: int) -> Reservation:
        tokens = int(estimated_input_tokens)
        cost = self.caps.cost_for_tokens(tokens)
        with self._condition:
            if self.caps.max_requests is not None and self.n_requests + 1 > self.caps.max_requests:
                self.denied.append("max_requests")
                raise BudgetExceeded("max_requests", 1, max(0, self.caps.max_requests - self.n_requests))
            if (
                self.caps.max_estimated_input_tokens is not None
                and self.reserved_tokens + tokens > self.caps.max_estimated_input_tokens
            ):
                self.denied.append("max_estimated_input_tokens")
                raise BudgetExceeded(
                    "max_estimated_input_tokens",
                    self.reserved_tokens + tokens,
                    self.caps.max_estimated_input_tokens - self.reserved_tokens,
                )
            if self.caps.max_cost_usd is not None and self.cost_ceiling_locked() + cost > self.caps.max_cost_usd:
                self.denied.append("max_cost_usd")
                raise BudgetExceeded(
                    "max_cost_usd",
                    self.cost_ceiling_locked() + cost,
                    self.caps.max_cost_usd - self.cost_ceiling_locked(),
                )
            self.n_requests += 1
            self.n_reservations += 1
            self.reserved_tokens += tokens
            self.reserved_cost += cost
            self.estimated_tokens += tokens
            return Reservation(tokens=tokens, cost_usd=cost, ticket=self.n_reservations)

    def cost_ceiling_locked(self) -> float:
        return self.billed_cost + self.reserved_cost

    def release(self, reservation: Reservation) -> None:
        """Undo a reservation that never dispatched (cancellation before send)."""
        with self._condition:
            self.reserved_tokens = max(0, self.reserved_tokens - reservation.tokens)
            self.reserved_cost = max(0.0, self.reserved_cost - reservation.cost_usd)
            self.estimated_tokens = max(0, self.estimated_tokens - reservation.tokens)
            self.n_requests = max(0, self.n_requests - 1)

    def reconcile(self, reservation: Reservation, reported_input_tokens: int | None) -> float:
        """Replace the estimate with server usage; unknown usage stays reserved.

        An error response with no usage must not be recorded as free: the
        conservative estimate remains *reserved* (and counts in the cost
        ceiling) and `released_without_usage` is incremented, so the ledger
        never under-states potential spend.  This intentionally errs towards
        stopping earlier than the real bill would.
        """
        with self._condition:
            if reported_input_tokens is None:
                self.released_without_usage += 1
                return 0.0
            self.reserved_tokens -= reservation.tokens
            self.reserved_cost -= reservation.cost_usd
            actual = int(reported_input_tokens)
            actual_cost = self.caps.cost_for_tokens(actual)
            self.billed_tokens += actual
            self.billed_cost += actual_cost
            return actual_cost

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "requests_reserved": self.n_requests,
                "reservations": self.n_reservations,
                "estimated_input_tokens": self.estimated_tokens,
                "reported_input_tokens": self.billed_tokens,
                "reported_cost_usd": round(self.billed_cost, 10),
                "reserved_cost_usd": round(self.reserved_cost, 10),
                "cost_ceiling_usd": round(self.billed_cost + self.reserved_cost, 10),
                "attempts_without_usage_report": self.released_without_usage,
                "denied_by_limit": sorted(set(self.denied)),
                "caps": {
                    "max_requests": self.caps.max_requests,
                    "max_estimated_input_tokens": self.caps.max_estimated_input_tokens,
                    "max_cost_usd": self.caps.max_cost_usd,
                    "price_per_million_input_tokens": self.caps.price_per_million_input_tokens,
                },
            }


class RateLimiter:
    """Token-bucket requests/min + tokens/sec with a shared Retry-After penalty.

    `sleeper` and `clock` are injectable so tests never actually wait.
    """

    def __init__(
        self,
        requests_per_minute: float | None = None,
        tokens_per_second: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = requests_per_minute
        self.tps = tokens_per_second
        self.clock = clock
        self.sleeper = sleeper
        self._lock = threading.Lock()
        self._request_times: list[float] = []
        self._token_times: list[tuple[float, int]] = []
        self._hold_until = 0.0
        self.sleep_events: list[dict[str, float]] = []

    def penalty(self, seconds: float) -> None:
        """Honour a server Retry-After by holding *all* threads, not just one."""
        with self._lock:
            self._hold_until = max(self._hold_until, self.clock() + max(0.0, seconds))

    def acquire(self, estimated_tokens: int, cancellation: Cancellation | None = None) -> float:
        """Block until the request may be dispatched; return total waited seconds."""
        waited = 0.0
        while True:
            with self._lock:
                now = self.clock()
                if self._hold_until > now:
                    delay = self._hold_until - now
                else:
                    delay = self._wait_needed(now, estimated_tokens)
                if delay <= 0:
                    self._request_times.append(now)
                    self._token_times.append((now, estimated_tokens))
                    self._prune(now)
                    return waited
            if cancellation is not None and cancellation.cancelled:
                raise Cancelled("run cancelled while waiting for rate limit")
            self.sleeper(delay)
            waited += delay
            self.sleep_events.append({"waited_seconds": delay, "at": self.clock()})

    def _prune(self, now: float) -> None:
        window = 60.0
        self._request_times = [t for t in self._request_times if now - t < window]
        self._token_times = [(t, n) for (t, n) in self._token_times if now - t < 1.0]

    def _wait_needed(self, now: float, estimated_tokens: int) -> float:
        self._prune(now)
        waits = [max(0.0, self._hold_until - now)]
        if self.rpm:
            recent = len(self._request_times)
            if recent + 1 > self.rpm:
                # Oldest-in-window must age out before we may dispatch.
                oldest = min(self._request_times[: recent + 1 - int(self.rpm)] or [now])
                waits.append(max(0.0, 60.0 - (now - oldest)))
        if self.tps:
            tokens_recent = sum(n for _, n in self._token_times)
            if tokens_recent + estimated_tokens > self.tps:
                oldest_token_time = min((t for t, _ in self._token_times), default=now)
                waits.append(max(0.0, 1.0 - (now - oldest_token_time)) or 1.0)
        return max(waits)
