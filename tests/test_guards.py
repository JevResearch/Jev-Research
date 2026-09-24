"""Guard tests: budget reservation, caps, cancellation, rate limiting."""

import threading

from jev_observatory.guards import (
    BudgetCaps,
    BudgetExceeded,
    BudgetLedger,
    Cancellation,
    RateLimiter,
)

PRICE = 0.042  # USD per million input tokens (vendor quote)


def _ledger(**kwargs):
    caps = BudgetCaps(price_per_million_input_tokens=PRICE, **kwargs)
    return BudgetLedger(caps, clock=lambda: 0.0)


def test_reservation_and_reconcile():
    ledger = _ledger()
    reservation = ledger.reserve(1000)
    assert ledger.reserved_tokens == 1000
    # Reported usage replaces the estimate.
    ledger.reconcile(reservation, 400)
    assert ledger.reserved_tokens == 0
    assert ledger.billed_tokens == 400
    assert ledger.billed_cost == 400 * PRICE / 1_000_000
    snap = ledger.snapshot()
    assert snap["estimated_input_tokens"] == 1000  # estimate recorded, not replaced
    assert snap["reported_input_tokens"] == 400


def test_error_without_usage_stays_conservative():
    """An error response with no usage must not be recorded as free."""
    ledger = _ledger()
    reservation = ledger.reserve(1000)
    ledger.reconcile(reservation, None)
    snap = ledger.snapshot()
    assert snap["reported_input_tokens"] == 0
    assert snap["attempts_without_usage_report"] == 1
    # The estimate remains reserved: the ledger never under-states spend.
    assert ledger.cost_ceiling() > 0
    assert ledger.reserved_tokens == 1000


def test_request_cap_denies_before_dispatch():
    ledger = _ledger(max_requests=2)
    ledger.reserve(100)
    ledger.reserve(100)
    try:
        ledger.reserve(100)
        raised = False
    except BudgetExceeded as exc:
        assert exc.limit == "max_requests"
        assert ledger.snapshot()["requests_reserved"] == 2  # nothing extra went out
        return
    raise AssertionError("expected BudgetExceeded")


def test_cost_cap_denies_before_dispatch():
    ledger = _ledger(max_cost_usd=100 * PRICE / 1_000_000)
    ledger.reserve(100)
    try:
        ledger.reserve(1)  # would exceed the tiny cap
    except BudgetExceeded:
        return
    raise AssertionError("cost cap not enforced")


def test_estimated_token_cap():
    ledger = _ledger(max_estimated_input_tokens=1500)
    ledger.reserve(1000)
    ledger.reserve(500)
    try:
        ledger.reserve(1)
    except BudgetExceeded:
        return
    raise AssertionError("token cap not enforced")


def test_release_undoes_reservation():
    ledger = _ledger(max_estimated_input_tokens=1000)
    reservation = ledger.reserve(1000)
    ledger.release(reservation)
    assert ledger.reserved_tokens == 0
    assert ledger.n_requests == 0
    ledger.reserve(1000)  # capacity restored


def test_reservation_is_thread_safe_under_concurrency():
    ledger = _ledger(max_estimated_input_tokens=10_000)
    ok = []
    denied = []

    def worker(n):
        for _ in range(n):
            try:
                ledger.reserve(500)
                ok.append(1)
            except BudgetExceeded:
                denied.append(1)

    threads = [threading.Thread(target=worker, args=(20,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 10_000 / 500 == 20 reservations fit exactly; the rest must be denied.
    assert len(ok) == 20, f"oversubscribed: {len(ok)} reservations went through"
    assert ledger.reserved_tokens == 10_000
    assert len(denied) == 140


def test_cancellation_flag():
    cancellation = Cancellation()
    assert not cancellation.cancelled
    cancellation.cancel()
    assert cancellation.cancelled
    try:
        cancellation.raise_if_cancelled()
        raise AssertionError("should have raised")
    except RuntimeError:
        pass


class _FakeClock:
    """Clock that advances only when the limiter sleeps, like a paused runtime.

    Every sleeper call advances the clock, so a `wait needed` loop terminates.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleeper(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_rpm_backoff_with_fake_clock():
    clock = _FakeClock()
    limiter = RateLimiter(requests_per_minute=2, clock=clock, sleeper=clock.sleeper)
    limiter.acquire(100)
    clock.now = 1.0
    limiter.acquire(100)
    clock.now = 2.0
    waited = limiter.acquire(100)  # third request inside the window must wait
    assert waited > 0 and clock.sleeps
    limiter.acquire(100)  # waited into a fresh slot; window now holds recent requests
    assert limiter.acquire(100) > 0  # 5th inside the window must wait again
    clock.now = 125.0  # all recent requests are now older than 60s
    waited = limiter.acquire(100)
    assert waited == 0.0  # window has aged out


def test_rate_limiter_token_bucket():
    clock = _FakeClock()
    limiter = RateLimiter(tokens_per_second=1000, clock=clock, sleeper=clock.sleeper)
    limiter.acquire(1000)
    waited = limiter.acquire(500)  # would exceed 1000 tokens in one second
    assert waited > 0


def test_rate_limiter_honours_retry_after_globally():
    clock = _FakeClock()
    limiter = RateLimiter(clock=clock, sleeper=clock.sleeper)
    limiter.penalty(30.0)
    waited = limiter.acquire(1)
    assert abs(waited - 30.0) < 1e-9
    assert clock.sleeps == [30.0]
    assert limiter.acquire(1) == 0.0  # penalty has been consumed


def test_rate_limiter_wait_respects_cancellation():
    clock = _FakeClock()
    limiter = RateLimiter(requests_per_minute=1, clock=clock,
                          sleeper=lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")))
    cancellation = Cancellation()
    limiter.acquire(1)
    cancellation.cancel()
    try:
        limiter.acquire(1, cancellation)
        raise AssertionError("should have raised Cancelled")
    except Exception as exc:
        from jev_observatory.guards import Cancelled

        assert isinstance(exc, Cancelled)