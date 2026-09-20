"""The capacity monitor and the refresh controllers.

The policies are tested without any cryptography at all. That is deliberate: the
specification requires the controller to be verifiable even where the backend
cannot perform real bootstrapping, so the decision logic must be exercisable in
isolation from whatever primitive happens to be available.
"""

from __future__ import annotations

import pytest

from src.bootstrapping.policies import (
    AdaptivePolicy,
    BaselinePolicy,
    Decision,
    DecisionLog,
    DecisionRecord,
    NoRefreshPolicy,
    build_policy,
)
from src.crypto.backend import CKKSParams, RefreshKind
from src.noise.monitor import (
    CanaryProbe,
    CapacityMonitor,
    CapacityReading,
    CapacityState,
    Provenance,
)


def reading(levels_remaining: int, max_depth: int = 10, precision: float | None = None):
    return CapacityReading(
        index=0, label="t", iteration=0, epoch=0,
        levels_consumed=max_depth - levels_remaining,
        levels_remaining=levels_remaining, max_depth=max_depth,
        serialized_bytes=None, measured_levels=None, precision_bits=precision,
        state=CapacityState.SAFE,
    )


# -- monitor --------------------------------------------------------------


def test_calibration_is_monotone_and_maps_size_back_to_level(owner, backend, params) -> None:
    monitor = CapacityMonitor(params)
    sizes = monitor.calibrate(owner, backend, slots=4)
    assert len(sizes) == params.max_depth + 1
    assert sizes == sorted(sizes, reverse=True), "ciphertexts must shrink as levels are consumed"
    for level, size in enumerate(sizes):
        assert monitor.measured_level(size) == level


def test_derived_level_agrees_with_measured_size(owner, backend, params) -> None:
    """The cross-check that makes the derived count trustworthy."""
    monitor = CapacityMonitor(params)
    monitor.calibrate(owner, backend, slots=4)
    vec = owner.encrypt([1.0] * 4)
    for step in range(params.max_depth):
        vec = backend.mul_plain(vec, [1.0] * 4)
        r = monitor.record(vec, label=f"s{step}", backend=backend)
        assert r.consistent, r
        assert r.measured_levels == r.levels_consumed == step + 1
    assert monitor.inconsistencies == []


def test_monitor_detects_wrong_depth_accounting(owner, backend, params) -> None:
    """If the bookkeeping lies, the monitor must notice rather than pass it on."""
    monitor = CapacityMonitor(params)
    monitor.calibrate(owner, backend, slots=4)
    vec = backend.mul_plain(owner.encrypt([1.0] * 4), [1.0] * 4)
    vec.depth = 0  # corrupt the counter
    r = monitor.record(vec, label="corrupt", backend=backend)
    assert r.consistent is False
    assert monitor.inconsistencies


def test_missing_measurements_stay_none(params) -> None:
    monitor = CapacityMonitor(params)
    r = monitor.record(type("V", (), {"depth": 2})(), label="x", measure_size=False)
    assert r.serialized_bytes is None
    assert r.precision_bits is None
    assert r.provenance["serialized_bytes"] == Provenance.UNAVAILABLE.value
    assert r.provenance["precision_bits"] == Provenance.UNAVAILABLE.value


def test_state_depends_on_what_the_next_step_needs(params) -> None:
    monitor = CapacityMonitor(params)
    # Not enough for even one more step.
    assert monitor.classify(2, needed=5) is CapacityState.EXHAUSTED
    # Room for exactly one more step, and then nothing.
    assert monitor.classify(1, needed=1) is CapacityState.CRITICAL
    assert monitor.classify(5, needed=3) is CapacityState.CRITICAL
    # Room for two or more.
    assert monitor.classify(2, needed=1) is CapacityState.SAFE
    assert monitor.classify(9, needed=1) is CapacityState.SAFE
    assert monitor.classify(0) is CapacityState.EXHAUSTED


def test_monitor_rejects_incoherent_thresholds(params) -> None:
    with pytest.raises(ValueError, match="critical < warning"):
        CapacityMonitor(params, warning_fraction=0.1, critical_fraction=0.5)


def test_canary_measures_real_precision_loss(owner, backend, params) -> None:
    canary = CanaryProbe(owner, backend, slots=4, value=1.0)
    first = canary.precision_bits()
    for _ in range(params.max_depth):
        canary.apply_plain(1.0)
    last = canary.precision_bits()
    assert first is not None and last is not None
    assert last < first, "precision must degrade as the chain is consumed"


def test_canary_reset_restores_precision(owner, backend, params) -> None:
    canary = CanaryProbe(owner, backend, slots=4, value=1.0)
    for _ in range(params.max_depth):
        canary.apply_plain(1.0)
    degraded = canary.precision_bits()
    canary.reset()
    assert canary.precision_bits() > degraded


def test_summary_reports_none_not_zero_when_nothing_measured(params) -> None:
    summary = CapacityMonitor(params).summary()
    assert summary["min_levels_remaining"] is None
    assert summary["precision_bits_first"] is None
    assert "no CKKS noise budget" in summary["disclaimer"]


# -- baseline policy ------------------------------------------------------


def test_baseline_refreshes_on_its_schedule_only() -> None:
    policy = BaselinePolicy(interval=2)
    assert policy.decide(reading(10), needed=5, step=0)[0] is Decision.CONTINUE
    assert policy.decide(reading(10), needed=5, step=1)[0] is Decision.CONTINUE
    assert policy.decide(reading(10), needed=5, step=2)[0] is Decision.REFRESH


def test_baseline_ignores_the_ciphertext_state() -> None:
    """The control condition: it refreshes on schedule even at zero levels left."""
    policy = BaselinePolicy(interval=5)
    decision, why = policy.decide(reading(0), needed=5, step=1)
    assert decision is Decision.CONTINUE
    assert "not consulted" in why


def test_baseline_rejects_a_zero_interval() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BaselinePolicy(interval=0)


# -- adaptive policy ------------------------------------------------------


def test_adaptive_refreshes_only_when_the_next_step_will_not_fit() -> None:
    policy = AdaptivePolicy()
    assert policy.decide(reading(5), needed=5, step=3)[0] is Decision.CONTINUE
    assert policy.decide(reading(4), needed=5, step=3)[0] is Decision.REFRESH


def test_adaptive_safety_margin_shifts_the_decision() -> None:
    at_five = reading(5)
    assert AdaptivePolicy(safety_margin=0).decide(at_five, 5, 1)[0] is Decision.CONTINUE
    assert AdaptivePolicy(safety_margin=1).decide(at_five, 5, 1)[0] is Decision.REFRESH


def test_adaptive_responds_to_the_activation_cost() -> None:
    """A percentage threshold could not do this."""
    policy = AdaptivePolicy()
    assert policy.decide(reading(4), needed=4, step=1)[0] is Decision.CONTINUE
    assert policy.decide(reading(4), needed=6, step=1)[0] is Decision.REFRESH


def test_adaptive_precision_floor_is_optional_and_off_by_default() -> None:
    plain = AdaptivePolicy()
    assert plain.min_precision_bits is None
    assert "without a key" in plain.describe()

    strict = AdaptivePolicy(min_precision_bits=15.0)
    decision, why = strict.decide(reading(10, precision=9.0), needed=5, step=1)
    assert decision is Decision.REFRESH
    assert "precision" in why


def test_adaptive_ignores_precision_when_it_was_not_measured() -> None:
    policy = AdaptivePolicy(min_precision_bits=15.0)
    assert policy.decide(reading(10, precision=None), needed=5, step=1)[0] is Decision.CONTINUE


def test_adaptive_rejects_a_negative_margin() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        AdaptivePolicy(safety_margin=-1)


# -- no-refresh control ---------------------------------------------------


def test_no_refresh_never_refreshes() -> None:
    policy = NoRefreshPolicy()
    assert policy.decide(reading(0), needed=5, step=99)[0] is Decision.CONTINUE
    assert "expected to fail" in policy.describe()


# -- decision log ---------------------------------------------------------


def test_log_records_continues_as_well_as_refreshes() -> None:
    log = DecisionLog()
    for i, decision in enumerate([Decision.CONTINUE, Decision.REFRESH, Decision.CONTINUE]):
        log.add(
            DecisionRecord(
                index=0, iteration=i, epoch=0, decision=decision, reason="r",
                policy="p", levels_remaining=5, levels_needed=5, max_depth=10,
                threshold="t", refresh_kind=RefreshKind.CLIENT_AIDED.value,
                refresh_seconds=0.5 if decision is Decision.REFRESH else None,
            )
        )
    summary = log.summary()
    assert summary["refreshes"] == 1
    assert summary["continues"] == 2
    assert summary["decisions"] == 3
    assert summary["refresh_seconds_mean"] == pytest.approx(0.5)


def test_log_reports_no_mean_when_there_were_no_refreshes() -> None:
    """A run with no refreshes has no average refresh time - not an average of zero."""
    assert DecisionLog().summary()["refresh_seconds_mean"] is None


def test_every_record_carries_the_refresh_kind() -> None:
    log = DecisionLog()
    log.add(
        DecisionRecord(
            index=0, iteration=0, epoch=0, decision=Decision.REFRESH, reason="r", policy="p",
            levels_remaining=0, levels_needed=5, max_depth=10, threshold="t",
            refresh_kind=RefreshKind.CLIENT_AIDED.value,
        )
    )
    assert log.rows()[0]["refresh_kind"] == "client_aided"


# -- construction ---------------------------------------------------------


def test_build_policy_from_config() -> None:
    assert isinstance(build_policy({"policy": "adaptive"}), AdaptivePolicy)
    assert build_policy({"policy": "baseline_fixed", "interval": 3}).interval == 3
    assert isinstance(build_policy({"policy": "no_refresh"}), NoRefreshPolicy)


def test_unknown_policy_raises_rather_than_defaulting() -> None:
    """A typo must not silently change which policy an experiment measured."""
    with pytest.raises(ValueError, match="Unknown refresh policy"):
        build_policy({"policy": "adpative"})


def test_canary_exhaustion_is_reported_not_raised(owner, backend, params) -> None:
    """A diagnostic probe running out of chain must not take the run down with it."""
    canary = CanaryProbe(owner, backend, slots=4, value=1.0)
    for _ in range(params.max_depth + 3):
        canary.apply_plain(1.0)  # must not raise
    assert canary.exhausted is True
    assert canary.reason
    assert canary.precision_bits() is None  # unmeasurable, not zero
    canary.reset()
    assert canary.exhausted is False
    assert canary.precision_bits() is not None
