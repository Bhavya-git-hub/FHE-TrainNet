"""When to refresh a ciphertext: a fixed schedule, an adaptive rule, and nothing.

This is the project's contribution, so two things matter more than the code:

1. **Every decision is recorded with its reason**, whether it was to refresh or to
   continue. A policy that only logs the refreshes hides exactly the evidence
   needed to argue it refreshed less often.

2. **No policy claims to bootstrap.** Each decision carries the `RefreshKind` the
   backend actually provides, so a client-aided refresh is reported as one all the
   way into the charts.

The comparison the experiments run is not "adaptive beats fixed". It is:

* a fixed interval that is too long makes the run FAIL on capacity exhaustion;
* a fixed interval that is too short wastes refreshes;
* exactly one interval is optimal, and it has to be found by tuning;
* the adaptive rule finds it without tuning, and keeps finding it when the
  configuration changes underneath it - a different activation degree, more
  features, a different batch size - which is when a tuned constant breaks.

Whether adaptive actually wins on wall-clock time is a measurement, not an
assumption, and the runner reports whatever it finds.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol

from src.crypto.backend import RefreshKind
from src.noise.monitor import CapacityReading


class Decision(str, Enum):
    CONTINUE = "continue"
    REFRESH = "refresh"


@dataclass
class DecisionRecord:
    """One controller decision, with everything needed to audit it."""

    index: int
    iteration: int
    epoch: int
    decision: Decision
    reason: str
    policy: str
    levels_remaining: int
    levels_needed: int
    max_depth: int
    threshold: str
    refresh_kind: str
    precision_bits: float | None = None
    serialized_bytes: int | None = None
    refresh_seconds: float | None = None
    cumulative_refreshes: int = 0
    timestamp: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


class DecisionLog:
    """Every decision, in order. Continues included."""

    def __init__(self) -> None:
        self.records: list[DecisionRecord] = []
        self.refresh_count = 0
        self.refresh_seconds = 0.0
        self.started = time.perf_counter()

    def add(self, record: DecisionRecord) -> DecisionRecord:
        record.index = len(self.records)
        record.elapsed_seconds = time.perf_counter() - self.started
        if record.decision is Decision.REFRESH:
            self.refresh_count += 1
            if record.refresh_seconds:
                self.refresh_seconds += record.refresh_seconds
        record.cumulative_refreshes = self.refresh_count
        self.records.append(record)
        return record

    @property
    def continues(self) -> int:
        return sum(1 for r in self.records if r.decision is Decision.CONTINUE)

    def rows(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def summary(self) -> dict[str, Any]:
        times = [r.refresh_seconds for r in self.records if r.refresh_seconds]
        return {
            "decisions": len(self.records),
            "refreshes": self.refresh_count,
            "continues": self.continues,
            "refresh_seconds_total": self.refresh_seconds,
            # None, not 0.0: a run with no refreshes has no average refresh time.
            "refresh_seconds_mean": (sum(times) / len(times)) if times else None,
            "refresh_seconds_max": max(times) if times else None,
        }


class RefreshPolicy(Protocol):
    """Decide, given the state of a ciphertext, whether to refresh before the next step."""

    name: str

    def describe(self) -> str: ...

    def threshold_text(self) -> str: ...

    def decide(self, reading: CapacityReading, needed: int, step: int) -> tuple[Decision, str]: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class BaselinePolicy:
    """Refresh every `interval` steps, regardless of ciphertext state.

    The control condition, and a fair representation of fixed-schedule policies:
    it is not told the level, so it cannot react to one. Its weakness is that
    `interval` must be chosen correctly in advance, and the correct value depends
    on the activation degree, the feature count and the modulus chain - all of
    which an evaluator can change from the dashboard.
    """

    interval: int = 1
    name: str = "baseline_fixed"

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(
                f"Baseline refresh interval must be at least 1 step; got {self.interval}."
            )

    def describe(self) -> str:
        return (
            f"Fixed policy: refresh every {self.interval} training step(s), without inspecting the "
            "ciphertext. Chosen in advance; not adaptive."
        )

    def threshold_text(self) -> str:
        return f"every {self.interval} step(s)"

    def decide(self, reading: CapacityReading, needed: int, step: int) -> tuple[Decision, str]:
        if step > 0 and step % self.interval == 0:
            return (
                Decision.REFRESH,
                f"fixed schedule reached: step {step} is a multiple of {self.interval} "
                f"(ciphertext state was not consulted; {reading.levels_remaining} level(s) "
                "remained)",
            )
        return (
            Decision.CONTINUE,
            f"fixed schedule not due: step {step} of every {self.interval} "
            f"({reading.levels_remaining} level(s) remained, not consulted)",
        )

    def to_dict(self) -> dict[str, Any]:
        return {"policy": self.name, "interval": self.interval, "describe": self.describe()}


@dataclass
class AdaptivePolicy:
    """Refresh when what remains will not cover the next step.

    Two differences from a threshold on a percentage:

    * The rule compares against `needed` - the depth the *next* step will actually
      consume - rather than against a fixed fraction. A percentage cannot know
      that a degree-5 activation needs one more level than a degree-3 one.
    * `safety_margin` keeps a configurable cushion, so an evaluator can watch the
      decisions change as they move it.

    `min_precision_bits` is optional and off by default, because reading measured
    precision requires the secret key. With it enabled the policy is owner-
    assisted; with it disabled the policy runs on indicators the untrusted compute
    zone can observe by itself. That distinction is real and is documented rather
    than blurred.
    """

    safety_margin: int = 0
    min_precision_bits: float | None = None
    name: str = "adaptive"

    def __post_init__(self) -> None:
        if self.safety_margin < 0:
            raise ValueError(f"safety_margin cannot be negative; got {self.safety_margin}.")
        if self.min_precision_bits is not None and self.min_precision_bits <= 0:
            raise ValueError(
                f"min_precision_bits must be positive when set; got {self.min_precision_bits}."
            )

    def describe(self) -> str:
        parts = [
            "Adaptive policy: refresh only when the levels remaining cannot cover the next "
            f"training step (plus a safety margin of {self.safety_margin})."
        ]
        if self.min_precision_bits is not None:
            parts.append(
                f"Also refreshes if measured canary precision falls below "
                f"{self.min_precision_bits} bits (requires the data-owner key)."
            )
        else:
            parts.append(
                "Uses only indicators the untrusted compute zone can observe without a key."
            )
        return " ".join(parts)

    def threshold_text(self) -> str:
        base = f"levels_remaining < next_step_depth + {self.safety_margin}"
        if self.min_precision_bits is not None:
            return f"{base}, or precision < {self.min_precision_bits} bits"
        return base

    def decide(self, reading: CapacityReading, needed: int, step: int) -> tuple[Decision, str]:
        required = needed + self.safety_margin
        if reading.levels_remaining < required:
            return (
                Decision.REFRESH,
                f"{reading.levels_remaining} level(s) remain; the next step needs {needed} "
                f"(+{self.safety_margin} margin = {required}). Continuing would exhaust the "
                "modulus chain.",
            )
        if (
            self.min_precision_bits is not None
            and reading.precision_bits is not None
            and reading.precision_bits < self.min_precision_bits
        ):
            return (
                Decision.REFRESH,
                f"measured canary precision {reading.precision_bits:.1f} bits is below the "
                f"{self.min_precision_bits} bit floor, although {reading.levels_remaining} "
                f"level(s) remain.",
            )
        return (
            Decision.CONTINUE,
            f"{reading.levels_remaining} level(s) remain and the next step needs {required}; "
            "the ciphertext is within its operating range.",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "safety_margin": self.safety_margin,
            "min_precision_bits": self.min_precision_bits,
            "describe": self.describe(),
        }


@dataclass
class NoRefreshPolicy:
    """Never refresh.

    Kept because it is the only way to show that the capacity limit is real. A run
    under this policy is *expected* to fail once the chain is exhausted; the
    failure is recorded as a FAILED run with its reason rather than hidden, and it
    is what makes the other policies' refresh counts meaningful.
    """

    name: str = "no_refresh"

    def describe(self) -> str:
        return (
            "No refreshing at all. Included as a control: this run is expected to fail with "
            "capacity exhaustion, demonstrating that the limit the other policies manage is real."
        )

    def threshold_text(self) -> str:
        return "never refresh"

    def decide(self, reading: CapacityReading, needed: int, step: int) -> tuple[Decision, str]:
        return (
            Decision.CONTINUE,
            f"policy never refreshes ({reading.levels_remaining} level(s) remain, next step needs "
            f"{needed})",
        )

    def to_dict(self) -> dict[str, Any]:
        return {"policy": self.name, "describe": self.describe()}


def build_policy(spec: dict[str, Any]) -> RefreshPolicy:
    """Construct a policy from configuration.

    Raises on an unknown name rather than quietly falling back to a default: a
    typo in a config file must not silently change which policy an experiment
    measured.
    """
    kind = str(spec.get("policy", "adaptive"))
    if kind in ("baseline_fixed", "baseline", "fixed"):
        return BaselinePolicy(interval=int(spec.get("interval", 1)))
    if kind == "adaptive":
        floor = spec.get("min_precision_bits")
        return AdaptivePolicy(
            safety_margin=int(spec.get("safety_margin", 0)),
            min_precision_bits=float(floor) if floor is not None else None,
        )
    if kind in ("no_refresh", "none"):
        return NoRefreshPolicy()
    raise ValueError(
        f"Unknown refresh policy '{kind}'. Valid values: baseline_fixed, adaptive, no_refresh."
    )
