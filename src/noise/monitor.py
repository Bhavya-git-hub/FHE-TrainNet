"""Ciphertext Level / Remaining Computation Capacity.

This module is where the project is most at risk of telling a comfortable lie, so
it is the most explicit.

Microsoft SEAL exposes **no noise budget for CKKS**, and TenSEAL binds no
`scale()`, `level()` or `noise_budget()` accessor on `CKKSVector`. Those three and
a `bootstrap()` call were all probed, and all four raise `AttributeError` (see
`docs/LIBRARY_CAPABILITIES.md`).
There is therefore no quantity in this system that may honestly be printed as
"noise = 73%", and none is. What exists instead are three indicators, each
carrying a tag saying how it was obtained:

* `levels_remaining` - DERIVED. Counted by the operation layer against the
  modulus chain we configured. Exact, because CKKS level consumption is
  deterministic, but it is our arithmetic and not a library reading.
* `serialized_bytes` - MEASURED. The length of `ciphertext.serialize()`. A
  ciphertext sheds one RNS limb per level consumed, so this falls in clean steps
  (174 kB per level at the shipped parameters, measured to within 70 bytes across
  six runs; the step scales with the prime size). This is the library's own account of the
  ciphertext, and it is used to *check* the derived level rather than merely sit
  beside it.
* `precision_bits` - MEASURED. A canary ciphertext of known value travels the
  same operation sequence; the data owner decrypts it and compares against the
  exact arithmetic result. This is real CKKS approximation error - the physical
  consequence of noise growth - and never a substitute for a noise budget.

The cross-check matters. If the derived level and the measured size disagree, the
operation layer's depth accounting is wrong, and `CapacityReading.consistent` is
False so it can be surfaced instead of silently skewing every decision after it.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from src.crypto.backend import CKKSParams, EncVector

# Label used wherever this quantity is shown. Deliberately not "noise".
METRIC_NAME = "Ciphertext Level / Remaining Computation Capacity"

METRIC_DISCLAIMER = (
    "Microsoft SEAL exposes no CKKS noise budget and TenSEAL binds no scale(), level() or "
    "noise_budget() accessor, so no literal ciphertext noise value is available and none is "
    "shown. Level is derived from the modulus chain; ciphertext size and precision are measured."
)


class Provenance(str, Enum):
    """How a number came to exist. Shown next to it everywhere."""

    MEASURED = "measured"
    DERIVED = "derived"
    UNAVAILABLE = "unavailable"


class CapacityState(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


@dataclass
class CapacityReading:
    """One observation of a ciphertext's remaining capacity."""

    index: int
    label: str
    iteration: int
    epoch: int
    levels_consumed: int
    levels_remaining: int
    max_depth: int
    serialized_bytes: int | None
    measured_levels: int | None
    precision_bits: float | None
    state: CapacityState
    consistent: bool = True
    timestamp: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0

    # Provenance is attached to the reading, not assumed by the reader.
    provenance: dict[str, str] = field(
        default_factory=lambda: {
            "levels_remaining": Provenance.DERIVED.value,
            "serialized_bytes": Provenance.MEASURED.value,
            "precision_bits": Provenance.MEASURED.value,
        }
    )

    @property
    def fraction_remaining(self) -> float | None:
        """Share of the chain still usable. A ratio of levels, never a noise share."""
        if self.max_depth <= 0:
            return None
        return max(0.0, self.levels_remaining / self.max_depth)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["fraction_remaining"] = self.fraction_remaining
        return data


class CapacityMonitor:
    """Records capacity readings and cross-checks the derived level.

    The calibration table is built once per context by encrypting a probe and
    multiplying it down the chain, recording the serialized size at each level.
    That turns a measured byte count back into a measured level, which is what
    makes the consistency check possible at all.
    """

    def __init__(
        self,
        params: CKKSParams,
        *,
        warning_fraction: float = 0.4,
        critical_fraction: float = 0.15,
    ) -> None:
        if not 0.0 < critical_fraction < warning_fraction < 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 < critical < warning < 1; got "
                f"critical={critical_fraction}, warning={warning_fraction}."
            )
        self.params = params
        self.warning_fraction = warning_fraction
        self.critical_fraction = critical_fraction
        self.readings: list[CapacityReading] = []
        self._calibration: list[int] = []
        self._bytes_per_level: float | None = None
        self._started = time.perf_counter()
        self.inconsistencies: list[str] = []

    # -- calibration -----------------------------------------------------

    def calibrate(self, owner: Any, backend: Any, *, slots: int = 8) -> list[int]:
        """Measure serialized size at every level of this context.

        Costs one encryption and `max_depth` plaintext multiplications, which is
        negligible next to training, and it is what lets every later reading be
        checked against the library instead of trusted.
        """
        probe = owner.encrypt([1.0] * slots, lineage="calibration")
        sizes = [backend.serialized_size(probe)]
        ones = [1.0] * slots
        for _ in range(self.params.max_depth):
            probe = backend.mul_plain(probe, ones)
            sizes.append(backend.serialized_size(probe))
        self._calibration = sizes
        span = sizes[0] - sizes[-1]
        self._bytes_per_level = span / self.params.max_depth if self.params.max_depth else None
        return sizes

    def measured_level(self, size_bytes: int) -> int | None:
        """Recover the level from a measured ciphertext size.

        Nearest-bucket rather than exact match: sizes vary by a few hundred bytes
        between ciphertexts at the same level, while the gap *between* levels is
        174 kB at the shipped parameters - three orders of magnitude larger - so
        the nearest bucket is unambiguous.
        """
        if not self._calibration:
            return None
        return min(
            range(len(self._calibration)),
            key=lambda i: abs(self._calibration[i] - size_bytes),
        )

    @property
    def calibration(self) -> list[int]:
        return list(self._calibration)

    @property
    def bytes_per_level(self) -> float | None:
        return self._bytes_per_level

    # -- recording -------------------------------------------------------

    def classify(self, levels_remaining: int, needed: int | None = None) -> CapacityState:
        """Traffic-light state for display.

        When the caller says how much the next step needs, that governs: a
        ciphertext with two levels left is fine for a one-level step and
        exhausted for a five-level one, and a fixed percentage cannot tell the
        difference.
        """
        if levels_remaining <= 0:
            return CapacityState.EXHAUSTED
        if needed is not None:
            if levels_remaining < needed:
                return CapacityState.EXHAUSTED
            if levels_remaining < 2 * needed:
                return CapacityState.CRITICAL
            return CapacityState.SAFE
        fraction = levels_remaining / self.params.max_depth if self.params.max_depth else 0.0
        if fraction <= self.critical_fraction:
            return CapacityState.CRITICAL
        if fraction <= self.warning_fraction:
            return CapacityState.WARNING
        return CapacityState.SAFE

    def record(
        self,
        vec: EncVector,
        *,
        label: str,
        iteration: int = 0,
        epoch: int = 0,
        backend: Any | None = None,
        canary: Callable[[], float | None] | None = None,
        needed: int | None = None,
        measure_size: bool = True,
    ) -> CapacityReading:
        """Take a reading. `measure_size=False` skips the ~19 ms serialization."""
        consumed = vec.depth
        remaining = self.params.max_depth - consumed

        size_bytes: int | None = None
        measured: int | None = None
        consistent = True
        if measure_size and backend is not None:
            size_bytes = backend.serialized_size(vec)
            measured = self.measured_level(size_bytes)
            if measured is not None and measured != consumed:
                consistent = False
                self.inconsistencies.append(
                    f"{label}: depth accounting says level {consumed}, measured ciphertext size "
                    f"({size_bytes} bytes) says level {measured}."
                )

        precision: float | None = None
        if canary is not None:
            precision = canary()

        reading = CapacityReading(
            index=len(self.readings),
            label=label,
            iteration=iteration,
            epoch=epoch,
            levels_consumed=consumed,
            levels_remaining=remaining,
            max_depth=self.params.max_depth,
            serialized_bytes=size_bytes,
            measured_levels=measured,
            precision_bits=precision,
            state=self.classify(remaining, needed),
            consistent=consistent,
            elapsed_seconds=time.perf_counter() - self._started,
        )
        if size_bytes is None:
            reading.provenance["serialized_bytes"] = Provenance.UNAVAILABLE.value
        if precision is None:
            reading.provenance["precision_bits"] = Provenance.UNAVAILABLE.value
        self.readings.append(reading)
        return reading

    # -- reporting -------------------------------------------------------

    def series(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.readings]

    def summary(self) -> dict[str, Any]:
        """Aggregate view. Missing quantities stay missing, never zero."""
        precisions = [r.precision_bits for r in self.readings if r.precision_bits is not None]
        return {
            "metric_name": METRIC_NAME,
            "disclaimer": METRIC_DISCLAIMER,
            "readings": len(self.readings),
            "max_depth": self.params.max_depth,
            "min_levels_remaining": (
                min((r.levels_remaining for r in self.readings), default=None)
            ),
            "final_levels_remaining": (self.readings[-1].levels_remaining if self.readings else None),
            "precision_bits_first": precisions[0] if precisions else None,
            "precision_bits_last": precisions[-1] if precisions else None,
            "precision_bits_min": min(precisions) if precisions else None,
            "bytes_per_level": self._bytes_per_level,
            "calibration_sizes": self._calibration,
            "consistency_failures": len(self.inconsistencies),
            "inconsistency_detail": self.inconsistencies[:10],
        }


class CanaryProbe:
    """A known value carried through the same operations as the real data.

    The canary is what turns "CKKS is approximate" into a number. It starts at a
    known value, undergoes the same multiplications the weights undergo, and the
    owner decrypts it and compares with the exact result computed in the clear.

    Note the trust implication, which is stated rather than glossed: reading this
    indicator needs the secret key, so it is an owner-zone measurement. The level
    and byte-size indicators need no key and are computable by the untrusted
    party. `AdaptivePolicy` can therefore be configured to use only the
    key-free indicators, and the honest deployment story differs between the two.
    """

    def __init__(self, owner: Any, backend: Any, *, slots: int, value: float = 1.0) -> None:
        self.owner = owner
        self.backend = backend
        self.slots = slots
        self.value = value
        self.expected = value
        self.exhausted = False
        self.reason: str | None = None
        self.vec = owner.encrypt([value] * slots, lineage="canary")

    def apply_plain(self, factor: float) -> None:
        """Mirror a multiplication onto the canary and onto its exact value.

        A canary that runs out of chain stops reporting rather than taking the run
        down with it. It consumes depth in step with the real ciphertexts, so the
        two exhaust together - and under a policy that never refreshes, the
        training failure is the result worth reporting. A diagnostic probe must not
        be what raises it.
        """
        try:
            self.vec = self.backend.mul_plain(self.vec, [factor] * self.slots)
            self.expected *= factor
        except Exception as exc:  # noqa: BLE001 - recorded as unavailable, with its reason
            self.exhausted = True
            self.reason = f"{type(exc).__name__}: {exc}"

    def reset(self) -> None:
        self.vec = self.owner.encrypt([self.value] * self.slots, lineage="canary")
        self.expected = self.value
        self.exhausted = False
        self.reason = None

    def precision_bits(self) -> float | None:
        """Bits of agreement between the decrypted canary and exact arithmetic.

        `None` when the two are bit-identical: "infinitely precise" is not a
        number that belongs on a chart, and substituting a large constant would
        be exactly the fabrication this project forbids. Also `None` once the
        probe has run out of chain - an unmeasurable quantity, not a zero.
        """
        if self.exhausted:
            return None
        try:
            decrypted = self.owner.decrypt(self.vec)[0]
        except Exception:  # noqa: BLE001 - a failed probe is reported as missing, not as zero
            return None
        if self.expected == 0:
            return None
        error = abs(decrypted - self.expected) / abs(self.expected)
        if error == 0:
            return None
        return -math.log2(error)
