"""The contract every crypto backend honours, and the vocabulary for refreshing.

The point of this indirection is `RefreshKind`. Two backends can both restore a
ciphertext's computation capacity, but only one of them does it with actual CKKS
bootstrapping; the other hands the ciphertext back to the key holder. Those are
not the same claim, and the system is built so that the difference survives all
the way from the primitive to the chart legend. `RefreshKind` travels inside
every decision record and every results file precisely so no layer can round it
off to "bootstrapped".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

from src.crypto.capabilities import Capabilities


class RefreshKind(str, Enum):
    """How a backend restores computation capacity.

    The `label` lives here, next to the mechanism, rather than in the UI layer,
    so that adding a backend forces its author to state plainly what its refresh
    actually is.
    """

    NATIVE_BOOTSTRAP = "native_bootstrap"
    CLIENT_AIDED = "client_aided"
    UNSUPPORTED = "unsupported"

    @property
    def label(self) -> str:
        return {
            RefreshKind.NATIVE_BOOTSTRAP: "ACTUAL CKKS BOOTSTRAPPING (library EvalBootstrap)",
            RefreshKind.CLIENT_AIDED: (
                "CLIENT-AIDED REFRESH - not CKKS bootstrapping "
                "(Microsoft SEAL does not implement it)"
            ),
            RefreshKind.UNSUPPORTED: "UNSUPPORTED - no refresh primitive available",
        }[self]

    @property
    def is_bootstrapping(self) -> bool:
        """Whether the word "bootstrapping" may honestly be used for this primitive."""
        return self is RefreshKind.NATIVE_BOOTSTRAP


class CryptoError(RuntimeError):
    """Base for failures an evaluator, not just a developer, must understand."""


class ParameterError(CryptoError):
    """The requested CKKS parameters are invalid or cryptographically unsound."""


class CapacityExhaustedError(CryptoError):
    """The ciphertext ran out of modulus chain before the operation completed.

    Raised in place of the library's bare `ValueError: scale out of bounds`, which
    tells an evaluator nothing about what to change.
    """


class MissingKeyError(CryptoError):
    """An operation needed the secret key and was invoked outside the owner zone."""


# SEAL's security tables: the largest total coeff-modulus bit count admitting a
# 128-bit security level at each ring dimension. Exceeding these makes SEAL reject
# the parameters outright, which we would rather explain than relay.
MAX_COEFF_BITS_128: dict[int, int] = {
    1024: 27,
    2048: 54,
    4096: 109,
    8192: 218,
    16384: 438,
    32768: 881,
}

# Below this scale, CKKS results stop being meaningful while continuing to look
# successful. Established by measurement - see the check in `CKKSParams`.
MIN_USABLE_SCALE_BITS = 25

# Below this scale the run is usable but precision is tight enough to be worth
# saying out loud in the interface.
LOW_PRECISION_SCALE_BITS = 32


@dataclass(frozen=True)
class CKKSParams:
    """CKKS parameters, validated on construction.

    Validation is not defensive noise. The capability probe found that
    `[50, 30, 30, 30, 50]` with a 2**40 scale is accepted by SEAL and then returns
    values with *negative* bits of precision - arithmetically meaningless output
    that still looks like a successful run. Rejecting it here is the difference
    between an error and a fabricated result.
    """

    poly_modulus_degree: int = 16384
    coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 40, 40, 40, 40, 40, 40, 50)
    scale_bits: int = 40

    def __post_init__(self) -> None:
        n = self.poly_modulus_degree
        if n not in MAX_COEFF_BITS_128:
            raise ParameterError(
                f"poly_modulus_degree must be one of {sorted(MAX_COEFF_BITS_128)}; got {n}."
            )
        if len(self.coeff_mod_bit_sizes) < 3:
            raise ParameterError(
                "coeff_mod_bit_sizes needs at least 3 primes (special + one level + final); got "
                f"{len(self.coeff_mod_bit_sizes)}. Multiplicative depth would be zero."
            )
        total = sum(self.coeff_mod_bit_sizes)
        budget = MAX_COEFF_BITS_128[n]
        if total > budget:
            raise ParameterError(
                f"coeff_mod_bit_sizes sums to {total} bits, above the {budget}-bit ceiling for "
                f"poly_modulus_degree={n} at the 128-bit security level. Either shorten the chain "
                f"or raise poly_modulus_degree to {self._next_degree(total)}."
            )
        middles = self.coeff_mod_bit_sizes[1:-1]
        if any(self.scale_bits > m for m in middles):
            raise ParameterError(
                f"scale_bits={self.scale_bits} exceeds a rescaling prime in {list(middles)}. "
                "Rescaling would destroy the message: measured precision goes negative while the "
                "run still appears to succeed. Set scale_bits at or below the smallest middle "
                f"prime ({min(middles)})."
            )
        if self.scale_bits < MIN_USABLE_SCALE_BITS:
            # Measured, not guessed: at scale 2**21 a single ciphertext-ciphertext
            # product of 1.5 and 2.0 returned 3.52 - about 2.5 bits of agreement -
            # while raising no error at all. A run at that scale looks successful and
            # its numbers are meaningless, which is the most dangerous failure this
            # system can have.
            raise ParameterError(
                f"scale_bits={self.scale_bits} is below the usable floor of "
                f"{MIN_USABLE_SCALE_BITS}. CKKS precision collapses there: at scale 2**21 a "
                "single multiplication was measured returning 3.52 where 3.0 was exact, with no "
                "error raised. Raise scale_bits (and the primes to match), or reduce the chain "
                "length so larger primes fit."
            )

    @staticmethod
    def _next_degree(total_bits: int) -> int:
        for degree, budget in sorted(MAX_COEFF_BITS_128.items()):
            if budget >= total_bits:
                return degree
        return max(MAX_COEFF_BITS_128)

    @property
    def max_depth(self) -> int:
        """Multiplications available before the chain is exhausted.

        `len(chain) - 2` was confirmed by probing: a chain of 8 primes reached
        depth 6 then raised, a chain of 10 reached 8, a chain of 20 reached 18.
        """
        return len(self.coeff_mod_bit_sizes) - 2

    @property
    def slots(self) -> int:
        """Plaintext slots in one ciphertext - the batch width we can pack."""
        return self.poly_modulus_degree // 2

    @property
    def precision_warning(self) -> str | None:
        """A caution about tight precision, or `None` when there is nothing to say.

        Separate from validation because this is a judgement, not a rule: these
        parameters work, and the results will simply carry fewer significant bits.
        The interface shows it so the trade is visible rather than discovered in
        the accuracy figures afterwards.
        """
        if self.scale_bits < LOW_PRECISION_SCALE_BITS:
            return (
                f"scale_bits={self.scale_bits} is low. Expect roughly "
                f"{self.scale_bits - 20}-{self.scale_bits - 12} bits of agreement with exact "
                "arithmetic, falling as the chain is consumed. Check the measured canary "
                "precision on the Capacity Monitor before trusting the accuracy figures."
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_mod_bit_sizes": list(self.coeff_mod_bit_sizes),
            "scale_bits": self.scale_bits,
            "max_depth": self.max_depth,
            "slots": self.slots,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CKKSParams":
        return cls(
            poly_modulus_degree=int(data["poly_modulus_degree"]),
            coeff_mod_bit_sizes=tuple(int(b) for b in data["coeff_mod_bit_sizes"]),
            scale_bits=int(data.get("scale_bits", 40)),
        )


@dataclass
class EncVector:
    """A ciphertext plus the bookkeeping the library will not give us.

    TenSEAL exposes no level accessor (verified against binding.cpp at v0.3.16 and
    v0.3.18), so `depth` is maintained by the operation layer. It is not a guess:
    `CapacityMonitor` re-derives depth from the measured serialized size and flags
    any disagreement, so this counter is checked against the library rather than
    trusted.
    """

    raw: Any
    depth: int = 0
    lineage: str = "ct"
    ops: int = 0
    history: list[str] = field(default_factory=list)

    def derive(self, raw: Any, *, cost: int, op: str) -> "EncVector":
        return EncVector(
            raw=raw,
            depth=self.depth + cost,
            lineage=self.lineage,
            ops=self.ops + 1,
            history=[*self.history[-15:], op],
        )


@runtime_checkable
class CryptoBackend(Protocol):
    """Operations the training engine may perform. Deliberately small.

    There is no argument anywhere that relaxes a restriction or grants the compute
    zone a secret key: the only way to decrypt is to hold the owner-zone object.
    """

    name: str
    params: CKKSParams

    def capabilities(self) -> Capabilities: ...

    def refresh_kind(self) -> RefreshKind: ...

    def encrypt(self, values: Sequence[float], *, lineage: str = "ct") -> EncVector: ...

    def decrypt(self, vec: EncVector) -> list[float]: ...

    def add(self, a: EncVector, b: EncVector) -> EncVector: ...

    def sub(self, a: EncVector, b: EncVector) -> EncVector: ...

    def mul(self, a: EncVector, b: EncVector) -> EncVector: ...

    def mul_plain(self, a: EncVector, other: float | Sequence[float]) -> EncVector: ...

    def add_plain(self, a: EncVector, other: float | Sequence[float]) -> EncVector: ...

    def polyval(self, a: EncVector, coeffs: Sequence[float]) -> EncVector: ...

    def matmul_plain(self, a: EncVector, matrix: list[list[float]]) -> EncVector: ...

    def serialized_size(self, vec: EncVector) -> int: ...
