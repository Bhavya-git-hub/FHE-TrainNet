"""CKKS via TenSEAL / Microsoft SEAL, split into an owner zone and a compute zone.

The split is the whole security story of the project (SRS NFR-01, NFR-02), so it
is enforced by construction rather than by convention. `DataOwnerZone` holds the
only context carrying a secret key. `TenSEALBackend` - the object the training
engine is handed - is built from a context produced by `make_context_public()`,
and ciphertexts cross between the two by serialization, exactly as they would
cross a network to a cloud worker.

That is not decoration. A probe confirmed that calling `decrypt()` on a
ciphertext living in the public context raises
`ValueError: the current context of the tensor doesn't hold a secret_key`. The
training engine therefore *cannot* read its inputs, and a test asserts it.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import tenseal as ts

from src.crypto.backend import (
    CapacityExhaustedError,
    CKKSParams,
    CryptoError,
    EncVector,
    MissingKeyError,
    ParameterError,
    RefreshKind,
)
from src.crypto.capabilities import Capabilities, attempt

# TenSEAL raises a bare ValueError carrying this text when a ciphertext has run
# out of modulus chain. Matching on it lets us re-raise something an evaluator can
# act on, without suppressing anything we did not recognise.
_EXHAUSTION_MARKERS = ("scale out of bounds", "end of modulus switching chain")


def _translate(exc: Exception, op: str, depth: int, max_depth: int) -> CryptoError:
    text = str(exc).lower()
    if any(marker in text for marker in _EXHAUSTION_MARKERS):
        return CapacityExhaustedError(
            f"Operation '{op}' exhausted the ciphertext's computation capacity: it was already at "
            f"depth {depth} of the {max_depth} multiplicative levels this modulus chain provides. "
            "Refresh the ciphertext before this operation, lengthen coeff_mod_bit_sizes, or raise "
            f"poly_modulus_degree. (library said: {exc})"
        )
    return CryptoError(f"Operation '{op}' failed at depth {depth}: {type(exc).__name__}: {exc}")


class TenSEALBackend:
    """The compute zone. Performs homomorphic work; cannot decrypt.

    `polyval` costs were measured, not assumed: a degree-3 polynomial consumed two
    levels and a degree-2 polynomial one, matching `ceil(log2(degree))`. The
    counter is still only bookkeeping, and `CapacityMonitor` checks it against the
    serialized size the library actually produces.
    """

    name = "tenseal"

    def __init__(self, public_context: "ts.Context", params: CKKSParams) -> None:
        self._ctx = public_context
        self.params = params

    # -- introspection ---------------------------------------------------

    def refresh_kind(self) -> RefreshKind:
        """SEAL implements no CKKS bootstrapping, so this backend never claims it."""
        return RefreshKind.CLIENT_AIDED

    def capabilities(self) -> Capabilities:
        return probe_tenseal(self.params)

    # -- operations ------------------------------------------------------

    def _guard(self, op: str, depth: int, fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed, explained error
            raise _translate(exc, op, depth, self.params.max_depth) from exc

    def encrypt(self, values: Sequence[float], *, lineage: str = "ct") -> EncVector:
        raise MissingKeyError(
            "The compute zone cannot encrypt: encryption is a data-owner operation. "
            "Use DataOwnerZone.encrypt()."
        )

    def decrypt(self, vec: EncVector) -> list[float]:
        raise MissingKeyError(
            "The compute zone holds no secret key by design (SRS NFR-02). Decryption is available "
            "only through DataOwnerZone.decrypt()."
        )

    def add(self, a: EncVector, b: EncVector) -> EncVector:
        raw = self._guard("add", a.depth, lambda: a.raw + b.raw)
        out = a.derive(raw, cost=0, op="add")
        out.depth = max(a.depth, b.depth)
        return out

    def sub(self, a: EncVector, b: EncVector) -> EncVector:
        raw = self._guard("sub", a.depth, lambda: a.raw - b.raw)
        out = a.derive(raw, cost=0, op="sub")
        out.depth = max(a.depth, b.depth)
        return out

    def mul(self, a: EncVector, b: EncVector) -> EncVector:
        """Ciphertext x ciphertext. One level, and the expensive primitive."""
        raw = self._guard("mul", max(a.depth, b.depth), lambda: a.raw * b.raw)
        out = a.derive(raw, cost=0, op="mul")
        out.depth = max(a.depth, b.depth) + 1
        return out

    def mul_plain(self, a: EncVector, other: float | Sequence[float]) -> EncVector:
        operand = other if isinstance(other, (int, float)) else list(other)
        raw = self._guard("mul_plain", a.depth, lambda: a.raw * operand)
        return a.derive(raw, cost=1, op="mul_plain")

    def add_plain(self, a: EncVector, other: float | Sequence[float]) -> EncVector:
        operand = other if isinstance(other, (int, float)) else list(other)
        raw = self._guard("add_plain", a.depth, lambda: a.raw + operand)
        return a.derive(raw, cost=0, op="add_plain")

    def polyval(self, a: EncVector, coeffs: Sequence[float]) -> EncVector:
        cost = polyval_depth(coeffs)
        raw = self._guard("polyval", a.depth, lambda: a.raw.polyval(list(coeffs)))
        return a.derive(raw, cost=cost, op=f"polyval(deg={_degree(coeffs)})")

    def square(self, a: EncVector) -> EncVector:
        raw = self._guard("square", a.depth, lambda: a.raw.square())
        return a.derive(raw, cost=1, op="square")

    def sum_slots(self, a: EncVector) -> EncVector:
        """Rotate-and-add across slots. Needs Galois keys in the public context.

        Free in modulus levels, but it collapses the result to a single slot. That
        single slot then has to be broadcast back out to be combined with a
        replicated value, and the broadcast costs a level *and* measurably degrades
        precision, so the training loop uses `matmul_plain` instead.
        """
        raw = self._guard("sum", a.depth, lambda: a.raw.sum())
        return a.derive(raw, cost=0, op="sum")

    def matmul_plain(self, a: EncVector, matrix: list[list[float]]) -> EncVector:
        """Ciphertext times a plaintext matrix. One level; needs Galois keys.

        Used with an all-ones matrix to sum across the batch slots and leave the
        total replicated in every slot - the reduction the weight update needs,
        in the shape the weight update needs it.
        """
        raw = self._guard("matmul_plain", a.depth, lambda: a.raw.mm(matrix))
        return a.derive(raw, cost=1, op="matmul_plain")

    def serialized_size(self, vec: EncVector) -> int:
        """Measured ciphertext size - the library-derived capacity indicator."""
        return len(vec.raw.serialize())


def _degree(coeffs: Sequence[float]) -> int:
    """Highest power with a non-zero coefficient."""
    nonzero = [i for i, c in enumerate(coeffs) if c != 0]
    return max(nonzero) if nonzero else 0


def polyval_depth(coeffs: Sequence[float]) -> int:
    """Levels a polynomial evaluation consumes.

    Measured against TenSEAL: degree 3 consumed two levels and degree 2 consumed
    one, which is `ceil(log2(degree))`. Treated as bookkeeping and cross-checked
    by `CapacityMonitor` rather than taken on faith.
    """
    degree = _degree(coeffs)
    if degree <= 1:
        return 1 if degree == 1 else 0
    cost = 0
    while (1 << cost) < degree:
        cost += 1
    return cost


class DataOwnerZone:
    """The trusted side: owns the secret key, encrypts, decrypts, refreshes.

    Nothing here is ever handed to the training engine except `backend()`, which
    returns a compute-zone object built on a public-only context.
    """

    def __init__(self, params: CKKSParams | None = None, *, generate_galois: bool = True) -> None:
        self.params = params or CKKSParams()
        try:
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.params.poly_modulus_degree,
                coeff_mod_bit_sizes=list(self.params.coeff_mod_bit_sizes),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as an explained ParameterError
            raise ParameterError(
                "SEAL rejected these CKKS parameters: "
                f"poly_modulus_degree={self.params.poly_modulus_degree}, "
                f"coeff_mod_bit_sizes={list(self.params.coeff_mod_bit_sizes)} "
                f"({sum(self.params.coeff_mod_bit_sizes)} bits). Library said: {exc}"
            ) from exc
        ctx.global_scale = 2.0**self.params.scale_bits
        if generate_galois:
            # Needed only for sum()/rotation. Costly at large ring dimensions, so it
            # is optional and the caller is told what it buys.
            ctx.generate_galois_keys()
        self._private = ctx
        self._public = ctx.copy()
        self._public.make_context_public()
        self.refresh_count = 0
        self.refresh_seconds = 0.0

    # -- zones -----------------------------------------------------------

    def backend(self) -> TenSEALBackend:
        """The compute-zone object. Holds no secret key."""
        return TenSEALBackend(self._public, self.params)

    @property
    def public_context(self) -> "ts.Context":
        return self._public

    def secret_key_fingerprint(self) -> str:
        """A short, non-reversible tag so the UI can show a key is present.

        The key material itself is never rendered, logged or serialized
        (Master Prompt 20).
        """
        import hashlib

        digest = hashlib.sha256(self._private.serialize(save_secret_key=True)).hexdigest()
        return f"sha256:{digest[:16]}"

    # -- owner operations ------------------------------------------------

    def encrypt(self, values: Sequence[float], *, lineage: str = "ct") -> EncVector:
        """Encrypt, then hand back a ciphertext that lives in the public context.

        The round trip through `serialize()` is what a real deployment does when it
        ships ciphertexts to a cloud worker, and it is what makes the compute zone
        genuinely unable to decrypt what it is given.
        """
        vec = ts.ckks_vector(self._private, list(values))
        public_side = ts.ckks_vector_from(self._public, vec.serialize())
        return EncVector(raw=public_side, depth=0, lineage=lineage)

    def decrypt(self, vec: EncVector) -> list[float]:
        """Authorized decryption (SRS FR-14, UC-06)."""
        try:
            owner_side = ts.ckks_vector_from(self._private, vec.raw.serialize())
            return list(owner_side.decrypt())
        except Exception as exc:  # noqa: BLE001 - reported with its reason (C3)
            raise CryptoError(
                f"Authorized decryption failed for a ciphertext at depth {vec.depth}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def refresh(self, vec: EncVector) -> tuple[EncVector, float]:
        """Restore computation capacity by decrypting and re-encrypting.

        This is NOT CKKS bootstrapping and the system never says it is. SEAL does
        not implement bootstrapping, so the capacity has to come from the key
        holder. The operation is real, its cost is measured, and it is reported
        under `RefreshKind.CLIENT_AIDED` everywhere it appears.

        The security consequence is stated rather than glossed: the owner sees the
        intermediate value. That is the price of this primitive, and it is the
        reason the OpenFHE backend exists.
        """
        start = time.perf_counter()
        plain = self.decrypt(vec)
        fresh = self.encrypt(plain, lineage=vec.lineage)
        elapsed = time.perf_counter() - start
        self.refresh_count += 1
        self.refresh_seconds += elapsed
        fresh.ops = vec.ops + 1
        fresh.history = [*vec.history[-15:], "refresh(client_aided)"]
        return fresh, elapsed

    def refresh_kind(self) -> RefreshKind:
        return RefreshKind.CLIENT_AIDED


# Key generation costs 20-35 s at n=16384 with a long chain, almost all of it
# Galois keys. Repeating it for every experiment in a dashboard session would
# dominate the demonstration, so contexts are cached per parameter set *in
# memory only*. They are deliberately never written to disk: a cached secret key
# on disk is a secret key that can be committed, backed up or copied, and the
# saving is not worth that (Master Prompt 20).
_ZONE_CACHE: dict[tuple[Any, ...], "DataOwnerZone"] = {}


def get_owner_zone(params: CKKSParams, *, generate_galois: bool = True) -> "DataOwnerZone":
    """A `DataOwnerZone` for these parameters, reusing one from this process if present."""
    key = (
        params.poly_modulus_degree,
        params.coeff_mod_bit_sizes,
        params.scale_bits,
        generate_galois,
    )
    zone = _ZONE_CACHE.get(key)
    if zone is None:
        zone = DataOwnerZone(params, generate_galois=generate_galois)
        _ZONE_CACHE[key] = zone
    return zone


def clear_zone_cache() -> None:
    """Drop cached contexts, and with them their secret keys."""
    _ZONE_CACHE.clear()


def probe_tenseal(params: CKKSParams | None = None) -> Capabilities:
    """Exercise the real TenSEAL CKKS API and report what worked.

    Every entry in the returned `Capabilities` came from a call that either
    returned or raised. Nothing is copied from documentation.
    """
    params = params or CKKSParams(
        poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), scale_bits=40
    )
    caps = Capabilities(
        backend="tenseal",
        library="TenSEAL (Microsoft SEAL)",
        library_version=getattr(ts, "__version__", "unknown"),
        scheme="CKKS",
    )

    owner = DataOwnerZone(params, generate_galois=True)
    be = owner.backend()
    a = owner.encrypt([1.5, 2.5, 3.5])
    b = owner.encrypt([2.0, 2.0, 2.0])

    caps.add(attempt("context_creation", lambda: params.to_dict(), evidence_fmt="{}"))
    caps.add(attempt("encrypt", lambda: f"{len(a.raw.serialize())} bytes", evidence_fmt="{}"))
    caps.add(attempt("decrypt", lambda: [round(v, 6) for v in owner.decrypt(a)], evidence_fmt="{}"))
    caps.add(attempt("add_ct_ct", lambda: owner.decrypt(be.add(a, b))[0], evidence_fmt="{:.6f}"))
    caps.add(attempt("sub_ct_ct", lambda: owner.decrypt(be.sub(a, b))[0], evidence_fmt="{:.6f}"))
    caps.add(attempt("mul_ct_ct", lambda: owner.decrypt(be.mul(a, b))[0], evidence_fmt="{:.6f}"))
    caps.add(
        attempt("mul_ct_plain", lambda: owner.decrypt(be.mul_plain(a, 2.0))[0], evidence_fmt="{:.6f}")
    )
    caps.add(attempt("square", lambda: owner.decrypt(be.square(a))[0], evidence_fmt="{:.6f}"))
    caps.add(
        attempt(
            "polyval_degree3",
            lambda: owner.decrypt(be.polyval(owner.encrypt([0.8, 0.8, 0.8]), [0.5, 0.197, 0, -0.004]))[0],
            evidence_fmt="{:.6f}",
        )
    )
    caps.add(attempt("sum_slots", lambda: owner.decrypt(be.sum_slots(a))[0], evidence_fmt="{:.6f}"))
    caps.add(attempt("serialize", lambda: len(a.raw.serialize()), evidence_fmt="{} bytes"))

    # Accessors TenSEAL does not expose. Recorded as probes so the absence is
    # evidenced rather than merely claimed in prose.
    caps.add(attempt("ciphertext_scale_accessor", lambda: a.raw.scale()))
    caps.add(attempt("ciphertext_level_accessor", lambda: a.raw.level()))
    caps.add(attempt("noise_budget_accessor", lambda: a.raw.noise_budget()))
    caps.add(attempt("native_bootstrap", lambda: a.raw.bootstrap()))

    # The security boundary, checked rather than asserted.
    def _compute_zone_cannot_decrypt() -> str:
        try:
            a.raw.decrypt()
        except Exception as exc:  # noqa: BLE001 - the raise IS the expected outcome
            return f"blocked: {type(exc).__name__}"
        raise AssertionError("compute-zone ciphertext decrypted without a secret key")

    caps.add(attempt("compute_zone_decrypt_blocked", _compute_zone_cannot_decrypt, evidence_fmt="{}"))

    # Measured exhaustion depth: multiply until the library refuses.
    def _measure_depth() -> str:
        vec = owner.encrypt([1.5] * 4)
        ones = [1.0] * 4
        reached = 0
        while reached < 40:
            try:
                vec = be.mul_plain(vec, ones)
                owner.decrypt(vec)
                reached += 1
            except Exception:  # noqa: BLE001 - the refusal is the measurement
                break
        return f"{reached} (params.max_depth predicts {params.max_depth})"

    caps.add(attempt("measured_multiplicative_depth", _measure_depth, evidence_fmt="{}"))

    caps.refresh_mechanism = RefreshKind.CLIENT_AIDED.value
    caps.notes = [
        "Microsoft SEAL does not implement CKKS bootstrapping, and TenSEAL exposes none. "
        "Capacity is restored by client-aided refresh (decrypt and re-encrypt in the data-owner "
        "zone), reported as RefreshKind.CLIENT_AIDED and never described as bootstrapping.",
        "TenSEAL binds no scale(), level() or noise-budget accessor on CKKSVector, so no literal "
        "ciphertext noise value is available and none is displayed. Remaining capacity is tracked "
        "as modulus-chain level, corroborated by measured serialized ciphertext size, and "
        "accompanied by measured precision from a canary probe.",
        "Ciphertexts cross from the owner zone to the compute zone by serialization, so the "
        "compute zone holds a public-only context and cannot decrypt its inputs.",
    ]
    return caps
