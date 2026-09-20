"""CKKS: key generation, encryption, arithmetic, and the trust boundary.

The boundary tests are the important ones. If `test_compute_zone_cannot_decrypt`
ever passes by accident - because someone handed the training engine a context
with a secret key in it - then the privacy claim this whole project rests on is
false, and everything else still looks fine.
"""

from __future__ import annotations

import math

import pytest

from src.crypto.backend import (
    CKKSParams,
    CapacityExhaustedError,
    MissingKeyError,
    ParameterError,
    RefreshKind,
)
from src.crypto.capabilities import ProbeResult, precision_bits
from src.crypto.tenseal_backend import DataOwnerZone, polyval_depth, probe_tenseal
from tests.conftest import MIN_PRECISION_BITS


# -- parameters ----------------------------------------------------------


def test_default_parameters_are_valid() -> None:
    params = CKKSParams()
    assert params.max_depth == len(params.coeff_mod_bit_sizes) - 2
    assert params.slots == params.poly_modulus_degree // 2


def test_rejects_chain_above_security_ceiling() -> None:
    with pytest.raises(ParameterError, match="above the 218-bit ceiling"):
        CKKSParams(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 40, 60))


def test_rejects_scale_larger_than_rescaling_prime() -> None:
    """The config that runs happily and returns negative bits of precision."""
    with pytest.raises(ParameterError, match="exceeds a rescaling prime"):
        CKKSParams(
            poly_modulus_degree=8192, coeff_mod_bit_sizes=(50, 30, 30, 30, 50), scale_bits=40
        )


def test_rejects_scale_below_usable_floor() -> None:
    """Measured: at scale 2**21, 1.5 * 2.0 came back as 3.52 with no error raised."""
    with pytest.raises(ParameterError, match="below the usable floor"):
        CKKSParams(
            poly_modulus_degree=8192, coeff_mod_bit_sizes=(40, 21, 21, 21, 21, 40), scale_bits=21
        )


def test_rejects_unknown_ring_dimension() -> None:
    with pytest.raises(ParameterError, match="poly_modulus_degree must be one of"):
        CKKSParams(poly_modulus_degree=12345)


def test_rejects_chain_too_short_for_any_depth() -> None:
    with pytest.raises(ParameterError, match="at least 3 primes"):
        CKKSParams(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 60))


# -- keys, encryption, decryption ----------------------------------------


def test_key_generation_produces_both_zones(owner: DataOwnerZone) -> None:
    assert owner.public_context.has_secret_key() is False
    assert owner.secret_key_fingerprint().startswith("sha256:")


def test_encrypt_decrypt_round_trip(owner: DataOwnerZone) -> None:
    values = [1.5, -2.25, 0.125, 3.0]
    recovered = owner.decrypt(owner.encrypt(values))[: len(values)]
    for got, want in zip(recovered, values):
        assert got == pytest.approx(want, abs=1e-4)
        assert precision_bits(got, want) > MIN_PRECISION_BITS


def test_ckks_is_approximate_not_exact(owner: DataOwnerZone) -> None:
    """The round trip must not be bit-exact - CKKS is approximate by design."""
    values = [1.5, 2.5, 3.5]
    recovered = owner.decrypt(owner.encrypt(values))[: len(values)]
    assert any(got != want for got, want in zip(recovered, values))


# -- the trust boundary --------------------------------------------------


def test_compute_zone_cannot_decrypt(owner: DataOwnerZone, backend) -> None:
    """The privacy claim, asserted rather than described."""
    enc = owner.encrypt([1.0, 2.0])
    with pytest.raises(Exception) as excinfo:
        enc.raw.decrypt()
    assert "secret_key" in str(excinfo.value).lower()


def test_backend_decrypt_refuses_with_an_explanation(owner: DataOwnerZone, backend) -> None:
    with pytest.raises(MissingKeyError, match="holds no secret key by design"):
        backend.decrypt(owner.encrypt([1.0]))


def test_backend_encrypt_refuses(owner: DataOwnerZone, backend) -> None:
    with pytest.raises(MissingKeyError, match="data-owner operation"):
        backend.encrypt([1.0])


def test_secret_key_never_appears_in_a_fingerprint(owner: DataOwnerZone) -> None:
    fingerprint = owner.secret_key_fingerprint()
    assert len(fingerprint) < 32
    raw = owner.public_context.serialize()
    assert fingerprint.split(":")[1].encode() not in raw


# -- homomorphic arithmetic ----------------------------------------------


@pytest.mark.parametrize(
    "op,expected",
    [
        ("add", 3.5),
        ("sub", -0.5),
        ("mul", 3.0),
    ],
)
def test_ciphertext_arithmetic(owner: DataOwnerZone, backend, op: str, expected: float) -> None:
    x, y = owner.encrypt([1.5] * 4), owner.encrypt([2.0] * 4)
    result = getattr(backend, op)(x, y)
    got = owner.decrypt(result)[0]
    assert got == pytest.approx(expected, abs=1e-2)
    assert precision_bits(got, expected) > MIN_PRECISION_BITS


def test_the_demonstration_expression(owner: DataOwnerZone, backend) -> None:
    """x * y + x, computed without decrypting x or y."""
    x, y = owner.encrypt([1.5] * 4), owner.encrypt([2.0] * 4)
    result = backend.add(backend.mul(x, y), x)
    assert owner.decrypt(result)[0] == pytest.approx(4.5, abs=1e-2)


def test_addition_consumes_no_levels(owner: DataOwnerZone, backend) -> None:
    x, y = owner.encrypt([1.0] * 4), owner.encrypt([2.0] * 4)
    assert backend.add(x, y).depth == 0


def test_multiplication_consumes_one_level(owner: DataOwnerZone, backend) -> None:
    x, y = owner.encrypt([1.0] * 4), owner.encrypt([2.0] * 4)
    assert backend.mul(x, y).depth == 1


def test_polynomial_activation(owner: DataOwnerZone, backend) -> None:
    from src.model.activation import SIGMOID_DEG3

    x = owner.encrypt([0.8] * 4)
    result = backend.polyval(x, list(SIGMOID_DEG3.coefficients))
    expected = float(SIGMOID_DEG3(0.8))
    assert owner.decrypt(result)[0] == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize(
    "coeffs,expected_depth",
    [([0.5, 0.2], 1), ([0.5, 0.2, 0.0, -0.01], 2), ([0.5, 0.2, 0, -0.01, 0, 0.001], 3)],
)
def test_polyval_depth_matches_measurement(coeffs: list[float], expected_depth: int) -> None:
    assert polyval_depth(coeffs) == expected_depth


def test_matmul_plain_replicates_the_sum(owner: DataOwnerZone, backend) -> None:
    """The reduction the weight update depends on."""
    values = [1.0, 2.0, 3.0, 4.0]
    x = owner.encrypt(values)
    ones = [[1.0] * 4 for _ in range(4)]
    result = owner.decrypt(backend.matmul_plain(x, ones))[:4]
    for slot in result:
        assert slot == pytest.approx(sum(values), abs=1e-2)


# -- capacity exhaustion --------------------------------------------------


def test_exhausting_the_chain_raises_an_explained_error(
    owner: DataOwnerZone, backend, params: CKKSParams
) -> None:
    """The library's bare 'scale out of bounds' becomes something actionable."""
    vec = owner.encrypt([1.5] * 4)
    with pytest.raises(CapacityExhaustedError) as excinfo:
        for _ in range(params.max_depth + 2):
            vec = backend.mul_plain(vec, [1.0] * 4)
    message = str(excinfo.value)
    assert "computation capacity" in message
    assert "Refresh the ciphertext" in message
    assert str(params.max_depth) in message


# -- refresh --------------------------------------------------------------


def test_refresh_restores_capacity_and_preserves_the_value(
    owner: DataOwnerZone, backend
) -> None:
    vec = owner.encrypt([1.25] * 4)
    for _ in range(2):
        vec = backend.mul_plain(vec, [1.0] * 4)
    assert vec.depth == 2
    before = owner.decrypt(vec)[0]

    refreshed, seconds = owner.refresh(vec)
    assert refreshed.depth == 0
    assert seconds > 0
    assert owner.decrypt(refreshed)[0] == pytest.approx(before, abs=1e-3)


def test_refresh_is_not_called_bootstrapping(owner: DataOwnerZone) -> None:
    """The claim this project must never make by accident."""
    kind = owner.refresh_kind()
    assert kind is RefreshKind.CLIENT_AIDED
    assert kind.is_bootstrapping is False
    assert "not CKKS bootstrapping" in kind.label


# -- the capability probe -------------------------------------------------


def test_probe_reports_real_results() -> None:
    caps = probe_tenseal(
        CKKSParams(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), scale_bits=40)
    )
    assert caps.supports("encrypt")
    assert caps.supports("mul_ct_ct")
    assert caps.supports("polyval_degree3")
    assert caps.supports("compute_zone_decrypt_blocked")
    # The four accessors TenSEAL does not bind. If any of these ever becomes
    # supported, the capacity monitor should use it and this test should be updated.
    for missing in (
        "ciphertext_scale_accessor",
        "ciphertext_level_accessor",
        "noise_budget_accessor",
        "native_bootstrap",
    ):
        assert caps.get(missing).result is ProbeResult.UNSUPPORTED
    assert caps.refresh_mechanism == RefreshKind.CLIENT_AIDED.value
    assert not any(p.result is ProbeResult.ERROR for p in caps.probes)


def test_probe_measures_the_depth_it_predicts() -> None:
    caps = probe_tenseal(
        CKKSParams(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), scale_bits=40)
    )
    evidence = caps.get("measured_multiplicative_depth").evidence
    assert evidence.startswith("2 ")
    assert "predicts 2" in evidence


def test_precision_bits_returns_none_when_exact() -> None:
    """'Infinitely precise' is not a number that may appear on a chart."""
    assert precision_bits(1.0, 1.0) is None
    assert precision_bits(1.0, 0.0) is None
    assert precision_bits(1.5 + 1e-6, 1.5) == pytest.approx(-math.log2(1e-6 / 1.5), abs=0.1)
