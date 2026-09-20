"""Shared fixtures.

CKKS key generation is expensive, so the owner zone is session-scoped and uses the
smallest parameter set that still produces trustworthy numbers. It is deliberately
*not* the smallest set that runs: an earlier version of the smoke configuration
used 21-bit primes, which ran fine and returned arithmetic that was wrong in the
second decimal place. Tests built on those parameters would have passed while
proving nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crypto.backend import CKKSParams  # noqa: E402
from src.crypto.tenseal_backend import DataOwnerZone  # noqa: E402
from src.data.loader import load_dataset, prepare  # noqa: E402
from src.experiments.config import ExperimentConfig  # noqa: E402

# Depth 4, ~13 bits of measured precision, ~1 s of key generation.
TEST_PARAMS = CKKSParams(
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=(45, 32, 32, 32, 32, 45),
    scale_bits=32,
)

# Precision floor the crypto tests assert against. Set from the measured value
# (13.1 bits for a ciphertext-ciphertext product at these parameters) with room
# for variation, so a genuine precision regression fails the suite.
MIN_PRECISION_BITS = 9.0


@pytest.fixture(scope="session")
def params() -> CKKSParams:
    return TEST_PARAMS


@pytest.fixture(scope="session")
def owner() -> DataOwnerZone:
    """One data-owner zone for the whole session; key generation is slow."""
    return DataOwnerZone(TEST_PARAMS, generate_galois=True)


@pytest.fixture(scope="session")
def backend(owner: DataOwnerZone):
    """The compute zone. Holds a public-only context and cannot decrypt."""
    return owner.backend()


@pytest.fixture(scope="session")
def smoke_config() -> ExperimentConfig:
    return ExperimentConfig.load(ROOT / "configs" / "fast_smoke.yaml")


@pytest.fixture(scope="session")
def smoke_split(smoke_config: ExperimentConfig):
    dataset = load_dataset(smoke_config.dataset)
    return prepare(
        dataset,
        test_fraction=smoke_config.test_fraction,
        seed=smoke_config.split_seed,
        n_samples=smoke_config.n_samples,
        feature_range=smoke_config.feature_range,
    )
