"""Polynomial activations, and honest accounting of what they cost and lose.

CKKS evaluates additions and multiplications. It cannot evaluate a sigmoid, so the
sigmoid is replaced by a polynomial - and a polynomial approximation is only a
good sigmoid on the interval it was fitted for. Two consequences are handled here
rather than assumed away:

* `depth_cost` is the number of modulus levels the evaluation consumes, which is
  what forces refreshing during training.
* `approximation_error` measures how far the polynomial actually is from the
  sigmoid over a given range, so the error introduced by this substitution is a
  reported number rather than a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """The function the polynomials approximate. Never used in encrypted code."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass(frozen=True)
class PolynomialActivation:
    """Coefficients in ascending order: `c[0] + c[1]x + c[2]x^2 + ...`.

    The default is the degree-3 least-squares fit to the sigmoid over roughly
    [-5, 5] that recurs throughout the FHE machine-learning literature. It is the
    default because it is cheap (two levels) and well behaved on the range the
    preprocessing guarantees, not because it is the most accurate polynomial
    available.
    """

    name: str
    coefficients: tuple[float, ...]

    @property
    def degree(self) -> int:
        nonzero = [i for i, c in enumerate(self.coefficients) if c != 0]
        return max(nonzero) if nonzero else 0

    @property
    def depth_cost(self) -> int:
        """Modulus levels one evaluation consumes.

        Measured against TenSEAL: degree 3 consumed two levels, degree 2 consumed
        one. That is `ceil(log2(degree))`, which is what this returns.
        """
        degree = self.degree
        if degree <= 1:
            return 1 if degree == 1 else 0
        cost = 0
        while (1 << cost) < degree:
            cost += 1
        return cost

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        """Evaluate in plaintext. The encrypted path uses backend.polyval.

        Overflow is not suppressed as a nuisance - it is expected. When training
        diverges, a high power of a large input genuinely exceeds a float, and
        `inf` is the correct answer. `assess_activation_range` detects exactly
        that, so the warning is silenced here and the condition is reported there.
        """
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            for power, coeff in enumerate(self.coefficients):
                if coeff:
                    out = out + coeff * np.power(x, power)
        return out

    @property
    def monotonic_limit(self) -> float | None:
        """Largest |z| for which this polynomial still increases with z.

        This is the most dangerous property a polynomial activation has, and it is
        invisible until training has already been ruined by it. A sigmoid is
        monotone everywhere; a polynomial fitted to it is not. The default
        degree-3 surrogate `0.5 + 0.23322z - 0.0098z^3` has a derivative that
        turns negative beyond |z| ~ 2.82, and past that point the activation
        *decreases* as its input grows.

        The consequence is not a small approximation error. It is a sign
        inversion in the gradient: training amplifies the error it should be
        correcting, weights run away, and accuracy collapses. Measured on the
        bundled iris split at learning rate 0.8, |z| reached 5.6 by epoch 3 and
        42.6 by epoch 5, with accuracy falling from 0.94 to 0.10 - and nothing
        raised an error at any point.

        Returns `None` for a polynomial that is monotone everywhere on a
        reasonable range (a degree-1 surrogate, for instance).
        """
        derivative = [i * c for i, c in enumerate(self.coefficients)][1:]
        if not derivative or all(c == 0 for c in derivative[1:]):
            return None
        grid = np.linspace(0.0, 20.0, 4001)
        slope = np.zeros_like(grid)
        for power, coeff in enumerate(derivative):
            if coeff:
                slope = slope + coeff * np.power(grid, power)
        negative = np.flatnonzero(slope <= 0)
        if negative.size == 0:
            return None
        return float(grid[negative[0]])

    def safe_input_range(self, fit_limit: float = 3.0) -> float:
        """The input range within which this activation can be trusted.

        The tighter of the range it was fitted on and the range on which it is
        still monotone.
        """
        limit = self.monotonic_limit
        return fit_limit if limit is None else min(fit_limit, limit)

    def approximation_error(self, limit: float = 3.0, samples: int = 401) -> dict[str, float]:
        """How far this polynomial is from a real sigmoid on [-limit, limit].

        Reported in results and shown in the dashboard. A polynomial that is wildly
        wrong at the edges of the operating range is a property of the experiment
        that belongs in the record, not a detail to discover later.
        """
        grid = np.linspace(-limit, limit, samples)
        approx = self(grid)
        exact = np.asarray(sigmoid(grid), dtype=float)
        err = np.abs(approx - exact)
        return {
            "range": float(limit),
            "max_abs_error": float(err.max()),
            "mean_abs_error": float(err.mean()),
            "rms_error": float(math.sqrt(float((err**2).mean()))),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "coefficients": list(self.coefficients),
            "degree": self.degree,
            "depth_cost": self.depth_cost,
            "monotonic_limit": self.monotonic_limit,
            "safe_input_range": self.safe_input_range(),
        }


# These three form a deliberate ladder: one, two and three modulus levels. Because
# depth is the scarce resource, the choice of activation is also a choice about how
# many training steps fit between refreshes, and the experiments treat it as such.
#
# There is no useful degree-2 member. The sigmoid is odd about its centre, so a
# least-squares fit on a symmetric interval drives every even coefficient to zero:
# fitting degree 2 returns the degree-1 polynomial. That is a property of the
# function, not an oversight, and it is why the ladder skips from 1 to 3.
#
# Coefficients are least-squares fits on [-3, 3] (the default `feature_range`),
# reproducible with `fit_polynomial`. The measured error of each is reported by
# `approximation_error` and recorded with every run.

# One level. Cheapest; no non-linearity at all, so it is the control that isolates
# what the non-linear term is worth.
SIGMOID_DEG1 = PolynomialActivation("sigmoid_deg1", (0.5, 0.18023))

# Two levels. The default: the cheapest genuinely non-linear option.
SIGMOID_DEG3 = PolynomialActivation("sigmoid_deg3", (0.5, 0.23322, 0.0, -0.0098))

# Three levels. Roughly five times more accurate than degree 3, and one level more
# expensive per step - the trade the scalability experiment measures.
SIGMOID_DEG5 = PolynomialActivation(
    "sigmoid_deg5", (0.5, 0.24638, 0.0, -0.01662, 0.0, 0.00068)
)

# The degree-3 surrogate quoted throughout the FHE machine-learning literature,
# kept so published numbers can be reproduced against it. Its fit on this range is
# measurably worse than SIGMOID_DEG3 above, which is why it is not the default.
SIGMOID_DEG3_LITERATURE = PolynomialActivation(
    "sigmoid_deg3_literature", (0.5, 0.197, 0.0, -0.004)
)

ACTIVATIONS: dict[str, PolynomialActivation] = {
    a.name: a
    for a in (SIGMOID_DEG1, SIGMOID_DEG3, SIGMOID_DEG5, SIGMOID_DEG3_LITERATURE)
}


def get_activation(name_or_coeffs: str | Sequence[float]) -> PolynomialActivation:
    """Resolve an activation by name, or build one from explicit coefficients."""
    if isinstance(name_or_coeffs, str):
        try:
            return ACTIVATIONS[name_or_coeffs]
        except KeyError:
            raise ValueError(
                f"Unknown activation '{name_or_coeffs}'. Available: {sorted(ACTIVATIONS)}. "
                "You may also pass an explicit coefficient list in ascending power order."
            ) from None
    coeffs = tuple(float(c) for c in name_or_coeffs)
    if not coeffs:
        raise ValueError("An activation needs at least one coefficient.")
    return PolynomialActivation("custom", coeffs)


def fit_polynomial(degree: int, limit: float = 3.0, samples: int = 2001) -> PolynomialActivation:
    """Least-squares fit of the sigmoid on [-limit, limit] at the given degree.

    Offered so an evaluator can ask "what if the approximation were better?" and
    get a real answer from a real fit rather than a hand-tuned constant.
    """
    if degree < 1:
        raise ValueError(f"degree must be at least 1; got {degree}.")
    grid = np.linspace(-limit, limit, samples)
    coeffs = np.polyfit(grid, np.asarray(sigmoid(grid), dtype=float), degree)[::-1]
    return PolynomialActivation(f"sigmoid_fit_deg{degree}", tuple(float(c) for c in coeffs))
