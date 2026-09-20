"""The model, and the one update rule both trainers obey.

The plaintext and encrypted trainers must be comparable, which means they must be
running the *same* algorithm - not merely similar ones. The rule is therefore
written down once, here, and each trainer implements exactly it: plaintext with
NumPy, encrypted with homomorphic operations in the same order.

The update is the logistic-regression gradient form

    delta = a - y,    w <- w - (lr/B) * X^T delta

which is the exact gradient of cross-entropy loss with respect to the pre-
activation when `a` is a true sigmoid. Here `a` is a *polynomial approximation*
of the sigmoid, so the rule is the polynomial analogue rather than an exact
gradient. That is a deliberate, standard choice in the encrypted-training
literature: it needs only additions and multiplications, and it avoids evaluating
the activation's derivative, which would cost another modulus level per step. The
plaintext baseline uses the identical rule, so the comparison stays like-for-like
and no accuracy difference can be attributed to the trainers disagreeing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.model.activation import PolynomialActivation, get_activation


@dataclass
class ModelConfig:
    """Shape and hyper-parameters. Every tunable is here, never at a call site."""

    n_features: int
    activation: str = "sigmoid_deg3"
    learning_rate: float = 0.8
    batch_size: int = 32
    epochs: int = 6
    init_scale: float = 0.1
    seed: int = 20260916

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1; got {self.n_features}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive; got {self.learning_rate}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}.")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1; got {self.epochs}.")

    @property
    def activation_fn(self) -> PolynomialActivation:
        return get_activation(self.activation)

    def depth_per_step(self, slots: int | None = None) -> int:
        """Modulus levels one encrypted training step consumes.

        Derived from the operation sequence in `EncryptedTrainer._step`, and
        confirmed by measurement (5 for a degree-3 activation at batch > 1):

            1  z = sum_j w_j * X_j          ciphertext x ciphertext
            +d a = polyval(z)               d = activation depth cost
            0  delta = a - Y                addition is free
            1  delta * Xs_j                 ciphertext x ciphertext
            1  .mm(ones)                    replicated sum, ciphertext x plaintext
            0  w_j <- w_j - grad_j          addition is free

        **A batch of one costs a level less.** The `mm` is only there to sum a
        gradient across the batch slots and hand it back replicated; with a single
        occupied slot that sum is the identity, so the step skips it entirely and
        costs `2 + activation_depth`.

        That is not a micro-optimisation. `mm` is the only operation in the
        training loop needing Galois rotation keys, and those keys measured
        1901 MB against 105 MB for the same context without them - 95% of the
        process footprint. Dropping to a single slot is what lets this run inside
        a 1 GB hosting tier at a ring dimension that keeps the precision honest.
        The trade is throughput: one sample per step instead of a whole batch.

        The adaptive controller uses this to decide whether the *next* step fits
        in what is left, which is what lets it refresh just in time instead of
        early. `CapacityMonitor` checks the prediction against measured ciphertext
        size, so an error here surfaces as a mismatch rather than a wrong answer.
        """
        reduction = 0 if slots == 1 else 1
        return 2 + reduction + self.activation_fn.depth_cost

    @property
    def needs_rotation_keys(self) -> bool:
        """Whether this configuration requires Galois keys at all.

        Only the cross-slot reduction needs them, so a batch of one does not.
        """
        return self.batch_size != 1

    def to_dict(self) -> dict[str, Any]:
        act = self.activation_fn
        return {
            "n_features": self.n_features,
            "activation": self.activation,
            "activation_detail": act.to_dict(),
            "activation_error_vs_sigmoid": act.approximation_error(),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "init_scale": self.init_scale,
            "seed": self.seed,
            "depth_per_step": self.depth_per_step(self.batch_size),
            "needs_rotation_keys": self.needs_rotation_keys,
        }


@dataclass
class Model:
    """Weights and bias. The encrypted trainer holds the same shape, encrypted."""

    weights: np.ndarray
    bias: float

    @classmethod
    def initial(cls, config: ModelConfig) -> "Model":
        """Small random weights from a seeded generator.

        Not zeros. CKKS represents an exact zero with an absolute error floor set
        by the scale rather than by the value, so a zero-initialised encrypted
        weight starts with unbounded *relative* error and the first few steps are
        numerically meaningless. Starting away from zero is ordinary ML practice
        and here it is also a numerical requirement.
        """
        rng = np.random.default_rng(config.seed)
        return cls(
            weights=rng.normal(0.0, config.init_scale, config.n_features),
            bias=float(rng.normal(0.0, config.init_scale)),
        )

    def copy(self) -> "Model":
        return Model(weights=self.weights.copy(), bias=self.bias)

    def forward(self, x: np.ndarray, activation: PolynomialActivation) -> np.ndarray:
        return np.asarray(activation(x @ self.weights + self.bias), dtype=float)

    def predict(self, x: np.ndarray, activation: PolynomialActivation) -> np.ndarray:
        """Threshold at the activation's value at zero, not a hardcoded 0.5.

        A polynomial surrogate need not pass through 0.5 at the origin, and using
        0.5 regardless would quietly bias the accuracy of any activation that does
        not.
        """
        midpoint = float(np.asarray(activation(0.0)))
        return (self.forward(x, activation) > midpoint).astype(int)

    def to_dict(self) -> dict[str, Any]:
        return {"weights": [float(w) for w in self.weights], "bias": float(self.bias)}


def batches(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled mini-batch indices.

    A trailing partial batch is dropped rather than padded. Padding an encrypted
    batch means multiplying by a mask, which costs a modulus level and would make
    the last step of an epoch consume more depth than every other step - an
    avoidable complication in the very accounting the project is measuring.
    """
    order = rng.permutation(n)
    full = n // batch_size
    if full == 0:
        return [order]
    return [order[i * batch_size : (i + 1) * batch_size] for i in range(full)]


@dataclass
class RangeAssessment:
    """How far the pre-activation values have drifted from trustworthy territory.

    Two different things are worth knowing and they are not the same:

    * Some samples sitting outside the polynomial's fitted range is ordinary. The
      approximation is worse for them than the headline error figure suggests,
      which is worth reporting and is not a failure.
    * The values *running away* is a failure, and a silent one. Past its monotone
      limit a polynomial surrogate decreases as its input grows, so the gradient
      inverts and training amplifies the error it should be correcting. Measured
      on the bundled iris split at learning rate 0.8: |z| went 5.6 -> 8.9 -> 42.6
      over three epochs while accuracy fell from 0.94 to 0.10, and nothing raised.

    `diverging` is a heuristic and is labelled as one: it trips when the largest
    pre-activation value exceeds twice the range the activation can be trusted on.
    It deliberately does *not* trip on growth alone. Healthy training starts from
    near-zero weights and grows |z| by a large factor on its way to a good fit;
    an earlier version of this check flagged that as divergence and was wrong.
    The threshold separates the healthy runs from the runaway ones on the bundled
    configurations; it is not a proof.
    """

    max_abs_z: float
    safe_range: float
    fraction_outside: float
    outside_range: bool
    diverging: bool
    message: str | None = None


def assess_activation_range(
    z: np.ndarray,
    activation: PolynomialActivation,
    *,
    fit_limit: float = 3.0,
    first_epoch_max: float | None = None,
) -> RangeAssessment:
    """Measure where the pre-activation values sit relative to the safe range."""
    z = np.asarray(z, dtype=float)
    finite = z[np.isfinite(z)]
    max_abs = float(np.abs(finite).max()) if finite.size else float("inf")
    safe = activation.safe_input_range(fit_limit)
    outside = float(np.mean(np.abs(z) > safe)) if z.size else 0.0

    diverging = (not np.isfinite(max_abs)) or max_abs > 2.0 * safe

    message: str | None = None
    if diverging:
        limit = activation.monotonic_limit
        message = (
            f"The pre-activation value reached {max_abs:.2f}, more than twice the "
            f"+/-{safe:.2f} range where '{activation.name}' can be trusted"
        )
        if limit is not None:
            message += (
                f". Beyond |z|={limit:.2f} this polynomial decreases as its input grows, which "
                "inverts the gradient and makes training diverge rather than converge"
            )
        message += (
            ". The accuracy figures for this run do not represent successful learning. "
            "Lower the learning rate, or choose an activation that is monotone on a wider range."
        )
    elif outside > 0:
        message = (
            f"{outside * 100:.1f}% of samples have |z| beyond the +/-{safe:.2f} range "
            f"'{activation.name}' was fitted on (largest {max_abs:.2f}). Training is healthy; "
            "the approximation error for those samples is larger than the headline figure."
        )

    return RangeAssessment(
        max_abs_z=max_abs,
        safe_range=safe,
        fraction_outside=outside,
        outside_range=outside > 0,
        diverging=diverging,
        message=message,
    )


def squared_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def accuracy(pred_labels: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(pred_labels == target.astype(int)))
