"""The unencrypted reference run.

This exists to answer one question: how much did encryption cost us? For that
answer to mean anything, this trainer must differ from the encrypted one in
exactly one respect - the arithmetic is done in the clear. Same initial weights,
same seed, same batch order, same update rule, same polynomial activation. Any
accuracy gap that remains is attributable to CKKS approximation and to nothing
else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.data.loader import Split
from src.model.network import (
    Model,
    ModelConfig,
    accuracy,
    assess_activation_range,
    batches,
    squared_loss,
)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_accuracy: float
    elapsed_seconds: float
    # Largest pre-activation value seen this epoch, and whether it left the range
    # the polynomial can be trusted on. See `PolynomialActivation.monotonic_limit`.
    max_abs_z: float = 0.0
    activation_range_exceeded: bool = False
    fraction_outside_range: float = 0.0


@dataclass
class PlaintextResult:
    model: Model
    epochs: list[EpochRecord] = field(default_factory=list)
    train_seconds: float = 0.0
    final_train_accuracy: float = 0.0
    final_test_accuracy: float = 0.0
    final_loss: float = 0.0
    steps: int = 0
    activation_range_exceeded: bool = False
    diverged: bool = False
    max_abs_z: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "plaintext",
            "train_seconds": self.train_seconds,
            "steps": self.steps,
            "final_train_accuracy": self.final_train_accuracy,
            "final_test_accuracy": self.final_test_accuracy,
            "final_loss": self.final_loss,
            "activation_range_exceeded": self.activation_range_exceeded,
            "diverged": self.diverged,
            "max_abs_z": self.max_abs_z,
            "warnings": self.warnings,
            "model": self.model.to_dict(),
            "epochs": [vars(e) for e in self.epochs],
        }


def train_plaintext(
    split: Split,
    config: ModelConfig,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> PlaintextResult:
    """Train in the clear using the rule defined in `src.model.network`."""
    activation = config.activation_fn
    first_epoch_max: float | None = None
    model = Model.initial(config)
    rng = np.random.default_rng(config.seed)
    x, y = split.x_train, split.y_train
    lr_over_b = config.learning_rate / config.batch_size

    result = PlaintextResult(model=model)
    started = time.perf_counter()

    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        for idx in batches(len(y), config.batch_size, rng):
            xb, yb = x[idx], y[idx]
            # The encrypted trainer performs these four lines as
            # mul / polyval / sub / mul+mm, in this order.
            z = xb @ model.weights + model.bias
            a = np.asarray(activation(z), dtype=float)
            delta = a - yb
            model.weights = model.weights - lr_over_b * (xb.T @ delta)
            model.bias = model.bias - lr_over_b * float(delta.sum())
            result.steps += 1

        train_pred = model.forward(x, activation)
        z_all = x @ model.weights + model.bias
        assessment = assess_activation_range(
            z_all, activation, first_epoch_max=first_epoch_max
        )
        if first_epoch_max is None:
            first_epoch_max = assessment.max_abs_z
        record = EpochRecord(
            epoch=epoch,
            train_loss=squared_loss(train_pred, y),
            train_accuracy=accuracy(model.predict(x, activation), y),
            test_accuracy=accuracy(model.predict(split.x_test, activation), split.y_test),
            elapsed_seconds=time.perf_counter() - epoch_started,
            max_abs_z=assessment.max_abs_z,
            activation_range_exceeded=assessment.outside_range,
            fraction_outside_range=assessment.fraction_outside,
        )
        result.epochs.append(record)
        result.max_abs_z = max(result.max_abs_z, assessment.max_abs_z)
        result.activation_range_exceeded |= assessment.outside_range
        if assessment.diverging and not result.diverged:
            result.diverged = True
            result.warnings.append(f"Epoch {epoch}: {assessment.message}")

        if progress is not None:
            progress(
                {
                    "mode": "plaintext",
                    "epoch": epoch + 1,
                    "epochs": config.epochs,
                    "loss": record.train_loss,
                    "train_accuracy": record.train_accuracy,
                    "test_accuracy": record.test_accuracy,
                    "elapsed": time.perf_counter() - started,
                }
            )

    result.train_seconds = time.perf_counter() - started
    result.final_loss = result.epochs[-1].train_loss
    result.final_train_accuracy = result.epochs[-1].train_accuracy
    result.final_test_accuracy = result.epochs[-1].test_accuracy
    return result
