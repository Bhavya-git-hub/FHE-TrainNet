"""Training a neural network on CKKS ciphertexts.

Every value the model touches is encrypted: the features, the labels, the weights
and the bias. The compute zone holds a public-only context, so it performs all of
this arithmetic without the ability to read any of it - verified, not asserted
(`capabilities.compute_zone_decrypt_blocked`).

Packing. Each feature column of a mini-batch becomes one ciphertext with `B`
occupied slots, and each weight is replicated across the same `B` slots. That
choice is deliberate and was arrived at by measurement:

* Keeping weights as single-slot ciphertexts looks cheaper but costs *two* levels
  per multiplication, because TenSEAL replicates a size-1 operand by rotation
  before it can multiply; it also injects around 3e-3 of absolute error into the
  broadcast. Replicated weights cost one level and stay clean.
* The gradient must come back replicated to be subtracted from a replicated
  weight. `dot` would fuse multiply-and-sum into one level but returns a size-1
  result, putting the expensive broadcast back on the subtraction. `mm` against a
  plaintext all-ones matrix performs the same reduction and returns it already
  replicated, for the same one level.

The resulting cost is `3 + activation_depth` levels per training step - five for
the default degree-3 activation - which is what `ModelConfig.depth_per_step`
reports and what the adaptive controller plans against.

The learning rate and batch size are folded into a second, pre-scaled encryption
of the features (`Xs = X * lr/B`). The owner produces it once, at depth 0, which
removes one plaintext multiplication - and therefore one modulus level - from
every single training step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.bootstrapping.policies import Decision, DecisionLog, DecisionRecord, RefreshPolicy
from src.crypto.backend import CapacityExhaustedError, CryptoError, EncVector, RefreshKind
from src.data.loader import Split
from src.model.network import (
    Model,
    ModelConfig,
    accuracy,
    assess_activation_range,
    batches,
    squared_loss,
)
from src.noise.monitor import CanaryProbe, CapacityMonitor


@dataclass
class EncryptedEpochRecord:
    epoch: int
    steps: int
    train_loss: float | None
    train_accuracy: float | None
    test_accuracy: float | None
    elapsed_seconds: float
    levels_remaining: int
    refreshes_so_far: int
    max_abs_z: float = 0.0
    activation_range_exceeded: bool = False
    fraction_outside_range: float = 0.0


@dataclass
class EncryptedResult:
    """Outcome of an encrypted run.

    `status` is one of SUCCEEDED / PARTIAL / FAILED. A run that dies on capacity
    exhaustion is FAILED with its reason, never a quietly shortened success.
    """

    status: str = "SUCCEEDED"
    reason: str | None = None
    model: Model | None = None
    epochs: list[EncryptedEpochRecord] = field(default_factory=list)
    train_seconds: float = 0.0
    steps: int = 0
    encrypted_ops: dict[str, int] = field(default_factory=dict)
    final_train_accuracy: float | None = None
    final_test_accuracy: float | None = None
    final_loss: float | None = None
    setup_seconds: float = 0.0
    refresh_kind: str = RefreshKind.UNSUPPORTED.value
    activation_range_exceeded: bool = False
    diverged: bool = False
    max_abs_z: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "train_seconds": self.train_seconds,
            "setup_seconds": self.setup_seconds,
            "steps": self.steps,
            "encrypted_ops": self.encrypted_ops,
            "encrypted_ops_total": sum(self.encrypted_ops.values()),
            "final_train_accuracy": self.final_train_accuracy,
            "final_test_accuracy": self.final_test_accuracy,
            "final_loss": self.final_loss,
            "refresh_kind": self.refresh_kind,
            "activation_range_exceeded": self.activation_range_exceeded,
            "diverged": self.diverged,
            "max_abs_z": self.max_abs_z,
            "warnings": self.warnings,
            "model": self.model.to_dict() if self.model else None,
            "epochs": [vars(e) for e in self.epochs],
        }


class OpCounter:
    """Counts homomorphic operations by kind. A measured quantity, reported as one."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def bump(self, op: str, n: int = 1) -> None:
        self.counts[op] = self.counts.get(op, 0) + n

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


class EncryptedTrainer:
    """Runs the training loop in the encrypted domain under a refresh policy."""

    def __init__(
        self,
        owner: Any,
        backend: Any,
        config: ModelConfig,
        policy: RefreshPolicy,
        monitor: CapacityMonitor,
        *,
        measure_every: int = 1,
        use_canary: bool = True,
    ) -> None:
        self.owner = owner
        self.backend = backend
        self.config = config
        self.policy = policy
        self.monitor = monitor
        self.log = DecisionLog()
        self.ops = OpCounter()
        self.measure_every = max(1, measure_every)
        self.use_canary = use_canary
        self.refresh_kind = owner.refresh_kind()

    # -- encrypted state -------------------------------------------------

    def _encrypt_weights(self, model: Model, slots: int) -> tuple[list[EncVector], EncVector]:
        """Weights replicated across the batch slots, bias likewise."""
        w = [
            self.owner.encrypt([float(model.weights[j])] * slots, lineage=f"w{j}")
            for j in range(self.config.n_features)
        ]
        b = self.owner.encrypt([float(model.bias)] * slots, lineage="b")
        self.ops.bump("encrypt", len(w) + 1)
        return w, b

    def _decrypt_model(self, w: list[EncVector], b: EncVector) -> Model:
        """Authorized decryption of the trained model (SRS FR-14, UC-06).

        Every slot of a weight ciphertext holds the same number, so slot 0 is the
        weight. The spread across slots is a direct read of accumulated CKKS
        error and is reported by `weight_slot_spread`.
        """
        weights = np.array([self.owner.decrypt(wj)[0] for wj in w], dtype=float)
        bias = float(self.owner.decrypt(b)[0])
        return Model(weights=weights, bias=bias)

    def weight_slot_spread(self, w: list[EncVector]) -> float | None:
        """Max spread between slots that should all hold the same weight.

        A measured, model-level view of accumulated approximation error.
        """
        try:
            spreads = [float(np.ptp(np.asarray(self.owner.decrypt(wj)))) for wj in w]
        except CryptoError:
            return None
        return max(spreads) if spreads else None

    # -- the loop --------------------------------------------------------

    def train(
        self,
        split: Split,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
        max_seconds: float | None = None,
    ) -> EncryptedResult:
        cfg = self.config
        activation = cfg.activation_fn
        first_epoch_max: float | None = None
        coeffs = list(activation.coefficients)
        needed = cfg.depth_per_step()
        result = EncryptedResult(refresh_kind=self.refresh_kind.value)

        setup_started = time.perf_counter()
        model = Model.initial(cfg)
        x, y = split.x_train, split.y_train
        rng = np.random.default_rng(cfg.seed)
        # Derived arithmetically rather than by calling `batches()` for a sample.
        # That call would draw a permutation from `rng`, leaving this trainer one
        # draw ahead of the plaintext one and silently giving the two different
        # batch orders - which would undermine the whole point of the comparison
        # while still producing plausible, nearly-identical numbers.
        slots = cfg.batch_size if len(y) // cfg.batch_size > 0 else len(y)
        if slots > self.monitor.params.slots:
            raise ValueError(
                f"batch_size {slots} exceeds the {self.monitor.params.slots} slots available at "
                f"poly_modulus_degree={self.monitor.params.poly_modulus_degree}."
            )

        self.monitor.calibrate(self.owner, self.backend, slots=slots)
        w_enc, b_enc = self._encrypt_weights(model, slots)
        canary = CanaryProbe(self.owner, self.backend, slots=slots) if self.use_canary else None
        # An all-ones plaintext matrix: multiplying by it sums across the batch
        # slots and leaves the sum replicated in every slot. The 1/B averaging is
        # already folded into the pre-scaled features.
        ones_matrix = np.ones((slots, slots)).tolist()
        scale = cfg.learning_rate / slots
        result.setup_seconds = time.perf_counter() - setup_started

        started = time.perf_counter()
        step = 0
        try:
            for epoch in range(cfg.epochs):
                epoch_started = time.perf_counter()
                epoch_steps = 0
                for idx in batches(len(y), cfg.batch_size, rng):
                    if len(idx) != slots:
                        continue  # keeps every step's depth identical (see network.batches)
                    if max_seconds is not None and time.perf_counter() - started > max_seconds:
                        raise TimeoutError(
                            f"time budget of {max_seconds:.0f}s reached after {step} step(s)"
                        )

                    # -- controller: decide BEFORE spending the next step's depth
                    measure = (step % self.measure_every) == 0
                    reading = self.monitor.record(
                        w_enc[0],
                        label=f"epoch{epoch}.step{step}.pre",
                        iteration=step,
                        epoch=epoch,
                        backend=self.backend,
                        # Reading the canary costs a decryption, so it is sampled
                        # on the same schedule as the size measurement rather than
                        # taken every step regardless.
                        canary=(canary.precision_bits if (canary and measure) else None),
                        needed=needed,
                        measure_size=measure,
                    )
                    decision, reason = self.policy.decide(reading, needed, step)
                    refresh_seconds: float | None = None
                    if decision is Decision.REFRESH:
                        w_enc, b_enc, refresh_seconds = self._refresh(w_enc, b_enc)
                        if canary:
                            canary.reset()
                    self.log.add(
                        DecisionRecord(
                            index=0,
                            iteration=step,
                            epoch=epoch,
                            decision=decision,
                            reason=reason,
                            policy=self.policy.name,
                            levels_remaining=reading.levels_remaining,
                            levels_needed=needed,
                            max_depth=self.monitor.params.max_depth,
                            threshold=self.policy.threshold_text(),
                            refresh_kind=self.refresh_kind.value,
                            precision_bits=reading.precision_bits,
                            serialized_bytes=reading.serialized_bytes,
                            refresh_seconds=refresh_seconds,
                        )
                    )

                    w_enc, b_enc = self._step(
                        w_enc,
                        b_enc,
                        x[idx],
                        y[idx],
                        coeffs,
                        ones_matrix,
                        scale,
                        canary,
                    )
                    step += 1
                    epoch_steps += 1

                    if progress is not None:
                        progress(
                            {
                                "mode": "encrypted",
                                "epoch": epoch + 1,
                                "epochs": cfg.epochs,
                                "step": step,
                                # Two different numbers, both worth seeing: what the
                                # controller decided on, and what the step left behind.
                                "levels_before": reading.levels_remaining,
                                "levels_after": self.monitor.params.max_depth - w_enc[0].depth,
                                "levels_remaining": self.monitor.params.max_depth - w_enc[0].depth,
                                "levels_needed": needed,
                                "max_depth": self.monitor.params.max_depth,
                                "state": reading.state.value,
                                "decision": decision.value,
                                "reason": reason,
                                "refreshes": self.log.refresh_count,
                                "refresh_kind": self.refresh_kind.label,
                                "is_actual_bootstrapping": self.refresh_kind.is_bootstrapping,
                                "precision_bits": reading.precision_bits,
                                "serialized_bytes": reading.serialized_bytes,
                                "elapsed": time.perf_counter() - started,
                            }
                        )

                # The owner decrypts a snapshot to report per-epoch metrics. That is
                # an owner-zone action on the owner's own model, and it is also the
                # only place the pre-activation range can be checked - which matters,
                # because a polynomial activation stops behaving like a sigmoid
                # outside its fitted range and inverts the gradient rather than
                # raising anything.
                snapshot = self._decrypt_model(w_enc, b_enc)
                preds = snapshot.forward(x, activation)
                assessment = assess_activation_range(
                    x @ snapshot.weights + snapshot.bias,
                    activation,
                    first_epoch_max=first_epoch_max,
                )
                if first_epoch_max is None:
                    first_epoch_max = assessment.max_abs_z
                result.max_abs_z = max(result.max_abs_z, assessment.max_abs_z)
                result.activation_range_exceeded |= assessment.outside_range
                if assessment.diverging and not result.diverged:
                    result.diverged = True
                    result.warnings.append(f"Epoch {epoch}: {assessment.message}")
                result.epochs.append(
                    EncryptedEpochRecord(
                        epoch=epoch,
                        steps=epoch_steps,
                        train_loss=squared_loss(preds, y),
                        train_accuracy=accuracy(snapshot.predict(x, activation), y),
                        test_accuracy=accuracy(
                            snapshot.predict(split.x_test, activation), split.y_test
                        ),
                        elapsed_seconds=time.perf_counter() - epoch_started,
                        levels_remaining=self.monitor.params.max_depth - w_enc[0].depth,
                        refreshes_so_far=self.log.refresh_count,
                        max_abs_z=assessment.max_abs_z,
                        activation_range_exceeded=assessment.outside_range,
                        fraction_outside_range=assessment.fraction_outside,
                    )
                )

        except CapacityExhaustedError as exc:
            result.status = "FAILED"
            result.reason = str(exc)
        except TimeoutError as exc:
            result.status = "PARTIAL"
            result.reason = str(exc)
        except CryptoError as exc:
            result.status = "FAILED"
            result.reason = f"{type(exc).__name__}: {exc}"

        result.train_seconds = time.perf_counter() - started
        result.steps = step
        result.encrypted_ops = self.ops.snapshot()

        # The model is decrypted only at the end, by the owner - the encrypted
        # model is what existed throughout training (SRS FR-13).
        if step > 0:
            try:
                result.model = self._decrypt_model(w_enc, b_enc)
                preds = result.model.forward(x, activation)
                result.final_loss = squared_loss(preds, y)
                result.final_train_accuracy = accuracy(result.model.predict(x, activation), y)
                result.final_test_accuracy = accuracy(
                    result.model.predict(split.x_test, activation), split.y_test
                )
            except CryptoError as exc:
                if result.status == "SUCCEEDED":
                    result.status = "PARTIAL"
                result.reason = (
                    f"{result.reason + '; ' if result.reason else ''}"
                    f"final decryption failed: {exc}"
                )
        return result

    # -- one training step ------------------------------------------------

    def _step(
        self,
        w: list[EncVector],
        b: EncVector,
        xb: np.ndarray,
        yb: np.ndarray,
        coeffs: list[float],
        ones_matrix: list[list[float]],
        scale: float,
        canary: CanaryProbe | None,
    ) -> tuple[list[EncVector], EncVector]:
        """One encrypted mini-batch update, mirroring `train_plaintext` exactly."""
        be = self.backend
        f = self.config.n_features

        # The owner encrypts this batch: the features, the pre-scaled features
        # (lr/B folded in at depth 0), and the labels.
        xc = [self.owner.encrypt(xb[:, j].tolist(), lineage=f"x{j}") for j in range(f)]
        xs = [self.owner.encrypt((xb[:, j] * scale).tolist(), lineage=f"xs{j}") for j in range(f)]
        yc = self.owner.encrypt(yb.tolist(), lineage="y")
        self.ops.bump("encrypt", 2 * f + 1)

        # z = sum_j w_j * X_j + b
        z = be.mul(w[0], xc[0])
        self.ops.bump("mul_ct_ct")
        for j in range(1, f):
            z = be.add(z, be.mul(w[j], xc[j]))
            self.ops.bump("mul_ct_ct")
            self.ops.bump("add")
        z = be.add(z, b)
        self.ops.bump("add")

        # a = polynomial activation, delta = a - y
        a = be.polyval(z, coeffs)
        self.ops.bump("polyval")
        delta = be.sub(a, yc)
        self.ops.bump("sub")

        # grad_j = replicated_sum(delta * Xs_j); the lr/B factor is already in Xs.
        new_w: list[EncVector] = []
        for j in range(f):
            prod = be.mul(delta, xs[j])
            self.ops.bump("mul_ct_ct")
            grad = be.matmul_plain(prod, ones_matrix)
            self.ops.bump("matmul_plain")
            new_w.append(be.sub(w[j], grad))
            self.ops.bump("sub")

        grad_b = be.matmul_plain(be.mul_plain(delta, scale), ones_matrix)
        self.ops.bump("mul_plain")
        self.ops.bump("matmul_plain")
        new_b = be.sub(b, grad_b)
        self.ops.bump("sub")

        if canary is not None:
            # Mirror the step's depth onto the canary so its measured precision
            # reflects what the weights have actually been through.
            for _ in range(self.config.depth_per_step()):
                canary.apply_plain(1.0)

        return new_w, new_b

    def _refresh(
        self, w: list[EncVector], b: EncVector
    ) -> tuple[list[EncVector], EncVector, float]:
        """Restore capacity on the model ciphertexts only.

        The feature ciphertexts are re-encrypted fresh for every batch and never
        accumulate depth, so the weights and bias are the only state that needs
        refreshing - `n_features + 1` ciphertexts per event.
        """
        total = 0.0
        refreshed: list[EncVector] = []
        for wj in w:
            new_w, seconds = self.owner.refresh(wj)
            refreshed.append(new_w)
            total += seconds
        new_b, seconds = self.owner.refresh(b)
        total += seconds
        self.ops.bump("refresh", len(w) + 1)
        return refreshed, new_b, total
