# Known limitations

Read this before drawing conclusions from anything the dashboard shows. Every
limitation here is a real constraint of the implementation or the library, stated
in the terms an examiner would use, not softened.

The capability claims below are generated evidence, not assertions: run
`python scripts/probe_library.py` and compare with
[`LIBRARY_CAPABILITIES.md`](LIBRARY_CAPABILITIES.md).

---

## 1. There is no CKKS bootstrapping in this system

**Microsoft SEAL does not implement CKKS bootstrapping, and TenSEAL exposes
none.** Probing `ciphertext.bootstrap()` raises `AttributeError`.

Capacity is instead restored by a **client-aided refresh**: the data owner
decrypts the ciphertext and re-encrypts it. This is reported everywhere as
`RefreshKind.CLIENT_AIDED`, whose label reads *"CLIENT-AIDED REFRESH - not CKKS
bootstrapping (Microsoft SEAL does not implement it)"*, and it appears on every
dashboard page that can show a refresh.

**What this does and does not invalidate.**

| Component | Real? |
|---|---|
| CKKS encryption, decryption, homomorphic arithmetic | Yes - real SEAL operations |
| Encrypted training (data, labels, weights, bias all ciphertexts) | Yes |
| The capacity metric and its measurement | Yes |
| The adaptive controller and every decision it makes | Yes |
| Refresh cost being real, measured wall-clock time | Yes |
| The refresh primitive being CKKS bootstrapping | **No** |

**The security consequence, stated plainly.** A client-aided refresh means the
key holder sees the intermediate weights. In the threat model where the data
owner is the one training the model on an untrusted server, that is a real
weakening: the owner must participate in training rather than hand the job over
entirely. It does not leak the training data to the *compute* zone, which still
never holds a secret key - but it is not the same security property as
bootstrapping, and it should not be presented as one.

Real bootstrapping is reachable through OpenFHE, which implements `EvalBootstrap`.
See [`OPENFHE_BACKEND.md`](OPENFHE_BACKEND.md) for the status of that backend and
why it is not the default.

## 2. There is no ciphertext noise measurement, because none is exposed

TenSEAL binds no `scale()`, `level()` or `noise_budget()` accessor on
`CKKSVector`. All three were probed; all three raise `AttributeError`. SEAL
provides a noise budget for BFV but **not for CKKS**.

No quantity in this system is therefore presented as ciphertext noise. What is
presented is *Ciphertext Level / Remaining Computation Capacity*, built from
three indicators, each tagged with how it was obtained:

| Indicator | Provenance | Caveat |
|---|---|---|
| Levels remaining | **derived** | Our own arithmetic against the modulus chain. Exact for CKKS, but not a library reading - so it is cross-checked against the measurement below, and a disagreement is reported as a consistency failure rather than absorbed. |
| Serialized ciphertext size | **measured** | Real, from `serialize()`. About 174 kB per level at the shipped parameters (n=16384, 35-bit primes), measured to within 70 bytes across six runs. It is a *consequence* of the level, not an independent quantity. |
| Canary precision (bits) | **measured** | Real CKKS approximation error. Requires the secret key to read - see limitation 3. |

## 3. The precision indicator is owner-assisted, and the others are not

Levels and ciphertext size are computable by the untrusted compute zone with no
key. **Canary precision is not** - reading it needs a decryption, so it is an
owner-zone measurement.

This matters for how the adaptive policy could be deployed. With
`adaptive_min_precision_bits` unset (the default) the controller uses only
key-free indicators and could genuinely run on the untrusted server. Set it, and
the controller needs a round trip to the key holder on every decision. The
dashboard says which mode is in force; the distinction is real and is not blurred.

## 4. The model is small, and deliberately so

A single-layer network: `z = w·x + b`, then a polynomial activation. No hidden
layer.

This is not a placeholder for something bigger that was not finished. A hidden
layer roughly doubles the multiplicative depth per training step, and at
n=16384 the chain provides 10 levels against the 5 a step already costs. A
two-layer network would need refreshing *within* every step, at which point the
comparison between refresh policies - the thing being studied - has nothing left
to vary. The single layer is what makes the research question answerable on a
laptop.

## 5. The gradient rule is the logistic form, not a derivative of the polynomial

Training uses `delta = a - y`, which is the exact gradient of cross-entropy with
respect to the pre-activation when `a` is a true sigmoid. Here `a` is a polynomial
*approximation* of the sigmoid, so this is the polynomial analogue rather than an
exact gradient of the loss actually being computed.

This is a deliberate, standard choice in the encrypted-training literature: it
needs only additions and multiplications, and it avoids evaluating the
activation's derivative, which would cost another modulus level per step. **The
plaintext baseline uses the identical rule**, so no accuracy difference between
the modes can be attributed to it.

## 6. Polynomial activations stop being sigmoids, and it is silent

The default degree-3 surrogate is monotone only for |z| < 2.82. Beyond that it
*decreases* as its input grows, which inverts the gradient: training amplifies the
error it should correct.

Measured on the bundled iris split at learning rate 0.8: |z| reached 5.6 by
epoch 3 and 42.6 by epoch 5, with test accuracy falling from 0.94 to 0.10. No
exception was raised at any point.

The trainers now measure the pre-activation range every epoch and report both the
fraction of samples outside the trusted range and a divergence warning when the
maximum exceeds twice it. The divergence threshold is a **heuristic**, labelled as
one in the code.

The shipped configurations use learning rates verified to stay inside the safe
range, and they are *not the same rate*: 0.3 for the four-feature demo, 0.05 for
the eight-feature benchmark. With twice as many terms in the pre-activation sum,
|z| grows faster; 0.3 there reached 8.47 and reported divergence. A constant tuned
for one configuration being wrong for the next is the same lesson the refresh
policies are about.

## 7. Features are clipped, and that is a correctness measure

`prepare()` standardizes and then clips to ±`feature_range` (default 3.0), because
the polynomial is only faithful on a bounded interval. The fraction of values
clipped is recorded on the split and reported with the run. On
`breast_cancer_top8` it is around 1.5%.

## 8. CKKS is approximate, and the approximation is not negligible

Measured by the canary probe during an actual demo run (n=16384, 35-bit scale):
**22.8 bits** of agreement on a fresh ciphertext, falling to **12.5 bits** by the
time the chain is consumed. A direct measurement of a single ciphertext-ciphertext
product at the same parameters gives 16.7 bits; the two differ because they carry
different values through different operations, which is itself the point - CKKS
precision is a property of the computation, not a constant of the parameters.

At the test parameters (n=8192, 32-bit scale) a product measures 13.1 bits.

Consequences that show up in practice:

- The encrypted and plaintext trainers diverge by a few parts in a thousand after
  a few steps. The test suite allows 0.5% relative - above the arithmetic's own
  precision, and tight enough to catch the two trainers taking different batch
  orders, which is a bug it did catch.
- **Encrypting an exact zero is pathological.** A zero-valued ciphertext has an
  absolute error floor set by the scale, so its *relative* error is unbounded.
  Weights are therefore initialised away from zero - ordinary ML practice that is
  here also a numerical requirement.
- A scale below 25 bits produces meaningless results that still look successful.
  Measured at scale 2^21: `1.5 * 2.0` returned `3.52`. `CKKSParams` now refuses
  such a configuration rather than running it.

## 9. Performance is not optimised, and the bottleneck is known

The dominant cost is the replicated sum - `mm` against a plaintext all-ones
matrix - which is O(batch) rotations: about 0.6 s at batch 8, 0.9 s at 32, 1.4 s
at 64, and there are `n_features + 1` of them per training step.

Everything runs single-threaded on CPU. No GPU, no batching across ciphertexts
beyond slot packing, no operation fusion. The measured figures are what this
implementation does, not what CKKS training can do in principle.

Key generation costs 20-35 s at n=16384 with a long chain, almost all of it Galois
keys. Contexts are cached per parameter set **in memory only** for the life of the
process. They are deliberately never written to disk - a cached secret key on disk
is a secret key that can be committed or copied.

## 10. The comparison is honest about what it does and does not prove

The adaptive policy is compared against a **fixed-interval** policy. On the shipped
demo configuration adaptive performs 5 refreshes against the fixed policy's 11.

What that does *not* show is that adaptive beats every possible fixed policy. A
fixed interval tuned to exactly the right value performs identically, because in
this configuration every training step consumes the same depth. The honest claim,
which the threshold sweep (`--demo 7`) demonstrates:

- too long an interval and the run **fails** on capacity exhaustion;
- too short and it **wastes** refreshes;
- exactly one interval is optimal, and it must be found by tuning;
- the correct value changes with the activation degree, the feature count and the
  modulus chain - all of which an evaluator can change from the dashboard;
- the adaptive policy is given no interval at all and finds the right behaviour in
  every case.

If a run shows adaptive doing no better, the verdict in the results says so. That
text is generated from the measurements, not written in advance.

## 11. Scale of evaluation

Datasets are 100-569 samples with 2-8 features, binary classification only. Runs
are 1-3 trials. This is a proof of concept sized to be reproducible on a laptop,
not a benchmark suite. Nothing here has been evaluated at a scale that would
support a claim about production workloads.

## 12. Not production software

No authentication, no key management beyond in-process caching, no multi-user
isolation, no audit trail, no protection of the dashboard itself. The threat model
covered is *"the compute zone must not read the training data"*, and only that.
