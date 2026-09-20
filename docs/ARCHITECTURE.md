# Architecture

## The shape of the problem

A data owner holds sensitive records and wants a model trained on them without
revealing them. CKKS makes that possible, and imposes one hard constraint in
return: every homomorphic multiplication consumes part of a finite *modulus
chain*, and when the chain is exhausted the ciphertext is unusable. Restoring
capacity is the single most expensive operation in the system.

That turns training into a scheduling problem. Refresh too often and the cost
dominates; refresh too late and the computation fails outright. **When to refresh
is what this project studies.**

## Flow

```
                      DATA-OWNER ZONE (holds the secret key)
  ┌──────────────────────────────────────────────────────────────────┐
  │  dataset.csv → validate → standardise → clip → split             │
  │       │                                                          │
  │       ├─ encrypt features X, pre-scaled features X·(lr/B), y     │
  │       ├─ encrypt initial weights w, bias b                       │
  │       └─ refresh(ct): decrypt + re-encrypt   ◄──────────┐        │
  └───────────────────────┬─────────────────────────────────┼────────┘
                          │ serialize()                     │ request
                          ▼                                 │
                      COMPUTE ZONE (public-only context)     │
  ┌──────────────────────────────────────────────────────────┼───────┐
  │  per training step, all on ciphertexts:                   │      │
  │     z     = Σⱼ wⱼ · Xⱼ + b          ct×ct        1 level  │      │
  │     a     = polyval(z, coeffs)      activation   d levels │      │
  │     delta = a − y                   free         0        │      │
  │     gⱼ    = (delta · Xsⱼ).mm(ones)  ct×ct + mm   2 levels │      │
  │     wⱼ   ← wⱼ − gⱼ                  free         0        │      │
  │                                     total: 3 + d levels   │      │
  │                                                           │      │
  │  CapacityMonitor  → levels (derived) + bytes (measured)    │      │
  │                     + canary precision (measured)          │      │
  │         │                                                  │      │
  │         ▼                                                  │      │
  │  RefreshPolicy.decide(reading, needed, step) ──────────────┘      │
  │     BaselinePolicy | AdaptivePolicy | NoRefreshPolicy             │
  └───────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              Scan of results → metrics, capacity series, decision log
                          │
                          ▼
         DATA-OWNER ZONE: authorized decryption of the final model
```

## The trust boundary is enforced, not described

`DataOwnerZone` holds the only context carrying a secret key. `TenSEALBackend` -
the object the training engine receives - is built from a context produced by
`make_context_public()`. Ciphertexts cross between them by `serialize()`, exactly
as they would cross a network to a cloud worker.

This is verified rather than asserted. `probe_library.py` attempts to decrypt a
compute-zone ciphertext and records the refusal; `test_compute_zone_cannot_decrypt`
asserts it. The training engine *cannot* read its inputs.

The secret key is never printed, logged, serialized into results, or written to
disk. The only place it is serialized at all is inside a SHA-256 fingerprint, so
the interface can show that a key exists without showing the key.

## Packing, and why it is shaped this way

Each feature column of a mini-batch becomes one ciphertext with `B` occupied
slots; each weight is replicated across the same `B` slots. Three alternatives
were measured and rejected:

| Approach | Depth per step | Why not |
|---|---|---|
| Single-slot weights | 5 | TenSEAL replicates a size-1 operand by rotation before multiplying, costing **two** levels per product and injecting ~3e-3 absolute error into the broadcast. |
| `dot` for the gradient | 5 | Fuses multiply-and-sum into one level, but returns a size-1 result - putting the expensive broadcast back on the subtraction. |
| **Replicated weights + `mm(ones)`** | **3 + d** | The reduction returns already replicated, for one level. Chosen. |

The learning rate and batch size are folded into a second, pre-scaled encryption
of the features (`Xs = X·lr/B`). The owner produces it once at depth 0, which
removes one plaintext multiplication - and therefore one modulus level - from
every training step.

## The capacity metric

SEAL exposes no CKKS noise budget, so this system reports none. What it reports
is three indicators, each carrying its provenance:

- **levels remaining** — *derived* from the modulus chain we configured.
- **serialized bytes** — *measured*, 174 kB per level at the shipped parameters
  (n=16384 with 35-bit primes); the step scales with the prime size.
- **canary precision** — *measured*, real CKKS approximation error on a known
  probe value carried through the same operations.

The first two are cross-checked on every reading. A calibration table, built once
per context by multiplying a probe down the chain, turns a measured byte count
back into a measured level; if that disagrees with the derived count, the reading
is flagged `consistent=False` and the run reports a consistency failure. The
depth accounting the controller depends on is therefore checked against the
library rather than trusted.

## The policies

All three implement one interface and are tested without any cryptography, because
the specification requires the controller to be verifiable even where the backend
cannot bootstrap.

- **`BaselinePolicy(interval)`** — refresh every *n* steps, without inspecting the
  ciphertext. The control condition.
- **`AdaptivePolicy(safety_margin, min_precision_bits)`** — refresh when the levels
  remaining will not cover the *next* step. Compares against the depth that step
  actually costs, not a fixed percentage, which is what lets it react when the
  activation degree changes.
- **`NoRefreshPolicy`** — never refresh. Expected to fail, and included so the
  capacity limit can be shown to be real rather than asserted.

Every decision - refresh *and* continue - is appended to `DecisionLog` with the
metric values it saw, the threshold, the reason and the `RefreshKind`. A log that
recorded only the refreshes would hide the evidence that matters most: the moments
the adaptive controller looked and decided it did not need one.

## Module map

| Module | Responsibility |
|---|---|
| `src/crypto/capabilities.py` | Probe result types; `precision_bits` |
| `src/crypto/backend.py` | `CryptoBackend` protocol, `CKKSParams` validation, `RefreshKind`, typed errors |
| `src/crypto/tenseal_backend.py` | `DataOwnerZone` (owner), `TenSEALBackend` (compute), `probe_tenseal` |
| `src/data/loader.py` | CSV loading, validation, standardise/clip/split |
| `src/model/activation.py` | Polynomial activations, depth cost, monotone limit, measured error |
| `src/model/network.py` | `ModelConfig`, `Model`, the shared update rule, range assessment |
| `src/training/plaintext_trainer.py` | The unencrypted mirror |
| `src/training/encrypted_trainer.py` | The encrypted loop |
| `src/noise/monitor.py` | `CapacityMonitor`, `CapacityReading`, `CanaryProbe` |
| `src/bootstrapping/policies.py` | The three policies and `DecisionLog` |
| `src/evaluation/resources.py` | CPU/memory sampling |
| `src/experiments/config.py` | `ExperimentConfig`, reproducibility snapshot |
| `src/experiments/runner.py` | Runs modes, compares, writes results |
| `src/experiments/registry.py` | Reads recorded runs back |
| `app/visualization/figures.py` | Plotly figures (pure functions) |
| `app/dashboard/` | Streamlit pages |

## Deviation from the suggested layout

The SRS and Technical Design both sketch `app/dashboard/`, `app/api/` and
`app/visualization/`. There is no `app/api/` here, and the reason is Streamlit:
the dashboard calls `src.experiments.runner` directly, in-process, so an HTTP
layer between them would add a second process, a serialization boundary and a
schema to keep in step, in exchange for nothing. Both documents allow the layout
to change where the chosen technology requires it, and the pieces the `api`
directory was to contain - the run entry points - exist as `scripts/`, which is
also what makes every result reproducible from a command line rather than only
from a UI.

`docker/` is an addition, holding the optional OpenFHE backend.

## Reproducibility

Every run writes `results/<run_id>/` containing `config.yaml`, `results.json`
(configuration, environment snapshot, all metrics), `summary.csv`, and per-mode
`capacity_*.csv`, `decisions_*.csv`, `epochs_*.csv`. The environment snapshot
records library versions, platform and a configuration fingerprint, and excludes
all key material.

Anything not measurable is recorded as `null`, never `0`. A plaintext run has no
refresh count; recording `0` would read as "it needed none", which is a different
claim from "the question does not apply".
