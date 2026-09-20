# The OpenFHE backend: real CKKS bootstrapping

## Why it exists

The default backend, TenSEAL over Microsoft SEAL, **cannot bootstrap**. SEAL does
not implement CKKS bootstrapping and TenSEAL exposes none; probing
`ciphertext.bootstrap()` raises `AttributeError`. The default backend therefore
restores capacity with a client-aided refresh - the key holder decrypts and
re-encrypts - and says so on every screen.

OpenFHE *does* implement `EvalBootstrap`. This backend exists so the project can
answer "what happens with actual bootstrapping?" with a measurement rather than a
projection.

## Why it is a container and not a dependency

The `openfhe` package on PyPI is published for Linux and macOS only. Its
classifiers list `Operating System :: MacOS :: MacOS X` and
`Operating System :: POSIX :: Linux`, and its release description reads *"This
release requires Ubuntu 24.04"*. There is no Windows wheel, and building it from
source needs a CMake toolchain that the development machine does not have.

So it runs in `ubuntu:24.04` with the wheel installed, and the repository is
**mounted** rather than copied in. That matters: the controller deciding when to
bootstrap inside the container is byte-for-byte the same `src/bootstrapping/policies.py`
the native backend uses. Only the primitive differs, which is exactly the
comparison worth making.

## What it does not do

It is **not** a per-operation bridge. Shipping megabyte ciphertexts across a pipe
for every multiplication would measure the pipe, not the cryptography. The
container runs a whole experiment in-process and returns one JSON document.

It therefore does not run encrypted *training* - it runs the **policy comparison**:
a chain of multiplications costing the same `depth_per_step` a training step
costs, under the fixed policy and the adaptive policy, bootstrapping for real when
the controller says to. What it measures is the policy and the cost of the
primitive.

## Using it

```bash
# 1. Docker must be running.
docker info

# 2. Build the image (slow the first time - Ubuntu plus OpenFHE).
python scripts/openfhe_backend.py --build

# 3. Ask the container what its OpenFHE build supports.
python scripts/openfhe_backend.py --probe

# 4. Run the comparison with real bootstrapping.
python scripts/openfhe_backend.py --run
```

## When it cannot run

This is the part that matters most. The backend has **no fallback path**. If
Docker is missing, not running, the image will not build, or the OpenFHE build
does not expose `EvalBootstrap`, it reports:

```
available : False
reason    : <the specific reason>
refresh   : unsupported -> UNSUPPORTED - no refresh primitive available
```

`BackendStatus.refresh_kind` returns `NATIVE_BOOTSTRAP` only when a probe has
actually found `EvalBootstrap` in the running container. It can never return it
speculatively, and there is no code path that quietly substitutes the TenSEAL
client-aided refresh and reports it as bootstrapping. A dashboard reading
"ACTUAL CKKS BOOTSTRAPPING" over client-aided results is the single most damaging
thing this project could produce, so the type system is arranged to prevent it.

## Expected cost, and why it is not the default

CKKS bootstrapping is expensive in a way that shapes everything around it:

- it needs a large ring dimension (2^16 is typical, against 2^14 for the default
  backend);
- it consumes its own depth budget on top of the levels the computation uses,
  which is why the container computes `GetBootstrapDepth(level_budget, secret_dist)`
  and adds it rather than guessing;
- key generation and `EvalBootstrapSetup` take substantially longer than the
  20-35 s the default backend already needs;
- a single `EvalBootstrap` call is orders of magnitude slower than the
  client-aided refresh it replaces.

That is the honest trade, and it is why the fast, native, laptop-friendly path is
the default and this one is opt-in. The client-aided refresh is cheap and weakens
the threat model; bootstrapping is expensive and does not.

## Status on this machine

Run `python scripts/openfhe_backend.py --probe` to see the current state. At the
time of writing, Docker Desktop is installed (version 29.7.2, WSL2 backend) but
its daemon is not always running, in which case the backend correctly reports
itself unavailable. **Phase A - the TenSEAL path - is the complete, self-sufficient
deliverable; this backend is an addition to it, not a dependency of it.**
