# Deploying the dashboard

## Streamlit Community Cloud (free)

**Main file path:** `app/dashboard/Home.py`

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns the
   repository, and authorise Streamlit to read it.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository `Bhavya-git-hub/FHE-TrainNet`, branch `main`, main file
   `app/dashboard/Home.py`.
4. Under **Advanced settings**, set Python to **3.12**. TenSEAL publishes
   manylinux wheels for CPython 3.11-3.14; 3.12 is the safest of those on this
   platform. `runtime.txt` requests it as well, but the dropdown is authoritative.
5. Deploy. The first build installs TenSEAL and takes a few minutes.

There is no API or CLI for Community Cloud deployment - step 1 is a browser OAuth
flow, so it cannot be scripted.

## The memory constraint, and what the app does about it

The free tier gives roughly **1 GB of RAM**. The local demo profile needs far
more than that, and the reason is specific:

| Context (n=16384, 12 primes) | Memory | Key generation |
|---|---|---|
| With Galois rotation keys | **1901 MB** | 23.5 s |
| Without Galois rotation keys | **105 MB** | 8.1 s |

Rotation keys are 95% of the footprint, and exactly one operation in the training
loop needs them: the cross-slot reduction that sums a gradient across a
mini-batch. **A batch of one has nothing to reduce**, so `configs/cloud.yaml`
sets `batch_size: 1`, the trainer skips that operation, and the keys are never
generated.

What that preserves, and what it costs:

- keeps n=16384 and a 35-bit scale, so precision stays honest
- keeps the 10-level chain, so adaptive and fixed still differ (2 steps between
  refreshes, the same structure as the local demo)
- costs one *less* level per step, since the skipped reduction was a level
- processes one sample per step instead of a batch, so the sample and epoch
  counts are reduced to keep a run watchable

Measured end-to-end on the cloud profile: **112 s wall, 545 MB peak**, with
plaintext, fixed and adaptive all reaching 0.833 accuracy and adaptive using
**6 refreshes against 13** - 54% fewer.

### The profile is chosen automatically

`src/runtime.py` reads the **cgroup** memory limit, not `psutil.virtual_memory()`.
In a container the latter reports the *host's* memory, which on a hosted tier is
far larger than the container may actually use - trusting it would select exactly
the profile that gets the process killed. Below 1500 MB the app opens on
`cloud.yaml` and says so on the overview page.

Override with the `FHE_TRAINNET_CONFIG` environment variable (`demo`, `cloud`,
`benchmark`, …) if you want to pin one regardless.

### A learning-rate trap worth knowing

The update is scaled by `lr/B`, so at `B=1` each step is 32x larger than at
`B=32`. Carrying the demo's `learning_rate: 0.3` into the cloud profile drove the
pre-activation past the degree-3 polynomial's monotone limit, the gradient
inverted, and accuracy collapsed to 0.417 with an overflow warning - while the
run still reported "SUCCEEDED". Measured at `B=1`: 0.3 diverges, **0.05 is
stable** at 0.833 with max|z| = 1.16. The trainers now detect and report that
condition, but the configuration avoids it in the first place.

## What ships with the deployment

`results/reference/` contains one real run, committed so a fresh deployment has
measurements to display instead of an empty dashboard. It was recorded locally
with `configs/demo.yaml`; its `notes.origin` field says so. Every figure derived
from it is from that run - nothing in it is synthetic.

## What will not work on the free tier

- **The OpenFHE backend.** It needs Docker, which hosted Streamlit does not
  provide. The page reports the backend unavailable, as it does anywhere without
  a Docker daemon, and never substitutes the client-aided refresh.
- **Long benchmark runs.** `configs/benchmark.yaml` takes minutes and expects the
  batched profile's memory. Run it locally.

## Running it locally instead

```bash
streamlit run app/dashboard/Home.py
```

The full profile is selected automatically when the machine has the memory for
it, and the sidebar can switch batch size to 1 at any time to see the low-memory
path.
