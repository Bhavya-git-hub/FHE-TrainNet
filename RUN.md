# Running FHE-TrainNet

Every command, in the order you would actually run them. All paths are relative to
the repository root.

- [1. Setup](#1-setup)
- [2. Verify the installation](#2-verify-the-installation)
- [3. The dashboard](#3-the-dashboard)
- [4. Experiments from the command line](#4-experiments-from-the-command-line)
- [5. Real CKKS bootstrapping with Docker](#5-real-ckks-bootstrapping-with-docker)
- [6. Reports, figures and export](#6-reports-figures-and-export)
- [7. Profiles and memory](#7-profiles-and-memory)
- [8. When something goes wrong](#8-when-something-goes-wrong)

---

## 1. Setup

Python **3.11 - 3.14**. TenSEAL publishes manylinux/Windows wheels across that
range; 3.12 is the safest on hosted platforms.

```bash
python -m venv .venv
```

Activate it. **Every new terminal needs this** - almost every problem reported
against this project has been a forgotten activation:

```bash
.venv\Scripts\activate
```

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

TenSEAL is pinned exactly (`0.3.18`). It is the only cryptographic dependency and
every capability claim in `docs/capabilities.json` was probed against that build,
so changing it means re-running the probe in step 2.

---

## 2. Verify the installation

Probe what the library actually supports. **Run this before anything else** - it
regenerates `docs/capabilities.json` and `docs/LIBRARY_CAPABILITIES.md`, which the
dashboard reads:

```bash
python scripts/probe_library.py
```

Run the test suite:

```bash
python -m pytest -q
```

Expect **182 passed**. A skip is not a pass - if any appear, `-rs` prints the
reason:

```bash
python -m pytest -rs
```

Run the full acceptance checklist as a program:

```bash
python scripts/validate.py
```

Exit code 0 only if every check passes. Several checks read the README and the
generated capability file and fail if they contradict each other.

---

## 3. The dashboard

```bash
streamlit run app/dashboard/Home.py
```

Opens on <http://localhost:8501>. Twelve pages; the ones worth knowing:

| Page | What it does |
|---|---|
| **Home** | Configuration, probed library capabilities, and *"Why this profile?"* - what the process can actually see |
| **Encryption Lab** | A real record, its ciphertext, and the authorized decryption |
| **Computation Lab** | Arithmetic on ciphertexts; additions free, multiplications cost a level |
| **Run Experiment** | The live training loop - levels draining, the controller deciding |
| **Capacity Monitor** | Derived level count against measured ciphertext size |
| **Bootstrapping Monitor** | Every decision, including the *continues* |
| **Comparison** | Plaintext vs fixed vs adaptive |
| **Faculty Demo** | One guided pass, ~2 minutes, nothing pre-recorded |
| **Moving Target** | Tunes the interval, then breaks it by changing the model |
| **OpenFHE Backend** | Real bootstrapping, when Docker is available |

**The first encrypted action costs 20-35 s of key generation**, cached for the
life of the process. Before a live demonstration, open **Encryption Lab** once so
that cost is already paid.

Pin a profile regardless of what the machine looks like:

```bash
FHE_TRAINNET_CONFIG=cloud streamlit run app/dashboard/Home.py
```

---

## 4. Experiments from the command line

```bash
python scripts/run_experiment.py --config configs/demo.yaml
```

Roughly two minutes: plaintext, fixed and adaptive on identical data.

```bash
python scripts/run_experiment.py --config configs/cloud.yaml
```

The low-memory profile - batch size 1, no rotation keys.

```bash
python scripts/run_experiment.py --config configs/no_refresh_control.yaml
```

**This one is expected to FAIL.** It is the control that proves the capacity limit
is real: without refreshing, the ciphertext genuinely runs out and training stops.
A non-zero exit code here is the correct result.

Override any field without editing a file:

```bash
python scripts/run_experiment.py --config configs/demo.yaml --set epochs=3 learning_rate=0.2
```

The nine demonstrations, and the one to run if you only run one:

```bash
python scripts/run_all_demos.py --demo 7
```

Scaling sweep - each point is a full encrypted run, so allow a few minutes:

```bash
python scripts/run_scalability.py --axis samples --sizes 20 40 60 80
```

---

## 5. Real CKKS bootstrapping with Docker

TenSEAL and Microsoft SEAL implement **no CKKS bootstrapping**. The default
backend restores capacity with a client-aided refresh and says so on every screen.
This container is the one that performs the real operation, through OpenFHE's
`EvalBootstrap`. It is Linux-only, hence Docker.

### Start Docker

Docker Desktop must be running, not merely installed. If `docker info` fails, the
privileged service is stopped - start it from an **Administrator** shell:

```bash
Start-Service com.docker.service
```

Then launch Docker Desktop and wait for the whale to stop animating.

### Raise Docker's memory limit first

**Docker Desktop -> Settings -> Resources -> Memory -> at least 6 GB**, then Apply
& Restart.

This is not optional. Bootstrapping key generation at ring 65536 was measured
holding **3.13 GB** steadily, and a 3.66 GB limit killed it after 898 seconds of
work with no output at all. Check what you have:

```bash
docker info --format "{{.MemTotal}}"
```

### Check what the machine can do

```bash
python scripts/openfhe_backend.py --probe
```

When it is working this reports `EvalBootstrap: True` and
`refresh kind: ACTUAL CKKS BOOTSTRAPPING (library EvalBootstrap)`. When Docker is
absent it reports the backend unavailable and exits non-zero - it never falls back
to the client-aided refresh and never calls it bootstrapping.

### Build the image

```bash
python scripts/openfhe_backend.py --build
```

Slow the first time: it compiles OpenFHE 1.5.1 from source. Allow 15-30 minutes.

### Run the comparison

```bash
python scripts/openfhe_backend.py --run --scaling-mod-size 40 --steps 4
```

**`--scaling-mod-size 40` matters.** The default of 45 is refused. Bootstrapping
reserves **18 levels** on top of the 10 the training needs, giving a
multiplicative depth of 28, and at 45 bits per level that exceeds what ring 65536
permits at the 128-bit security level. 40 bits fits, keeping all ten usable levels
and the same two-steps-between-bootstraps structure as the TenSEAL side.

Measured inside the container, if you want to choose your own:

| Level budget | Levels reserved |
|---|---|
| `[1, 1]` | 16 |
| `[2, 2]` | 18 (default) |
| `[3, 3]` | 20 |

**Do not lower `--ring-dim` below 65536.** Every smaller ring is refused for
bootstrapping at the 128-bit level - even the minimum bootstrap depth of 20 levels
needs that much modulus, and OpenFHE names 65536 itself as the requirement. The
only way to make a smaller ring work is to weaken the security level, which is not
a trade this project makes.

Expect **20+ minutes** and high CPU. Real bootstrapping is genuinely this
expensive, which is the finding, not a defect.

---

## 6. Reports, figures and export

```bash
python scripts/make_report.py
```

Nine figures and `REPORT.md` for the most recent run, written to
`reports/<run_id>/`. A figure whose data was never measured is written as a
labelled placeholder saying so - never omitted, never filled in.

A specific run:

```bash
python scripts/make_report.py --run-id 20260922-111001-demo-902679
```

Output lives under `results/<run_id>/` - `results.json`, `config.yaml`,
`summary.csv`, plus per-mode capacity, decision and epoch CSVs. `results/` and
`reports/` are gitignored; everything in them regenerates from the commands above.

---

## 7. Profiles and memory

Two shipped profiles, chosen automatically:

| Profile | Batch | Rotation keys | Measured peak |
|---|---|---|---|
| `demo.yaml` | 32 | yes - 1901 MB | **2196 MB** |
| `cloud.yaml` | 1 | none - 105 MB | **545 MB** |

Rotation keys are 95% of the footprint and exactly one operation needs them: the
cross-slot reduction that sums a gradient across a mini-batch. A batch of one has
nothing to reduce, so that operation is skipped, the keys are never generated, and
a step costs one level *less*.

Selection reads the **cgroup** limit, not `psutil` - in a container the latter
reports the host's memory. On Streamlit Community Cloud the low-memory profile is
chosen outright, because no figure that process can read is the one being
enforced. The Home page's *"Why this profile?"* panel shows every input to that
decision.

A learning-rate trap worth knowing: the update scales by `lr/B`, so at `B=1` each
step is 32x larger than at `B=32`. Carrying `learning_rate: 0.3` into the cloud
profile drives the pre-activation past the degree-3 polynomial's monotone limit,
the gradient inverts, and accuracy collapses to 0.417 - while the run still
reports SUCCEEDED. At `B=1`, **0.05 is stable**.

---

## 8. When something goes wrong

**`ModuleNotFoundError: No module named 'tenseal'`** - the virtual environment is
not active. Every script detects this and names the path to activate.

**The dashboard says no experiment has been run** - a fresh checkout has no runs.
Press **Run Experiment**, or run one from the command line. `results/reference/`
ships one real recorded run so a fresh deployment has something to display.

**A mode reports FAILED with `CapacityExhaustedError`** - that is a legitimate
result, not a crash, and for `no_refresh_control.yaml` it is the expected one.
The message names the operation, the depth reached and the chain that provided it.

**The OpenFHE container is killed with no output** - the memory limit. See
section 5; raise Docker's allowance rather than lowering the ring dimension.

**Accuracy collapses and a run still says SUCCEEDED** - the learning rate has
pushed the pre-activation outside the polynomial's monotone range. The trainers
detect and report this; see section 7.
