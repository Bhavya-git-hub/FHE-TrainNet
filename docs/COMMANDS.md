# Every command, in the order you would run them

All commands run from the `FHE_TrainNet` directory with the virtual environment
active.

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
```

## First time

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/probe_library.py      # ~10 s  - MUST pass before anything else
python scripts/build_datasets.py     # writes datasets/*.csv (already committed)
python -m pytest -q                  # ~40 s  - 141 tests
```

`probe_library.py` regenerates `docs/LIBRARY_CAPABILITIES.md` and
`docs/capabilities.json`. Everything the project claims about the cryptography is
checked against that file, so run it after any dependency change.

## The dashboard

```bash
streamlit run app/dashboard/Home.py
```

Then open <http://localhost:8501>. The first encrypted run in a session pays
20–35 s of key generation; it is cached per parameter set for the life of the
process.

## Experiments

```bash
# Demo Mode - plaintext + fixed + adaptive, ~2 minutes
python scripts/run_experiment.py --config configs/demo.yaml

# Benchmark Mode - 3 trials per mode on the larger dataset, ~10 minutes
python scripts/run_experiment.py --config configs/benchmark.yaml

# The control that proves the capacity limit is real. EXPECTED TO FAIL.
python scripts/run_experiment.py --config configs/no_refresh_control.yaml

# Smallest end-to-end run, seconds
python scripts/run_experiment.py --config configs/fast_smoke.yaml
```

Select modes, trials or any configuration field without editing a file:

```bash
python scripts/run_experiment.py --config configs/demo.yaml --modes plaintext fhe_adaptive
python scripts/run_experiment.py --config configs/demo.yaml --trials 3
python scripts/run_experiment.py --config configs/demo.yaml \
    --set epochs=3 learning_rate=0.2 activation=sigmoid_deg5
```

Exit code is non-zero if a mode failed (other than `fhe_no_refresh`, which is
expected to).

## The nine demonstrations

```bash
python scripts/run_all_demos.py --list
python scripts/run_all_demos.py                    # all nine
python scripts/run_all_demos.py --demo 5 --demo 7  # selected
```

| # | Question |
|---|---|
| 1 | How does CKKS protect the data? |
| 2 | Can we compute on encrypted data? |
| 3 | Can we train an ML model while keeping the data encrypted? |
| 4 | How does ciphertext noise/capacity change during training? |
| 5 | When does the adaptive controller decide to bootstrap? |
| 6 | What happens if the threshold changes? |
| 7 | How does adaptive behaviour compare with a baseline policy? |
| 8 | How does workload size affect encrypted training? |
| 9 | Can the authorized user recover the final result? |

**Demo 7 is the one to run if you only run one.** It sweeps the fixed interval and
shows the tuning cliff the adaptive policy avoids.

## Scalability

```bash
python scripts/run_scalability.py --axis samples --sizes 20 40 60 80
python scripts/run_scalability.py --axis features
python scripts/run_scalability.py --axis batch --sizes 8 16 32
```

Each point is a full encrypted training run, so a four-point sweep takes a few
minutes. Sizes that fail are kept in the output and marked on the chart rather
than dropped. Writes `results/scalability/<timestamp>/` with CSV, JSON and a
figure. The **Scalability** dashboard page does the same interactively.

## Reports and figures

```bash
python scripts/make_report.py                  # newest run
python scripts/make_report.py --run-id 20260917-084640-demo_main-af6079
```

Writes nine figures and `REPORT.md` to `reports/<run_id>/`, drawn entirely from
files under `results/<run_id>/`. A figure whose data was never measured is written
as a labelled placeholder saying so, not omitted and not filled in.

## Tests

```bash
python -m pytest -q                       # everything
python -m pytest tests/test_crypto.py -v  # the cryptography and the trust boundary
python -m pytest -k honesty -v            # the claims that could fail silently
python -m pytest -k "not slow" -q
```

## Optional: real CKKS bootstrapping (OpenFHE in Docker)

```bash
docker info                                   # the daemon must be running
python scripts/openfhe_backend.py --probe     # what this machine can do
python scripts/openfhe_backend.py --build     # slow the first time
python scripts/openfhe_backend.py --run       # the comparison, with EvalBootstrap
```

If Docker is not running, `--probe` reports the backend unavailable and exits
non-zero. It never falls back to the TenSEAL client-aided refresh — see
[`OPENFHE_BACKEND.md`](OPENFHE_BACKEND.md).

## Where the output goes

| Path | Contents |
|---|---|
| `results/<run_id>/results.json` | Configuration, environment snapshot, every metric |
| `results/<run_id>/config.yaml` | The exact configuration, re-runnable |
| `results/<run_id>/summary.csv` | One row per mode per trial |
| `results/<run_id>/capacity_<mode>_t<n>.csv` | Every capacity reading |
| `results/<run_id>/decisions_<mode>_t<n>.csv` | Every controller decision, refreshes and continues |
| `results/<run_id>/epochs_<mode>_t<n>.csv` | Per-epoch loss and accuracy |
| `results/demos/` | Output from `run_all_demos.py` |
| `results/scalability/<ts>/` | Scaling sweep CSV, JSON and figure |
| `results/openfhe/` | Output from the OpenFHE backend |
| `reports/<run_id>/` | Figures and `REPORT.md` |
| `docs/LIBRARY_CAPABILITIES.md` | Generated by `probe_library.py` |

`results/` and `reports/` are gitignored. Everything in them regenerates from the
commands above.

## Full validation

```bash
python scripts/validate.py          # fast checks: environment, crypto, honesty, tests
python scripts/validate.py --full   # also runs the demo, the control and the report
```

Implements the Master Prompt section 25 checklist as a program. It is deliberately
suspicious of the documentation - several checks read the README and the generated
capability file and fail if they contradict each other. Exit code 0 only if every
check passes.

**Run it with the virtual environment active.** Using a system Python without the
dependencies produces `ModuleNotFoundError: No module named 'tenseal'` on several
checks; the script now detects that and says so rather than reporting it as a
project failure.

## Full validation by hand, from a clean checkout

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/probe_library.py                                   # capabilities
python -m pytest -q                                               # tests
python scripts/run_experiment.py --config configs/demo.yaml       # the headline result
python scripts/run_experiment.py --config configs/no_refresh_control.yaml  # expected FAIL
python scripts/run_all_demos.py                                   # all nine demos
python scripts/run_scalability.py --axis samples                  # scaling curves
python scripts/make_report.py                                     # figures
streamlit run app/dashboard/Home.py                               # the dashboard
```
