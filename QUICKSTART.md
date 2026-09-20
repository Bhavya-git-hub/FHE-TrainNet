# Quickstart

Three commands to a running dashboard. Everything below has been executed on this
machine; the timings are what it actually took.

## 1. Set up (once, ~3 minutes)

```bash
cd C:\Users\bhavy\Downloads\FHE_TrainNet
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No compiler, no GPU, no Docker. Verified on Python 3.14.3 / Windows 11 x64.

## 2. Prove the cryptography is real (~10 seconds)

```bash
python scripts/probe_library.py
```

This calls every CKKS operation and writes down what happened. Run it before you
believe anything else — it is the evidence behind every claim the project makes.
Expect:

```
  yes   encrypt                        331221 bytes
  yes   mul_ct_ct                      3.000000
  yes   polyval_degree3                0.655552
  NO    ciphertext_scale_accessor      AttributeError: ... has no attribute 'scale'
  NO    noise_budget_accessor          AttributeError: ... has no attribute 'noise_budget'
  NO    native_bootstrap               AttributeError: ... has no attribute 'bootstrap'
  yes   compute_zone_decrypt_blocked   blocked: ValueError
  yes   measured_multiplicative_depth  2 (params.max_depth predicts 2)
```

Those three `NO` rows are why the dashboard says what it says about bootstrapping
and noise. They are not failures.

## 3. Run it

```bash
streamlit run app/dashboard/Home.py
```

Opens at <http://localhost:8501>. Go to **Faculty Demo** and press
**Run the full demonstration** — about 1–3 minutes end to end.

---

## If you only have five minutes

| Do this | See this |
|---|---|
| **Encryption Lab** → *Encrypt this record* | A real record becomes ~2 MB of ciphertext in ~700 ms, and decrypts back agreeing to ~26 bits — not exactly. CKKS is approximate. |
| **Computation Lab** → *Compute on ciphertexts* | `x·y + x` = 4.500028 without ever decrypting `x` or `y`. The multiplication **consumed a modulus level**; switch to `x + y` and nothing is consumed. |
| **Faculty Demo** → *Run the full demonstration* | The whole story: encrypt → compute → train → capacity falls → controller decides → decrypt the model. |
| **Capacity Monitor** | The three indicators with provenance, and the green box confirming derived levels agree with measured ciphertext size. |

## The one chart that matters

On **Capacity Monitor** after a run: the fixed policy is a flat line at 5 levels —
it refreshes every step and never uses the lower half of its budget. The adaptive
policy is a sawtooth down to 0. Same work, less than half the refreshes.

---

## Without the dashboard

```bash
# The headline comparison, ~2 minutes
python scripts/run_experiment.py --config configs/demo.yaml

# Repeated trials, ~15 minutes
python scripts/run_experiment.py --config configs/benchmark.yaml

# Proof the capacity limit is real. THIS RUN IS SUPPOSED TO FAIL.
python scripts/run_experiment.py --config configs/no_refresh_control.yaml

# The nine demonstrations
python scripts/run_all_demos.py --list
python scripts/run_all_demos.py --demo 7     # the one to run if you run one

# Figures + markdown report from the latest run
python scripts/make_report.py

# Scaling curves, every point a real run
python scripts/run_scalability.py --axis samples --sizes 40 60 80 100

# Tests, and the project checked against its own claims
python -m pytest -q
python scripts/validate.py --full
```

## What you should see

Demo configuration, measured:

| Mode | Time | Accuracy | Refreshes |
|---|---|---|---|
| Plaintext reference | 0.00 s | 0.900 | n/a |
| FHE + fixed policy | 59.3 s | 0.900 | 11 |
| **FHE + adaptive** | **34.3 s** | **0.900** | **5** |

`--demo 7`, the argument for the adaptive controller:

| Fixed interval | Refreshes | Time | Outcome |
|---|---|---|---|
| every 1 step | 11 | 78.4 s | works, wastes half its refreshes |
| every 2 steps | 5 | 48.3 s | optimal — *if you knew to pick 2* |
| every 3 steps | — | — | **FAILED**: capacity exhausted |
| every 4 steps | — | — | **FAILED**: capacity exhausted |
| **adaptive** | **5** | **43.8 s** | **given no interval at all** |

---

## Two things that look like bugs and are not

**"Refresh primitive in use: CLIENT-AIDED REFRESH — not CKKS bootstrapping."**
Correct and deliberate. Microsoft SEAL implements no CKKS bootstrapping — the
probe in step 2 proves it. Capacity is restored by the key holder decrypting and
re-encrypting, and every screen says so, because a result produced that way must
never be mistaken for one produced by bootstrapping. The encryption, the encrypted
training, the capacity metric and the controller are all real.
See `docs/LIMITATIONS.md` §1, and the **OpenFHE Backend** page for the container
that does perform real `EvalBootstrap`.

**`no_refresh_control` reports FAILED.** That is the point of it. With no
refreshing the ciphertext exhausts its modulus chain and the library refuses to
continue. If it ever passed, the limit the other policies manage would not be real.

## If something goes wrong

| Symptom | Cause |
|---|---|
| `ParameterError: scale_bits=21 is below the usable floor` | Working as intended — at that scale `1.5 × 2.0` was measured returning `3.52`. Raise `scale_bits`. |
| `coeff_mod_bit_sizes sums to N bits, above the ceiling` | 128-bit security limit for that ring. The message says what to change. |
| First run takes 20–35 s before anything happens | Key generation, mostly Galois keys. Cached per parameter set for the life of the process; later runs in the same session skip it. |
| `FAILED ... exhausted the ciphertext's computation capacity` | The refresh policy did not fire in time. Expected for the control; otherwise lower `baseline_interval` or use adaptive. |
| "the pre-activation value reached X, more than twice the range" | The polynomial activation left the region where it behaves like a sigmoid. Lower the learning rate. That run's accuracy is not meaningful. |
| Docker errors from `openfhe_backend.py` | Expected unless Docker Desktop is running. The backend reports unavailable and never falls back. |

## Where to read next

- `README.md` — the full picture
- `docs/DEMO_SCRIPT.md` — the 5–10 minute faculty walkthrough, with the questions they will ask
- `docs/LIMITATIONS.md` — **read before drawing conclusions**
- `docs/COMMANDS.md` — every command
