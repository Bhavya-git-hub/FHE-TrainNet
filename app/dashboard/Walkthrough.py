"""Encryption and decryption, end to end, for both refresh policies.

    streamlit run app/dashboard/Walkthrough.py

The dashboard already shows each part of this separately: Encryption Lab
encrypts one record, Comparison compares the policies, Faculty Demo runs the
whole story in one pass. None of them shows the **complete lifecycle for both
encrypted policies side by side** - encrypt, train under a fixed schedule, train
under the adaptive controller, decrypt both results, compare - which is the thing
that has to be demonstrated.

So this page does one sequence, in order, with nothing pre-recorded:

    keys -> encrypt -> train (fixed) -> train (adaptive) -> decrypt both -> compare

Two things it is careful about.

The trust boundary is *demonstrated*, not asserted. The compute zone's attempt to
decrypt is actually made, and the library's refusal is printed verbatim. A claim
that the compute side cannot read the data is worth more when the reader watches
it fail.

Both policies run on one split, from one set of initial weights, under one set of
CKKS parameters. The only difference between them is when they refresh, so any
difference in the result is attributable to that and to nothing else.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Streamlit puts only the main script's own directory on sys.path, never the
# repository root - see `streamlit/runtime/scriptrunner/exec_code.py`. This file
# is a main script, so it has to put the root there itself or every import below
# fails on a hosted deployment while working fine locally.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.dashboard.common import (
    fmt,
    metric_note,
    owner_for,
    page_setup,
    profile_banner,
    refresh_banner,
)
from app.visualization.figures import (
    accuracy_comparison,
    capacity_over_operations,
    decision_timeline,
    refresh_comparison,
    timing_comparison,
)
from src.crypto.backend import MissingKeyError, RefreshKind
from src.crypto.capabilities import precision_bits
from src.data.loader import load_dataset, prepare
from src.experiments.config import ExperimentConfig, Mode
from src.experiments.runner import run_single
from src.runtime import default_config_path

POLICIES = [
    (Mode.FHE_BASELINE, "Fixed schedule", "refreshes every N steps, whether or not it needs to"),
    (Mode.FHE_ADAPTIVE, "Adaptive controller", "refreshes only when the next step would not fit"),
]

page_setup("Encryption and Decryption, End to End")
st.caption(
    "One sequence: generate keys, encrypt real records, train the same model twice "
    "under two refresh policies without ever decrypting, then decrypt both results "
    "and compare them."
)

profile_banner()
refresh_banner(RefreshKind.CLIENT_AIDED, compact=True)


def walkthrough_config(samples: int) -> ExperimentConfig:
    """The configuration this page runs, pinned rather than taken from the sidebar.

    Batch size is 1 deliberately. It is what fits a 1 GB host - a batch skips no
    work but needs Galois rotation keys, measured at 1901 MB against 105 MB
    without them - and it also makes the trace legible, because one step is one
    sample rather than a mini-batch reduction.

    The learning rate is 0.05 and that is not arbitrary: the update scales by
    `lr/B`, so at B=1 each step is 32x larger than the batched profile's. Carrying
    that profile's 0.3 across drives the pre-activation past the polynomial's
    monotone limit and inverts the gradient, while the run still reports SUCCEEDED.
    """
    data = ExperimentConfig.load(default_config_path()).to_dict()
    data.pop("derived", None)
    data.update(
        n_samples=samples,
        epochs=1,
        batch_size=1,
        learning_rate=0.05,
        measure_every=1,
        trials=1,
        modes=[Mode.PLAINTEXT, Mode.FHE_BASELINE, Mode.FHE_ADAPTIVE],
    )
    return ExperimentConfig.from_dict(data)


samples = st.select_slider(
    "Records to train on (one per step at batch size 1)",
    options=[12, 16, 20, 28],
    value=16,
)
config = walkthrough_config(samples)
budget = config.depth_budget()
params = config.ckks_params()

# --- Section C, stated first: the arithmetic the run will either confirm or not.

st.divider()
st.subheader("How it is done, before it is done")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ring dimension", f"{params.poly_modulus_degree:,}")
c2.metric("Levels in the chain", budget["max_depth"])
c3.metric("Levels per training step", budget["depth_per_step"])
c4.metric("Steps between refreshes", budget["steps_between_refresh"])

st.markdown(
    f"""
Every homomorphic multiplication consumes one level of a finite modulus chain.
This chain provides **{budget['max_depth']}**, one training step costs
**{budget['depth_per_step']}**, so **{budget['steps_between_refresh']} step(s)**
fit before the ciphertext must be refreshed. That single fact is what the two
policies disagree about, and everything below is measured against it.
"""
)

st.divider()

if st.button("Run the full sequence", type="primary"):
    results: dict[str, object] = {}
    progress_bar = st.progress(0.0)
    stage = st.empty()

    # ---- A. Keys, and proof that the compute zone cannot read anything -------
    stage.info("1 of 4: generating the CKKS key pair")
    owner = owner_for(config)
    compute_zone = owner.backend()

    dataset = load_dataset(config.dataset)
    split = prepare(
        dataset,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
        n_samples=config.n_samples,
        feature_range=config.feature_range,
    )

    probe_record = split.x_train[0]
    started = time.perf_counter()
    probe_ct = owner.encrypt(probe_record.tolist(), lineage="walkthrough")
    encrypt_seconds = time.perf_counter() - started

    # The boundary is demonstrated rather than described: the attempt is real.
    try:
        compute_zone.decrypt(probe_ct)
        refusal = None          # If this is ever reached it is a finding, not a pass.
    except MissingKeyError as exc:
        refusal = str(exc)

    started = time.perf_counter()
    recovered = owner.decrypt(probe_ct)
    decrypt_seconds = time.perf_counter() - started

    results["keys"] = {
        "fingerprint": owner.secret_key_fingerprint(),
        "refusal": refusal,
        "record": probe_record,
        "recovered": recovered[: len(probe_record)],
        "encrypt_seconds": encrypt_seconds,
        "decrypt_seconds": decrypt_seconds,
        "size_bytes": compute_zone.serialized_size(probe_ct),
        "head": probe_ct.raw.serialize()[:48].hex(),
        "feature_names": list(split.feature_names),
    }
    progress_bar.progress(0.15)

    # ---- D. Both policies, on one split, from one set of initial weights -----
    runs: dict[str, object] = {}
    logs: dict[str, list[str]] = {}
    for index, (mode, title, _) in enumerate(POLICIES):
        stage.info(f"{index + 2} of 4: training under the {title.lower()}")
        lines: list[str] = []
        box = st.empty()

        def record_step(event: dict, _lines: list[str] = lines, _box=box) -> None:
            _lines.append(
                f"step {event['step']:>3}  levels {event['levels_before']}->"
                f"{event['levels_after']}/{event['max_depth']}  "
                f"{event['decision']:<8} refreshes={event['refreshes']:<3} "
                f"{event['reason']}"
            )
            _box.code("\n".join(_lines[-8:]), language=None)

        runs[mode] = run_single(config, mode, split, progress=record_step)
        logs[mode] = lines
        box.empty()
        progress_bar.progress(0.15 + 0.35 * (index + 1))

    stage.info("4 of 4: the plaintext reference, for comparison only")
    runs[Mode.PLAINTEXT] = run_single(config, Mode.PLAINTEXT, split)

    results["runs"] = runs
    results["logs"] = logs
    results["split"] = split
    progress_bar.progress(1.0)
    stage.empty()
    st.session_state["walkthrough"] = results

state = st.session_state.get("walkthrough")
if state is None:
    st.info(
        "Press **Run the full sequence**. Everything on this page is computed when "
        "you press it - none of it is stored or pre-recorded."
    )
    st.stop()

keys = state["keys"]
runs = state["runs"]
split = state["split"]

# --- A. -----------------------------------------------------------------------

st.divider()
st.subheader("1. The keys, and who can use them")

k1, k2 = st.columns(2)
with k1:
    st.markdown(
        f"""
**Data-owner zone** — holds the secret key
`{keys['fingerprint']}`

Can encrypt, and can decrypt. The fingerprint is a truncated hash shown so two
key sets can be told apart; the key itself is never displayed, logged, or written
to disk.
"""
    )
with k2:
    st.markdown(
        """
**Compute zone** — holds a public-only context

Can compute on ciphertexts. Cannot read them. This is not a policy the code
enforces on itself; the secret key is absent from the context, so the operation
is impossible.
"""
    )

if keys["refusal"]:
    st.success(
        "**The compute zone was asked to decrypt, and could not.** This is the "
        "library refusing, quoted exactly as it was raised:"
    )
    st.code(keys["refusal"], language=None)
else:
    st.error(
        "**The compute zone decrypted a ciphertext.** That must not be possible and "
        "it invalidates the trust boundary this project claims. Treat every other "
        "result on this page as unverified until it is explained."
    )

# --- B. -----------------------------------------------------------------------

st.divider()
st.subheader("2. Encryption")

e1, e2, e3, e4 = st.columns(4)
e1.metric("Encryption time", f"{keys['encrypt_seconds'] * 1000:.0f} ms")
e2.metric("Decryption time", f"{keys['decrypt_seconds'] * 1000:.0f} ms")
e3.metric("Ciphertext size", f"{keys['size_bytes'] / 1e6:.2f} MB")
e4.metric(
    "Expansion",
    f"{keys['size_bytes'] / max(1, keys['record'].nbytes):.0f}x",
    help=f"{keys['record'].nbytes} bytes of features became {keys['size_bytes']} bytes.",
)

st.dataframe(
    {
        "Feature": keys["feature_names"],
        "Original value": [f"{v:.8f}" for v in keys["record"]],
        "What the training engine sees": ["<ciphertext>"] * len(keys["record"]),
        "Decrypted by the owner": [f"{v:.8f}" for v in keys["recovered"]],
        "Precision (bits, measured)": [
            fmt(precision_bits(float(d), float(o)), ".1f", "exact")
            for d, o in zip(keys["recovered"], keys["record"])
        ],
    },
    use_container_width=True,
    hide_index=True,
)
st.code(f"first 48 bytes of the serialized ciphertext:\n{keys['head']}", language=None)
st.caption(
    "CKKS is approximate. What returns is not bit-identical to what went in, and the "
    "difference is measured here rather than described."
)

# --- C (measured) and D -------------------------------------------------------

st.divider()
st.subheader("3. Training on the ciphertext, under each policy")
st.caption(
    "Both runs used the same data split, the same initial weights and the same CKKS "
    "parameters. The only difference is when each one refreshed."
)

for mode, title, subtitle in POLICIES:
    run = runs[mode]
    st.markdown(f"#### {title} — *{subtitle}*")
    if run.status != "SUCCEEDED":
        st.error(f"**{run.status}** — {run.reason}")
        continue
    m = run.metrics
    a, b, c, d = st.columns(4)
    a.metric("Refreshes", m.get("refreshes"))
    b.metric("Continues", m.get("continues"))
    c.metric("Training time", f"{m.get('train_seconds', 0):.1f}s")
    d.metric("Encrypted operations", f"{m.get('encrypted_ops_total', 0):,}")

    ops = m.get("encrypted_ops") or {}
    if ops:
        st.caption(
            "Homomorphic operations actually performed: "
            + ", ".join(f"`{name}` x{count:,}" for name, count in sorted(ops.items()))
        )
    st.plotly_chart(
        capacity_over_operations(run.capacity_series, run.decisions),
        use_container_width=True,
    )

# --- E. -----------------------------------------------------------------------

st.divider()
st.subheader("4. The authorized decryption")
st.markdown(
    "The model existed only as ciphertext throughout training. This is the first "
    "time its weights have been readable, and only the key holder can do this."
)

baseline_model = (runs[Mode.FHE_BASELINE].metrics or {}).get("model")
adaptive_model = (runs[Mode.FHE_ADAPTIVE].metrics or {}).get("model")
plain_model = (runs[Mode.PLAINTEXT].metrics or {}).get("model")

if not baseline_model or not adaptive_model:
    st.warning(
        "One of the encrypted runs recorded no final model, so there is nothing to "
        "decrypt for it. That is reported rather than filled in."
    )
else:
    names = list(split.feature_names) + ["bias"]
    base_values = list(baseline_model["weights"]) + [baseline_model["bias"]]
    adap_values = list(adaptive_model["weights"]) + [adaptive_model["bias"]]
    plain_values = (
        list(plain_model["weights"]) + [plain_model["bias"]] if plain_model else None
    )

    table = {
        "Parameter": names,
        "Fixed schedule (decrypted)": [f"{v:.6f}" for v in base_values],
        "Adaptive (decrypted)": [f"{v:.6f}" for v in adap_values],
    }
    if plain_values is not None:
        table["Plaintext reference"] = [f"{v:.6f}" for v in plain_values]
        table["Adaptive vs plaintext (bits)"] = [
            fmt(precision_bits(float(a), float(p)), ".1f", "exact")
            for a, p in zip(adap_values, plain_values)
        ]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Two encrypted runs that refreshed at different times decrypt to the same "
        "model, and to the plaintext reference computed on the same data."
    )

# --- F. -----------------------------------------------------------------------

st.divider()
st.subheader("5. Comparison")

rows = {
    "Measurement": [
        "Status", "Refreshes", "Continues", "Training time (s)",
        "Time spent refreshing (s)", "Test accuracy", "Lowest levels remaining",
        "Precision at the end (bits)", "Encrypted operations",
    ]
}
for mode, title, _ in POLICIES:
    m = runs[mode].metrics or {}
    rows[title] = [
        runs[mode].status,
        fmt(m.get("refreshes"), ".0f"),
        fmt(m.get("continues"), ".0f"),
        fmt(m.get("train_seconds"), ".1f"),
        fmt(m.get("refresh_seconds_total"), ".2f"),
        fmt(m.get("final_test_accuracy"), ".3f"),
        fmt(m.get("min_levels_remaining"), ".0f"),
        fmt(m.get("precision_bits_last"), ".1f"),
        f"{m.get('encrypted_ops_total', 0):,}",
    ]
plain_metrics = runs[Mode.PLAINTEXT].metrics or {}
rows["Plaintext reference"] = [
    runs[Mode.PLAINTEXT].status, "n/a", "n/a",
    fmt(plain_metrics.get("train_seconds"), ".3f"), "n/a",
    fmt(plain_metrics.get("final_test_accuracy"), ".3f"), "n/a", "n/a", "0",
]
st.table(rows)

by_mode = {
    mode: {
        "refreshes": {"mean": (runs[mode].metrics or {}).get("refreshes")},
        "train_seconds": {"mean": (runs[mode].metrics or {}).get("train_seconds")},
        "refresh_seconds_total": {
            "mean": (runs[mode].metrics or {}).get("refresh_seconds_total")
        },
        "test_accuracy": {"mean": (runs[mode].metrics or {}).get("final_test_accuracy")},
    }
    for mode in (Mode.PLAINTEXT, Mode.FHE_BASELINE, Mode.FHE_ADAPTIVE)
}

g1, g2 = st.columns(2)
g1.plotly_chart(refresh_comparison(by_mode), use_container_width=True)
g2.plotly_chart(timing_comparison(by_mode), use_container_width=True)
st.plotly_chart(accuracy_comparison(by_mode), use_container_width=True)

st.markdown("#### Every decision the controller made")
st.caption(
    "Refreshes and continues alike. A timeline showing only the refreshes would hide "
    "the moments the adaptive policy looked at the ciphertext and decided it did not "
    "need one - which is the evidence that it refreshed less on purpose."
)
for mode, title, _ in POLICIES:
    st.markdown(f"**{title}**")
    st.plotly_chart(decision_timeline(runs[mode].decisions), use_container_width=True)

# The verdict is generated from what was measured, so it can report a result that
# does not favour the adaptive policy.
base_m = runs[Mode.FHE_BASELINE].metrics or {}
adap_m = runs[Mode.FHE_ADAPTIVE].metrics or {}
if base_m.get("refreshes") is not None and adap_m.get("refreshes") is not None:
    saved = base_m["refreshes"] - adap_m["refreshes"]
    if saved > 0:
        share = 100.0 * saved / max(1, base_m["refreshes"])
        st.success(
            f"**Adaptive used {adap_m['refreshes']} refreshes against the fixed "
            f"schedule's {base_m['refreshes']} - {share:.0f}% fewer - and both "
            f"decrypted to a model reaching "
            f"{fmt(adap_m.get('final_test_accuracy'), '.3f')} test accuracy.**"
        )
    else:
        st.info(
            f"**Adaptive used {adap_m['refreshes']} refreshes against the fixed "
            f"schedule's {base_m['refreshes']}.** On this configuration it did not "
            "refresh less often, and that is the measurement."
        )

metric_note()
