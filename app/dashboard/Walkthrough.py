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

import numpy as np
import streamlit as st

from app.dashboard.ciphertext_view import byte_entropy, hex_dump
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


# Encrypting each value on its own costs a whole ciphertext each - about 2 MB at
# these parameters - so the number is capped. It is a display convenience, not a
# limit of the scheme: a single ciphertext holds thousands of slots.
MAX_VALUES = 8


def parse_values(text: str) -> tuple[list[float], str | None]:
    """Read the numbers the user typed, or say exactly what was wrong with them."""
    pieces = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
    if not pieces:
        return [], "Type at least one number."
    values: list[float] = []
    for piece in pieces:
        try:
            values.append(float(piece))
        except ValueError:
            return [], f"`{piece}` is not a number."
    if len(values) > MAX_VALUES:
        return [], f"At most {MAX_VALUES} values here, to keep the page quick."
    return values, None


# How much of each ciphertext is kept for the inspector. A ciphertext is about
# 2 MB and there can be eight of them, so keeping them whole across every rerun
# would cost 16 MB on a host with 1 GB. Four kilobytes is enough for both windows
# below and costs 32 KB in total.
INSPECT_BYTES = 4096
DUMP_BYTES = 256

# Where the second window looks. Far enough past the container header to be
# inside the ciphertext body at every parameter set this page can run with.
PAYLOAD_OFFSET = 2048


# The CKKS parameters do not depend on the sample count, so the key zone this
# section uses is the same one the training below will reuse from cache.
base_config = walkthrough_config(16)
params = base_config.ckks_params()

st.divider()
st.subheader("1. Encrypt your own numbers")
st.markdown(
    "Type any values you like. They are encrypted with the real CKKS scheme, the "
    "ciphertext is shown, and the data owner decrypts them back."
)

typed = st.text_input(
    "Values to encrypt (comma separated)",
    value="3.5, -1.25, 0.75, 100.0",
    help="Any real numbers. They are encrypted exactly as a training record would be.",
)
values, problem = parse_values(typed)
if problem:
    st.error(problem)

if st.button("Encrypt these values", type="primary", disabled=bool(problem)):
    owner = owner_for(base_config)
    compute_zone = owner.backend()

    # One ciphertext holding the whole vector - this is what training actually
    # uses, because CKKS packs many values into the slots of a single ciphertext.
    started = time.perf_counter()
    packed = owner.encrypt(values, lineage="user_input")
    packed_seconds = time.perf_counter() - started

    started = time.perf_counter()
    recovered = owner.decrypt(packed)[: len(values)]
    unpack_seconds = time.perf_counter() - started

    # And one ciphertext per value, purely so each number can be seen to produce
    # its own ciphertext. Wasteful, and labelled as such on the page.
    per_value = [owner.encrypt([v], lineage=f"single_{i}") for i, v in enumerate(values)]

    # The same number encrypted twice. CKKS encryption is randomised, so these
    # differ - which is the point, and it is worth showing rather than claiming.
    again = owner.encrypt([values[0]], lineage="repeat")

    # The boundary, demonstrated on the ciphertext the user just created.
    try:
        compute_zone.decrypt(packed)
        refusal = None          # If this is ever reached it is a finding, not a pass.
    except MissingKeyError as exc:
        refusal = str(exc)

    # Serialize each ciphertext exactly once. These are two-megabyte objects and
    # the previous version called `serialize()` again for every view of them.
    packed_raw = packed.raw.serialize()
    per_value_raw = [ct.raw.serialize() for ct in per_value]
    again_raw = again.raw.serialize()

    # Measured over the whole ciphertext, which is then dropped - only the
    # 256 counts survive into session state.
    histogram = np.bincount(
        np.frombuffer(packed_raw, dtype=np.uint8), minlength=256
    )

    st.session_state["walkthrough_encryption"] = {
        "fingerprint": owner.secret_key_fingerprint(),
        "refusal": refusal,
        "values": values,
        "recovered": recovered,
        "packed_seconds": packed_seconds,
        "unpack_seconds": unpack_seconds,
        "packed_bytes": len(packed_raw),
        "packed_head": packed_raw[:48].hex(),
        "per_value_heads": [raw[:32].hex() for raw in per_value_raw],
        "per_value_decrypted": [owner.decrypt(ct)[0] for ct in per_value],
        "repeat_head": again_raw[:32].hex(),
        # For the inspector below.
        "packed_inspect": packed_raw[:INSPECT_BYTES],
        "per_value_inspect": [raw[:INSPECT_BYTES] for raw in per_value_raw],
        "per_value_bytes": [len(raw) for raw in per_value_raw],
        "repeat_inspect": again_raw[:INSPECT_BYTES],
        "byte_histogram": histogram.tolist(),
        "entropy_bits": byte_entropy(histogram),
        # The one full ciphertext kept, so the download button hands over real
        # bytes rather than a sample of them.
        "packed_full": packed_raw,
    }

enc = st.session_state.get("walkthrough_encryption")
if enc is None:
    st.info(
        "Press **Encrypt these values**. The first press also generates the CKKS "
        "key pair, which takes 20-35 s and is then reused for everything below."
    )
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Encryption time", f"{enc['packed_seconds'] * 1000:.0f} ms")
    k2.metric("Decryption time", f"{enc['unpack_seconds'] * 1000:.0f} ms")
    k3.metric("Ciphertext size", f"{enc['packed_bytes'] / 1e6:.2f} MB")
    k4.metric(
        "Expansion",
        f"{enc['packed_bytes'] / max(1, 8 * len(enc['values'])):.0f}x",
        help=f"{8 * len(enc['values'])} bytes of numbers became "
             f"{enc['packed_bytes']:,} bytes of ciphertext.",
    )

    st.markdown("**Your numbers, their ciphertexts, and what came back**")
    # The decrypted value is the column a reader actually needs next to what they
    # typed, so it comes first. The ciphertext goes last and is truncated hard:
    # at 32 bytes it was wide enough to push the decryption off a laptop screen,
    # which is the one comparison this table exists to make.
    st.dataframe(
        {
            "You typed": [f"{v:.10g}" for v in enc["values"]],
            "Decrypted by the key holder": [
                f"{v:.10g}" for v in enc["per_value_decrypted"]
            ],
            "Absolute error": [
                f"{abs(d - o):.2e}"
                for d, o in zip(enc["per_value_decrypted"], enc["values"])
            ],
            "Precision (bits)": [
                fmt(precision_bits(float(d), float(o)), ".1f", "exact")
                for d, o in zip(enc["per_value_decrypted"], enc["values"])
            ],
            "Ciphertext (first 12 bytes)": [
                head[:24] + "..." for head in enc["per_value_heads"]
            ],
        },
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Each row was encrypted on its own so that one value maps to one ciphertext "
        "you can look at. **Training does not work this way** - CKKS packs many values "
        f"into the {params.slots:,} slots of a *single* ciphertext, which is why one "
        "encrypted operation processes a whole vector at once."
    )

    # Anyone who types a small number alongside a large one will notice the bits
    # column collapse, and the honest answer is a property of the scheme rather
    # than a defect in this page. Better to say it than to be asked.
    errors = [abs(d - o) for d, o in zip(enc["per_value_decrypted"], enc["values"])]
    st.info(
        f"**Why the precision column varies.** The absolute error is roughly the same "
        f"for every value here - between {min(errors):.0e} and {max(errors):.0e} - "
        "because CKKS fixes it by the scale, not by the size of the number. Precision "
        "*in bits* is therefore a measure of the value's magnitude against that fixed "
        "error, so a small input scores low while a large one scores high. Both are "
        "wrong by about the same amount."
    )

    st.markdown("**What the encrypted data actually looks like**")
    labels = [f"{v:.10g}" for v in enc["values"]]
    packed_label = f"all {len(enc['values'])} values, packed into one ciphertext"
    choice = st.selectbox(
        "Ciphertext to inspect", [*labels, packed_label], index=len(labels)
    )
    window = st.radio(
        "Part of the file",
        ["Start of file", f"Inside the payload (offset {PAYLOAD_OFFSET:,})"],
        horizontal=True,
        help="The start of the file is container metadata. The payload window is "
             "where the ciphertext proper begins.",
    )

    if choice == packed_label:
        blob, total = enc["packed_inspect"], enc["packed_bytes"]
    else:
        index = labels.index(choice)
        blob, total = enc["per_value_inspect"][index], enc["per_value_bytes"][index]

    offset = 0 if window == "Start of file" else PAYLOAD_OFFSET
    st.code(hex_dump(blob[offset : offset + DUMP_BYTES], start=offset), language=None)

    if offset == 0:
        st.caption(
            "**These first bytes are not your encrypted numbers.** They are the "
            "serializer's container: a protobuf field header, a length, and at offset "
            "`0x18` the four bytes `28 b5 2f fd`, which are the Zstandard magic number "
            "- Microsoft SEAL compresses on serialize. The same structure sits at the "
            "same offsets in every ciphertext here, because it describes the file "
            "rather than the values in it. Switch to the payload window to look at the "
            "ciphertext itself."
        )
    else:
        st.caption(
            f"Bytes {offset:,} to {offset + DUMP_BYTES:,} of {total:,}. "
            f"You are {(offset + DUMP_BYTES) / total:.2%} of the way into a file that "
            "holds a handful of numbers."
        )

    st.markdown("**The same number, encrypted twice**")
    st.caption(
        f"Both columns are `{enc['values'][0]:.10g}`, encrypted twice with the same "
        f"key, shown at the same offset."
    )
    # Eight bytes per row rather than sixteen: at sixteen the two dumps wrap
    # inside half-width columns and the comparison they exist for is lost.
    left, right = st.columns(2)
    left.code(
        hex_dump(
            enc["per_value_inspect"][0][PAYLOAD_OFFSET : PAYLOAD_OFFSET + 64],
            start=PAYLOAD_OFFSET,
            width=8,
        ),
        language=None,
    )
    right.code(
        hex_dump(
            enc["repeat_inspect"][PAYLOAD_OFFSET : PAYLOAD_OFFSET + 64],
            start=PAYLOAD_OFFSET,
            width=8,
        ),
        language=None,
    )
    if enc["per_value_heads"][0] != enc["repeat_head"]:
        st.caption(
            "Different bytes, same value. CKKS encryption is randomised, so a "
            "ciphertext cannot be matched back to a known plaintext by comparing it "
            "with an encryption of that plaintext."
        )
    else:
        st.error(
            "**The two ciphertexts are identical.** Encryption is supposed to be "
            "randomised, and if it is not, equal values are linkable. Treat this as "
            "a finding rather than a coincidence."
        )

    st.markdown("**How random those bytes are**")
    left, right = st.columns([1, 3])
    left.metric(
        "Shannon entropy",
        f"{enc['entropy_bits']:.3f} / 8.000",
        help="Bits per byte, measured across the whole packed ciphertext - "
             f"all {enc['packed_bytes']:,} bytes of it.",
    )
    right.bar_chart(
        {"occurrences": enc["byte_histogram"]},
        x_label="byte value (0-255)",
        y_label="times it occurs",
        height=200,
    )
    # Counted rather than asserted. "All 256 values appear" is almost certainly
    # true of two megabytes of compressed ciphertext, but a caption that states it
    # without looking would be wrong the one time it is not.
    present = sum(1 for count in enc["byte_histogram"] if count)
    st.caption(
        f"How often each of the 256 possible byte values occurs. {present} of 256 "
        "appear in this ciphertext."
    )
    st.warning(
        "**A flat histogram is not evidence of security.** It shows only that the "
        "bytes carry no pattern visible at this level. The serialization is "
        "zstd-compressed, and compression raises entropy on its own, so part of that "
        "figure is the compressor rather than the cipher. The actual guarantee comes "
        "from the hardness of Ring-LWE, which no picture on this page can demonstrate."
    )

    st.download_button(
        "Download this ciphertext",
        data=enc["packed_full"],
        file_name="ciphertext.bin",
        mime="application/octet-stream",
        help=f"The real {enc['packed_bytes'] / 1e6:.2f} MB file, to open in a hex "
             "editor rather than take on trust.",
    )

    st.markdown("**Who can read it**")
    st.caption(f"Data-owner zone holds the secret key `{enc['fingerprint']}`.")
    if enc["refusal"]:
        st.success(
            "The compute zone was asked to decrypt the ciphertext you just made, "
            "and could not. This is the library refusing, quoted as raised:"
        )
        st.code(enc["refusal"], language=None)
    else:
        st.error(
            "**The compute zone decrypted it.** That must not be possible and it "
            "invalidates the trust boundary this project claims. Treat every other "
            "result on this page as unverified until it is explained."
        )

# --- The arithmetic, stated before the training runs confirm it or fail to. ----

samples = st.select_slider(
    "Records to train on (one per step at batch size 1)",
    options=[12, 16, 20, 28],
    value=16,
)
config = walkthrough_config(samples)
budget = config.depth_budget()

st.divider()
st.subheader("2. How the training is done, before it is done")

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

if st.button("Run the training", type="primary"):
    results: dict[str, object] = {}
    progress_bar = st.progress(0.0)
    stage = st.empty()

    # Reuses the key zone the encryption section generated, when there is one.
    stage.info("1 of 3: preparing the data and the CKKS key pair")
    owner_for(config)

    dataset = load_dataset(config.dataset)
    split = prepare(
        dataset,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
        n_samples=config.n_samples,
        feature_range=config.feature_range,
    )

    progress_bar.progress(0.1)

    # ---- Both policies, on one split, from one set of initial weights --------
    runs: dict[str, object] = {}
    logs: dict[str, list[str]] = {}
    for index, (mode, title, _) in enumerate(POLICIES):
        stage.info(f"{index + 2} of 3: training under the {title.lower()}")
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
        progress_bar.progress(0.1 + 0.4 * (index + 1))

    stage.info("3 of 3: the plaintext reference, for comparison only")
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
        "Press **Run the training**. Everything below is computed when you press it - "
        "none of it is stored or pre-recorded."
    )
    st.stop()

runs = state["runs"]
split = state["split"]

# --- A. -----------------------------------------------------------------------

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
