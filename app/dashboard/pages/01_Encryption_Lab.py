"""Encryption Lab - Master Prompt section 13B, Technical Design demos 1 and 2.

Shows a real record, its real ciphertext, and what comes back on decryption. The
point an evaluator should leave with is that the training engine is handed the
middle column and never the first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from app.dashboard.common import fmt, metric_note, owner_for, page_setup, sidebar_controls
from src.crypto.capabilities import precision_bits
from src.data.loader import load_dataset, prepare

page_setup("Encryption Lab")
config = sidebar_controls()

st.markdown(
    "Encrypt a real record from the selected dataset, inspect the ciphertext, and decrypt "
    "it back. CKKS is *approximate*: what returns is not bit-identical to what went in, and "
    "the difference is measured here rather than described."
)

dataset = load_dataset(config.dataset)
split = prepare(
    dataset, test_fraction=config.test_fraction, seed=config.split_seed,
    n_samples=config.n_samples, feature_range=config.feature_range,
)

row_index = st.number_input(
    "Training record to encrypt", 0, max(0, len(split.y_train) - 1), 0, step=1
)
record = split.x_train[int(row_index)]
label = split.y_train[int(row_index)]

if st.button("Encrypt this record", type="primary"):
    owner = owner_for(config)
    backend = owner.backend()

    t0 = time.perf_counter()
    enc = owner.encrypt(record.tolist(), lineage="demo_record")
    encrypt_seconds = time.perf_counter() - t0

    size_bytes = backend.serialized_size(enc)

    t0 = time.perf_counter()
    decrypted = owner.decrypt(enc)
    decrypt_seconds = time.perf_counter() - t0

    st.session_state["enc_lab"] = {
        "record": record, "label": label, "decrypted": decrypted,
        "encrypt_seconds": encrypt_seconds, "decrypt_seconds": decrypt_seconds,
        "size_bytes": size_bytes,
        "ciphertext_head": enc.raw.serialize()[:48].hex(),
        "fingerprint": owner.secret_key_fingerprint(),
    }

state = st.session_state.get("enc_lab")
if state is None:
    st.info("Press **Encrypt this record** to run real CKKS encryption on it.")
    st.stop()

params = config.ckks_params()
c1, c2, c3 = st.columns(3)
c1.metric("Encryption time", f"{state['encrypt_seconds'] * 1000:.0f} ms")
c2.metric("Ciphertext size", f"{state['size_bytes'] / 1e6:.2f} MB")
c3.metric(
    "Expansion factor",
    f"{state['size_bytes'] / max(1, record.nbytes):.0f}x",
    help=f"{record.nbytes} bytes of plaintext features became {state['size_bytes']} bytes "
         "of ciphertext.",
)

st.subheader("Plaintext in, ciphertext held, plaintext back")
errors = [
    precision_bits(float(d), float(o)) for d, o in zip(state["decrypted"], state["record"])
]
st.dataframe(
    {
        "Feature": split.feature_names,
        "Original value": [f"{v:.8f}" for v in state["record"]],
        "What the compute zone sees": ["<ciphertext>"] * len(state["record"]),
        "Decrypted by the owner": [f"{v:.8f}" for v in state["decrypted"][: len(state["record"])]],
        "Absolute error": [
            f"{abs(d - o):.2e}" for d, o in zip(state["decrypted"], state["record"])
        ],
        "Precision (bits, measured)": [fmt(e, ".1f", "exact") for e in errors],
    },
    use_container_width=True, hide_index=True,
)

st.caption(
    f"Label for this record: **{int(state['label'])}** - encrypted alongside the features "
    "during training, never sent in the clear."
)

st.subheader("Ciphertext metadata")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ring dimension", f"{params.poly_modulus_degree:,}")
m2.metric("Slots available", f"{params.slots:,}")
m3.metric("Multiplicative depth", params.max_depth)
m4.metric("Scale", f"2^{params.scale_bits}")
st.code(f"first 48 bytes of the serialized ciphertext:\n{state['ciphertext_head']}", language=None)
st.caption(
    f"Secret key fingerprint `{state['fingerprint']}` - a truncated hash shown so two key sets "
    "can be told apart. The key itself is never displayed, logged, or written to disk."
)

metric_note()
