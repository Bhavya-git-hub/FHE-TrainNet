"""Homomorphic Computation Lab - Master Prompt section 13C and section 8.

Computes on ciphertexts without decrypting the operands, then decrypts only to
verify. The capacity readings before and after each operation are where the cost
of a multiplication becomes visible: additions are free, multiplications are not.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import fmt, metric_note, owner_for, page_setup, sidebar_controls
from src.crypto.capabilities import precision_bits
from src.model.activation import ACTIVATIONS

page_setup("Homomorphic Computation Lab")
config = sidebar_controls()

st.markdown(
    "Two values are encrypted, an expression is evaluated **on the ciphertexts**, and only the "
    "result is decrypted. The operands are never decrypted at any point - the compute zone "
    "could not decrypt them even if it tried, because it holds a public-only context."
)

c1, c2 = st.columns(2)
x_val = c1.number_input("x", value=1.5, step=0.1, format="%.4f")
y_val = c2.number_input("y", value=2.0, step=0.1, format="%.4f")

operations = {
    "x * y + x": lambda be, x, y: be.add(be.mul(x, y), x),
    "x + y": lambda be, x, y: be.add(x, y),
    "x - y": lambda be, x, y: be.sub(x, y),
    "x * y": lambda be, x, y: be.mul(x, y),
    "x^2": lambda be, x, y: be.square(x),
    "x * y * x (depth 2)": lambda be, x, y: be.mul(be.mul(x, y), x),
}
expected = {
    "x * y + x": lambda x, y: x * y + x,
    "x + y": lambda x, y: x + y,
    "x - y": lambda x, y: x - y,
    "x * y": lambda x, y: x * y,
    "x^2": lambda x, y: x * x,
    "x * y * x (depth 2)": lambda x, y: x * y * x,
}

choice = st.selectbox("Operation", list(operations), index=0)
act_name = st.selectbox(
    "...or evaluate a polynomial activation on x", ["(none)"] + list(ACTIVATIONS), index=0
)

if st.button("Compute on ciphertexts", type="primary"):
    owner = owner_for(config)
    be = owner.backend()
    params = config.ckks_params()

    x_enc = owner.encrypt([x_val] * 4, lineage="x")
    y_enc = owner.encrypt([y_val] * 4, lineage="y")
    before = params.max_depth - x_enc.depth

    rows = []
    if act_name != "(none)":
        act = ACTIVATIONS[act_name]
        t0 = time.perf_counter()
        result = be.polyval(x_enc, list(act.coefficients))
        elapsed = time.perf_counter() - t0
        exact = float(act(x_val))
        label = f"{act_name}(x)"
    else:
        t0 = time.perf_counter()
        result = operations[choice](be, x_enc, y_enc)
        elapsed = time.perf_counter() - t0
        exact = float(expected[choice](x_val, y_val))
        label = choice

    decrypted = owner.decrypt(result)[0]
    after = params.max_depth - result.depth

    st.success(f"Computed `{label}` on ciphertexts in **{elapsed * 1000:.1f} ms**.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decrypted result", f"{decrypted:.8f}")
    m2.metric("Exact result", f"{exact:.8f}")
    m3.metric("Absolute error", f"{abs(decrypted - exact):.2e}")
    m4.metric(
        "Precision (measured)",
        fmt(precision_bits(decrypted, exact), ".1f", "exact") + " bits",
    )

    st.subheader("What the operation cost")
    st.table(
        {
            "": ["Levels remaining", "Ciphertext size (measured)"],
            "Before": [before, f"{be.serialized_size(x_enc) / 1e6:.2f} MB"],
            "After": [after, f"{be.serialized_size(result) / 1e6:.2f} MB"],
            "Consumed": [before - after, "-"],
        }
    )
    if before == after:
        st.info(
            "This operation consumed no modulus levels. Additions and subtractions are free in "
            "CKKS; only multiplications (and polynomial evaluation, which is built from them) "
            "consume the chain."
        )
    else:
        st.warning(
            f"This operation consumed **{before - after} level(s)** of the "
            f"{config.ckks_params().max_depth} available. That consumption is what the adaptive "
            "controller exists to manage."
        )

    st.caption(f"Operation history recorded on the result ciphertext: `{result.history}`")

metric_note()
