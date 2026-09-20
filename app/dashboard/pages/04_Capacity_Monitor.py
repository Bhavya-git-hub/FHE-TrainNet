"""Noise / Capacity Monitor - Master Prompt section 13E.

Three indicators of the same underlying thing, each labelled with how it was
obtained. The derived level count and the measured ciphertext size are plotted
together on purpose: when they agree, the depth accounting is confirmed by the
library, and when they disagree the page says so rather than picking one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import fmt, metric_note, page_setup, require_run, sidebar_controls
from app.visualization.figures import (
    capacity_over_operations,
    ciphertext_size_over_operations,
    precision_over_operations,
)
from src.experiments.registry import load_series
from src.noise.monitor import METRIC_NAME

page_setup("Capacity Monitor")
sidebar_controls()
payload = require_run()

st.markdown(f"### {METRIC_NAME}")
metric_note()

encrypted_runs = [r for r in payload["runs"] if r["mode"] != "plaintext"]
if not encrypted_runs:
    st.info("This run contains no encrypted mode, so there is no capacity data to show.")
    st.stop()

labels = [f"{r['mode']} (trial {r['trial']})" for r in encrypted_runs]
choice = st.selectbox("Run", labels, index=0)
selected = encrypted_runs[labels.index(choice)]

series = load_series(payload["run_id"], "capacity", selected["mode"], selected["trial"])
decisions = load_series(payload["run_id"], "decisions", selected["mode"], selected["trial"])
capacity = selected["metrics"].get("capacity", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Chain depth", capacity.get("max_depth", "-"))
c2.metric("Lowest level reached", fmt(capacity.get("min_levels_remaining"), "d"))
c3.metric(
    "Precision at start",
    fmt(capacity.get("precision_bits_first"), ".1f") + " bits",
    help="Measured on the canary probe, not a library noise reading.",
)
c4.metric(
    "Precision at end",
    fmt(capacity.get("precision_bits_last"), ".1f") + " bits",
    delta=(
        None
        if capacity.get("precision_bits_first") is None or capacity.get("precision_bits_last") is None
        else f"{capacity['precision_bits_last'] - capacity['precision_bits_first']:.1f}"
    ),
)

failures = capacity.get("consistency_failures", 0)
if failures:
    st.error(
        f"**{failures} consistency failure(s).** The derived level count disagreed with the "
        "measured ciphertext size. The depth accounting is wrong and every decision after the "
        "first disagreement is suspect."
    )
    for detail in capacity.get("inconsistency_detail", []):
        st.code(detail, language=None)
else:
    st.success(
        "**Derived level count agrees with measured ciphertext size at every reading.** "
        "The depth accounting the controller relies on is confirmed by the library itself."
    )

st.plotly_chart(
    capacity_over_operations(
        series, decisions, threshold=selected["metrics"].get("depth_per_step")
    ),
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.plotly_chart(ciphertext_size_over_operations(series), use_container_width=True)
    if capacity.get("bytes_per_level"):
        st.caption(
            f"Measured: about **{capacity['bytes_per_level'] / 1e3:.0f} kB per level**. "
            "This is the library shedding one RNS limb each time the chain is consumed - "
            "independent evidence that the derived level count is right."
        )
with right:
    st.plotly_chart(precision_over_operations(series), use_container_width=True)
    st.caption(
        "Measured CKKS approximation error on a canary probe. Reading it requires the secret "
        "key, so this is an owner-zone measurement; the level and size indicators need no key."
    )

with st.expander("Calibration table (how a byte count becomes a level)"):
    sizes = capacity.get("calibration_sizes") or []
    if sizes:
        st.dataframe(
            {
                "Level": list(range(len(sizes))),
                "Serialized bytes (measured)": sizes,
                "Drop from previous": ["-"] + [f"{sizes[i - 1] - sizes[i]:,}" for i in range(1, len(sizes))],
            },
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Built once per context by encrypting a probe and multiplying it down the chain. "
            "Nearest-bucket matching turns any measured size back into a level."
        )
    else:
        st.info("No calibration table was recorded for this run.")

with st.expander("Raw capacity readings"):
    st.dataframe(series, use_container_width=True, hide_index=True)
