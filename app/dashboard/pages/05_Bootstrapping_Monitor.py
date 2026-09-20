"""Bootstrapping Monitor - Master Prompt section 13F.

Every decision, refreshes and continues alike. A page that showed only the
refreshes would hide the evidence that matters most: the moments the adaptive
controller looked at the ciphertext and decided it did *not* need to refresh.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import fmt, page_setup, refresh_banner, require_run, sidebar_controls
from app.visualization.figures import decision_timeline
from src.crypto.backend import RefreshKind
from src.experiments.registry import load_series

page_setup("Bootstrapping Monitor")
sidebar_controls()
payload = require_run()

encrypted = [r for r in payload["runs"] if r["mode"] != "plaintext"]
if not encrypted:
    st.info("This run contains no encrypted mode, so there are no refresh decisions to show.")
    st.stop()

kinds = {r["metrics"].get("refresh_kind") for r in encrypted}
kind = RefreshKind(next(iter(kinds))) if len(kinds) == 1 and None not in kinds else RefreshKind.CLIENT_AIDED
refresh_banner(kind)

labels = [f"{r['mode']} (trial {r['trial']})" for r in encrypted]
choice = st.selectbox("Run", labels, index=len(labels) - 1)
selected = encrypted[labels.index(choice)]
decisions = load_series(payload["run_id"], "decisions", selected["mode"], selected["trial"])
metrics = selected["metrics"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Policy", metrics.get("policy", {}).get("policy", "-"))
c2.metric("Refreshes", fmt(metrics.get("refreshes"), "d"))
c3.metric("Continues", fmt(metrics.get("continues"), "d"))
c4.metric(
    "Mean refresh time",
    fmt(metrics.get("refresh_seconds_mean"), ".3f", "no refreshes") + (
        " s" if metrics.get("refresh_seconds_mean") is not None else ""
    ),
)

st.info(f"**Policy in force:** {metrics.get('policy', {}).get('describe', '-')}")
st.caption(f"Decision rule: `{decisions[0]['threshold'] if decisions else '-'}`")

if not metrics.get("is_actual_bootstrapping", False):
    st.info(
        "**Every event below is a client-aided refresh, not CKKS bootstrapping.** "
        "The controller, the metric and the decisions are real; the primitive that restores "
        "capacity is a decrypt-and-re-encrypt performed by the key holder, because Microsoft "
        "SEAL implements no CKKS bootstrapping. Its security consequence is that the owner "
        "sees the intermediate weights - which is the price of this primitive, and the reason "
        "the OpenFHE backend exists."
    )

st.plotly_chart(decision_timeline(decisions), use_container_width=True)

st.subheader("Decision log")
st.caption(
    "Every decision the controller made, in order, with the reason it gave at the time. "
    "Continues are included - they are the evidence that the adaptive policy declined to "
    "refresh when it could have."
)
if decisions:
    view = [
        {
            "Step": d.get("iteration"),
            "Epoch": d.get("epoch"),
            "Decision": str(d.get("decision", "")).upper(),
            "Levels left": d.get("levels_remaining"),
            "Levels needed": d.get("levels_needed"),
            "Precision (bits)": fmt(d.get("precision_bits"), ".1f"),
            "Refresh time (s)": fmt(d.get("refresh_seconds"), ".3f", "-"),
            "Cumulative refreshes": d.get("cumulative_refreshes"),
            "Reason": d.get("reason"),
        }
        for d in decisions
    ]
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)
else:
    st.info("No decisions were recorded for this run.")

with st.expander("Compare the policies on this same run"):
    rows = []
    for r in encrypted:
        m = r["metrics"]
        rows.append(
            {
                "Mode": r["mode"],
                "Status": r["status"],
                "Policy": m.get("policy", {}).get("policy"),
                "Refreshes": fmt(m.get("refreshes"), "d"),
                "Continues": fmt(m.get("continues"), "d"),
                "Refresh seconds": fmt(m.get("refresh_seconds_total"), ".2f"),
                "Train seconds": fmt(m.get("train_seconds"), ".2f"),
                "Test accuracy": fmt(m.get("final_test_accuracy")),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
