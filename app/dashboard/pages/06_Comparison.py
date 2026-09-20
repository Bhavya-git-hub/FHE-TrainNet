"""Experiment Comparison - Master Prompt section 13G.

Plaintext reference against fixed-policy FHE against adaptive FHE, on identical
inputs. The verdict text is generated from the measurements, so when the adaptive
policy does not win, this page says the adaptive policy did not win.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import fmt, stat, metric_note, page_setup, require_run, sidebar_controls
from app.visualization.figures import (
    accuracy_comparison,
    loss_curves,
    refresh_comparison,
    timing_comparison,
)
from src.experiments.registry import load_series

page_setup("Experiment Comparison")
sidebar_controls()
payload = require_run()

by_mode = payload["comparison"]["by_mode"]

st.subheader("Verdict")
verdict = payload["comparison"]["verdict"]
if verdict:
    for line in verdict:
        st.markdown(f"- {line}")
else:
    st.info("Not enough successful modes in this run to draw a comparison.")
st.caption(
    "Generated from this run's measurements. The project's claim is that the adaptive strategy "
    "is measurable, not that it always wins."
)

st.divider()
st.subheader("Side by side")
rows = []
for mode, s in by_mode.items():
    rows.append(
        {
            "Mode": mode,
            "Trials OK": f"{s['succeeded']}/{s['trials']}",
            "Train seconds": stat(s, "train_seconds", ".2f"),
            "Test accuracy": stat(s, "test_accuracy"),
            "Final loss": stat(s, "final_loss", ".4f"),
            "Refreshes": stat(s, "refreshes", ".1f"),
            "Refresh seconds": stat(s, "refresh_seconds_total", ".2f"),
            "Encrypted ops": stat(s, "encrypted_ops_total", ".0f"),
            "Peak RSS (MB)": stat(s, "peak_rss_mb", ".0f"),
            "Mean CPU %": stat(s, "mean_cpu_percent", ".0f"),
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)
st.caption(
    "'not measured' means exactly that. A plaintext run has no refresh count, so the cell is "
    "empty rather than zero - zero would read as 'it needed none', which is a different claim."
)

for mode, s in by_mode.items():
    for reason in s["reasons"]:
        st.error(f"**{mode}** did not complete: {reason}")

st.divider()
c1, c2 = st.columns(2)
c1.plotly_chart(accuracy_comparison(by_mode), use_container_width=True)
c2.plotly_chart(timing_comparison(by_mode), use_container_width=True)

c3, c4 = st.columns(2)
c3.plotly_chart(refresh_comparison(by_mode), use_container_width=True)

epochs_by_mode = {}
for run in payload["runs"]:
    if run["trial"] != 0:
        continue
    series = load_series(payload["run_id"], "epochs", run["mode"], run["trial"])
    if series:
        epochs_by_mode[run["mode"]] = series
c4.plotly_chart(loss_curves(epochs_by_mode), use_container_width=True)

st.divider()
st.subheader("Were these runs actually comparable?")
config = payload["config"]
split = payload["split"]
st.markdown(
    f"""
All modes in this run shared:

- dataset **{config['dataset']}**, split seed **{config['split_seed']}**,
  {split['train_samples']} train / {split['test_samples']} test samples
- model seed **{config['model_seed']}**, so every mode started from identical weights
- activation **{config['activation']}**, learning rate **{config['learning_rate']}**,
  batch size **{config['batch_size']}**, **{config['epochs']}** epochs
- CKKS parameters n={config['poly_modulus_degree']},
  chain `{config['coeff_mod_bit_sizes']}`, scale 2^{config['scale_bits']}

The only difference between `fhe_baseline` and `fhe_adaptive` is the refresh policy.
"""
)
if split["clipped_fraction"] > 0:
    st.caption(
        f"{split['clipped_fraction'] * 100:.2f}% of feature values were clipped to the "
        f"±{config['feature_range']} range the polynomial activation is faithful on."
    )

metric_note()
