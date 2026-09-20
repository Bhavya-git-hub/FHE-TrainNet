"""Scalability - Master Prompt section 13H and Technical Design demo 9.

Runs the same encrypted workload at several sizes and plots what changes. Each
point is a real run; nothing is extrapolated, and a point that failed is drawn as
a failure rather than omitted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import fmt, page_setup, sidebar_controls
from app.visualization.figures import scalability
from src.data.loader import load_dataset, prepare
from src.experiments.config import ExperimentConfig, Mode
from src.experiments.runner import run_single

page_setup("Scalability")
config = sidebar_controls()

st.markdown(
    "Run the encrypted pipeline at several workload sizes and measure how cost grows. "
    "Every point below is an actual run - this page does not extrapolate."
)

axis = st.radio(
    "Vary",
    ["Dataset size (samples)", "Model size (features)", "Batch size (slots used)"],
    horizontal=True,
)

if axis.startswith("Dataset"):
    dataset = load_dataset(config.dataset)
    options = [n for n in (20, 40, 60, 80, 100, 150, 200) if n <= dataset.n_samples]
    default = options[: min(4, len(options))]
elif axis.startswith("Model"):
    dataset = load_dataset(config.dataset)
    options = list(range(1, dataset.n_features + 1))
    default = options[: min(4, len(options))]
else:
    options = [8, 16, 32, 64]
    default = options[:3]

chosen = st.multiselect("Sizes to run", options, default=default)
mode = st.selectbox("Mode", [Mode.FHE_ADAPTIVE, Mode.FHE_BASELINE], index=0)

st.warning(
    "Each point is a full encrypted training run. With the default parameters a point takes "
    "roughly 10-60 s, so three or four points is a sensible demonstration."
)

if st.button("Run scalability sweep", type="primary") and chosen:
    progress = st.progress(0.0)
    status = st.empty()
    points: list[dict] = []

    for i, size in enumerate(sorted(chosen)):
        data = config.to_dict()
        data.pop("derived", None)
        if axis.startswith("Dataset"):
            data["n_samples"] = int(size)
            data["batch_size"] = min(config.batch_size, max(8, int(size * 0.5)))
        elif axis.startswith("Batch"):
            data["batch_size"] = int(size)
        run_config = ExperimentConfig.from_dict(data)

        dataset = load_dataset(run_config.dataset)
        split = prepare(
            dataset, test_fraction=run_config.test_fraction, seed=run_config.split_seed,
            n_samples=run_config.n_samples, feature_range=run_config.feature_range,
        )
        if axis.startswith("Model"):
            # Truncate features to the requested model width.
            split.x_train = split.x_train[:, : int(size)]
            split.x_test = split.x_test[:, : int(size)]
            split.feature_names = split.feature_names[: int(size)]

        status.info(f"Running size {size} ({i + 1}/{len(chosen)})...")
        started = time.perf_counter()
        result = run_single(run_config, mode, split)
        points.append(
            {
                "size": int(size),
                "status": result.status,
                "train_seconds": result.metrics.get("train_seconds"),
                "wall_seconds": time.perf_counter() - started,
                "refreshes": result.metrics.get("refreshes"),
                "encrypted_ops": result.metrics.get("encrypted_ops_total"),
                "peak_rss_mb": result.resources.get("peak_rss_mb"),
                "test_accuracy": result.metrics.get("final_test_accuracy"),
                "steps": result.metrics.get("steps"),
            }
        )
        progress.progress((i + 1) / len(chosen))

    status.success(f"Completed {len(points)} point(s).")
    st.session_state["scalability"] = {"axis": axis, "mode": mode, "points": points}

state = st.session_state.get("scalability")
if not state:
    st.info("Choose sizes and press **Run scalability sweep**.")
    st.stop()

points = state["points"]
label = {"Dataset size (samples)": "Samples", "Model size (features)": "Features",
         "Batch size (slots used)": "Batch size"}[state["axis"]]

st.subheader(f"{state['mode']} - varying {label.lower()}")
st.dataframe(
    [
        {
            label: p["size"], "Status": p["status"],
            "Train seconds": fmt(p["train_seconds"], ".2f"),
            "Steps": fmt(p["steps"], "d"),
            "Refreshes": fmt(p["refreshes"], "d"),
            "Encrypted ops": fmt(p["encrypted_ops"], "d"),
            "Peak RSS (MB)": fmt(p["peak_rss_mb"], ".0f"),
            "Test accuracy": fmt(p["test_accuracy"]),
        }
        for p in points
    ],
    use_container_width=True, hide_index=True,
)

failed = [p for p in points if p["status"] != "SUCCEEDED"]
if failed:
    st.error(
        f"{len(failed)} point(s) did not succeed: "
        + ", ".join(f"{p['size']} ({p['status']})" for p in failed)
        + ". They are shown above rather than dropped from the chart data."
    )

c1, c2 = st.columns(2)
c1.plotly_chart(
    scalability(points, "size", "train_seconds", label, "Training seconds",
                f"{label} against training time"),
    use_container_width=True,
)
c2.plotly_chart(
    scalability(points, "size", "peak_rss_mb", label, "Peak RSS (MB)",
                f"{label} against peak memory"),
    use_container_width=True,
)
c3, c4 = st.columns(2)
c3.plotly_chart(
    scalability(points, "size", "refreshes", label, "Refresh events",
                f"{label} against refresh frequency"),
    use_container_width=True,
)
c4.plotly_chart(
    scalability(points, "size", "encrypted_ops", label, "Homomorphic operations",
                f"{label} against encrypted operation count"),
    use_container_width=True,
)
