"""Run Experiment / Training Monitor - Master Prompt section 13D and section 14.

The live view exists because a final accuracy number does not demonstrate
anything. What an evaluator should see is the loop running: each step consuming
levels, the controller deciding, and the refresh happening when it decides one is
needed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import (
    fmt,
    stat,
    metric_note,
    page_setup,
    refresh_banner,
    sidebar_controls,
    store_run,
)
from src.crypto.backend import RefreshKind
from src.experiments.runner import run_experiment

page_setup("Run Experiment")
config = sidebar_controls()
refresh_banner(RefreshKind.CLIENT_AIDED, compact=True)

budget = config.depth_budget()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modulus depth", budget["max_depth"])
c2.metric("Depth per step", budget["depth_per_step"])
c3.metric("Steps between refreshes", budget["steps_between_refresh"])
c4.metric("Modes", len(config.modes))

st.markdown(
    f"Running **{', '.join(config.modes)}** on `{config.dataset}` with "
    f"{config.trials} trial(s) each. All modes share one dataset split, one set of initial "
    "weights and one set of CKKS parameters, so the only difference between the encrypted "
    "modes is the refresh policy."
)

if st.button("Run experiment", type="primary"):
    header = st.empty()
    metrics_box = st.empty()
    decision_box = st.empty()
    log_box = st.empty()
    progress_bar = st.progress(0.0)

    lines: list[str] = []
    state = {"mode": "", "total_steps": 0}
    total_expected = max(1, config.epochs * config.trials * len(config.modes))

    def on_event(event: dict) -> None:
        kind = event.get("event")
        if kind == "mode_start":
            state["mode"] = event["mode"]
            header.subheader(f"Running: {event['mode']} (trial {event['trial']})")
            return
        if kind == "mode_end":
            m = event["metrics"]
            lines.append(
                f"[{event['mode']}] {event['status']} - "
                f"{fmt(m.get('train_seconds'), '.1f')}s, "
                f"accuracy {fmt(m.get('final_test_accuracy'))}, "
                f"refreshes {fmt(m.get('refreshes'), 'd')}"
            )
            log_box.code("\n".join(lines[-14:]), language=None)
            return

        if event.get("mode") == "encrypted":
            cols = metrics_box.columns(5)
            cols[0].metric("Epoch", f"{event['epoch']}/{event['epochs']}")
            cols[1].metric("Step", event["step"])
            cols[2].metric(
                "Levels remaining",
                f"{event['levels_after']}/{event['max_depth']}",
                delta=f"{event['levels_after'] - event['levels_before']}",
                help=f"Each step consumes {event['levels_needed']} level(s).",
            )
            cols[3].metric("Refreshes", event["refreshes"])
            cols[4].metric("Elapsed", f"{event['elapsed']:.1f}s")

            if event["decision"] == "refresh":
                decision_box.error(
                    f"**REFRESH** at step {event['step']} - {event['reason']}\n\n"
                    f"Primitive used: {event['refresh_kind']}"
                )
            else:
                decision_box.success(
                    f"**CONTINUE** at step {event['step']} - {event['reason']}"
                )
            state["total_steps"] += 1
            lines.append(
                f"step {event['step']:>3}  levels {event['levels_before']}->"
                f"{event['levels_after']}  {event['decision']:<8} "
                f"precision {fmt(event.get('precision_bits'), '.1f')} bits"
            )
            log_box.code("\n".join(lines[-14:]), language=None)

        elif event.get("mode") == "plaintext":
            cols = metrics_box.columns(4)
            cols[0].metric("Epoch", f"{event['epoch']}/{event['epochs']}")
            cols[1].metric("Loss", f"{event['loss']:.4f}")
            cols[2].metric("Train accuracy", f"{event['train_accuracy']:.3f}")
            cols[3].metric("Test accuracy", f"{event['test_accuracy']:.3f}")
            progress_bar.progress(min(1.0, event["epoch"] / max(1, event["epochs"])))

    started = time.perf_counter()
    try:
        payload = run_experiment(config, progress=on_event)
    except Exception as exc:  # noqa: BLE001 - shown to the evaluator with its reason
        st.error(f"**The experiment could not run.**\n\n`{type(exc).__name__}: {exc}`")
        st.stop()
    progress_bar.progress(1.0)
    store_run(payload)
    header.subheader(f"Finished in {time.perf_counter() - started:.1f}s")

    st.success(f"Results written to `results/{payload['run_id']}`")

    st.subheader("What was measured")
    rows = []
    for mode, stats in payload["comparison"]["by_mode"].items():
        rows.append(
            {
                "Mode": mode,
                "OK": f"{stats['succeeded']}/{stats['trials']}",
                "Train seconds": stat(stats, "train_seconds", ".2f"),
                "Test accuracy": stat(stats, "test_accuracy"),
                "Refreshes": stat(stats, "refreshes", ".1f"),
                "Refresh seconds": stat(stats, "refresh_seconds_total", ".2f"),
                "Encrypted ops": stat(stats, "encrypted_ops_total", ".0f"),
                "Peak RSS (MB)": stat(stats, "peak_rss_mb", ".0f"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Verdict")
    for line in payload["comparison"]["verdict"]:
        st.markdown(f"- {line}")
    for mode, stats in payload["comparison"]["by_mode"].items():
        for reason in stats["reasons"]:
            st.error(f"**{mode}**: {reason}")
    st.caption(
        "These statements are generated from the measurements of this run. If the adaptive "
        "policy did not improve on the fixed one, the verdict says so."
    )

metric_note()
