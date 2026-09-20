"""Experiment History and Report/Export - Master Prompt sections 13I and 13J.

Every run ever recorded, and a way to take any of them away as CSV or JSON. The
export deliberately includes the configuration and the environment snapshot, not
just the numbers: a metric without the configuration that produced it is not
reproducible, which defeats the purpose of recording it.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import fmt, stat, page_setup, sidebar_controls
from src.experiments.registry import list_runs, load_run, load_series

page_setup("History & Export")
sidebar_controls()

runs = list_runs()
if not runs:
    st.info(
        "No runs recorded yet. Use **Run Experiment**, or "
        "`python scripts/run_experiment.py --config configs/demo.yaml`."
    )
    st.stop()

st.subheader(f"{len(runs)} recorded run(s)")
st.dataframe([r.row() for r in runs], use_container_width=True, hide_index=True)

broken = [r for r in runs if r.error]
if broken:
    st.warning(
        f"{len(broken)} run directory could not be read and is listed with an error rather "
        "than hidden: " + ", ".join(f"`{r.run_id}` ({r.error})" for r in broken)
    )

st.divider()
ids = [r.run_id for r in runs if not r.error]
if not ids:
    st.stop()
chosen = st.selectbox("Inspect and export a run", ids, index=0)
payload = load_run(chosen)

tab_summary, tab_config, tab_env, tab_raw = st.tabs(
    ["Summary", "Configuration", "Environment", "Raw JSON"]
)

with tab_summary:
    rows = []
    for mode, s in payload["comparison"]["by_mode"].items():
        rows.append(
            {
                "Mode": mode, "OK": f"{s['succeeded']}/{s['trials']}",
                "Train seconds": stat(s, "train_seconds", ".2f"),
                "Test accuracy": stat(s, "test_accuracy"),
                "Refreshes": stat(s, "refreshes", ".1f"),
                "Peak RSS (MB)": stat(s, "peak_rss_mb", ".0f"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    for line in payload["comparison"]["verdict"]:
        st.markdown(f"- {line}")

with tab_config:
    st.json(payload["config"])
    st.caption(f"Configuration fingerprint: `{payload['config_fingerprint']}`")

with tab_env:
    st.json(payload["reproducibility"])
    st.caption(
        "Recorded with every run so a result can be traced to the exact library versions that "
        "produced it. No key material is included."
    )

with tab_raw:
    st.json(payload, expanded=False)

st.divider()
st.subheader("Export")

c1, c2, c3 = st.columns(3)
c1.download_button(
    "Full results (JSON)",
    json.dumps(payload, indent=2, default=str),
    file_name=f"{chosen}_results.json",
    mime="application/json",
    use_container_width=True,
)

summary_csv = io.StringIO()
writer = csv.writer(summary_csv)
writer.writerow(
    ["mode", "trials", "succeeded", "train_seconds_mean", "test_accuracy_mean",
     "final_loss_mean", "refreshes_mean", "refresh_seconds_mean", "encrypted_ops_mean",
     "peak_rss_mb_mean", "mean_cpu_percent"]
)
def _mean(stats: dict, key: str):
    """Raw mean for the CSV, or blank when this run never recorded that metric.

    A blank cell reads as "not recorded" in any spreadsheet; a zero would read as
    a measurement that was taken.
    """
    entry = stats.get(key)
    if not isinstance(entry, dict) or entry.get("mean") is None:
        return ""
    return entry["mean"]


for mode, s in payload["comparison"]["by_mode"].items():
    writer.writerow(
        [mode, s["trials"], s["succeeded"], _mean(s, "train_seconds"),
         _mean(s, "test_accuracy"), _mean(s, "final_loss"), _mean(s, "refreshes"),
         _mean(s, "refresh_seconds_total"), _mean(s, "encrypted_ops_total"),
         _mean(s, "peak_rss_mb"), _mean(s, "mean_cpu_percent")]
)
c2.download_button(
    "Comparison table (CSV)", summary_csv.getvalue(),
    file_name=f"{chosen}_summary.csv", mime="text/csv", use_container_width=True,
)

config_yaml = (ROOT / "results" / chosen / "config.yaml")
if config_yaml.exists():
    c3.download_button(
        "Configuration (YAML)", config_yaml.read_text(encoding="utf-8"),
        file_name=f"{chosen}_config.yaml", mime="text/yaml", use_container_width=True,
    )

st.markdown("**Per-run series**")
encrypted = [r for r in payload["runs"] if r["mode"] != "plaintext"]
for run in encrypted:
    cols = st.columns(3)
    for i, kind in enumerate(("capacity", "decisions", "epochs")):
        series = load_series(payload["run_id"], kind, run["mode"], run["trial"])
        if not series:
            continue
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(series[0].keys()))
        w.writeheader()
        w.writerows(series)
        cols[i].download_button(
            f"{kind} - {run['mode']}", buf.getvalue(),
            file_name=f"{chosen}_{kind}_{run['mode']}_t{run['trial']}.csv",
            mime="text/csv", use_container_width=True,
            key=f"{kind}_{run['mode']}_{run['trial']}",
        )

st.caption(
    f"All files for this run are on disk at `results/{chosen}/` and can be copied directly."
)
