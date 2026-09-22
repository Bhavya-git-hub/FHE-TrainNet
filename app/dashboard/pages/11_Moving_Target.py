"""The Moving Target - why the refresh interval cannot simply be tuned once.

The obvious objection to this whole project is one sentence long: *if exactly one
fixed interval is optimal, tune it once and keep it.* Every other page answers
half of that - the comparison page shows adaptive using fewer refreshes than one
particular baseline - but none of them answers the objection itself, because the
configuration never moves.

This page moves it. It tunes the fixed interval on one model, then makes a change
any practitioner would make for reasons having nothing to do with encryption - a
better polynomial approximation of the sigmoid - and runs the *same tuned value*
again. The tuned constant fails on capacity exhaustion. The adaptive controller,
given no interval at all, survives the change untouched.

The claim being tested is the one in `src/bootstrapping/policies.py`: that the
adaptive rule "keeps finding it when the configuration changes underneath it".
Until this page existed that claim was argued in a docstring and demonstrated
nowhere, which is the kind of gap this project is supposed to refuse.

The configuration is pinned here rather than taken from the sidebar. A
demonstration whose numbers depend on whatever the last visitor left in a slider
is not reproducible, and this one is meant to be run in front of an audience.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import metric_note, page_setup, refresh_banner
from src.crypto.backend import RefreshKind
from src.data.loader import load_dataset, prepare
from src.experiments.config import ExperimentConfig, Mode
from src.experiments.runner import run_single
from src.model.activation import ACTIVATIONS
from src.runtime import default_config_path

page_setup("The Moving Target")
st.caption(
    "Tune the refresh interval until it is optimal, then change the model - and "
    "watch the tuned value stop working."
)

# The two configurations differ in one field. Degree 1 is a poor approximation of
# the sigmoid and degree 3 is the standard one, so moving between them is an
# ordinary modelling decision - which is the point. Nothing here is chosen to
# break the baseline; the baseline breaks because the depth budget moved.
BEFORE = "sigmoid_deg1"
AFTER = "sigmoid_deg3"

refresh_banner(RefreshKind.CLIENT_AIDED, compact=True)


def base_config(samples: int) -> ExperimentConfig:
    """The pinned configuration, sized so a full demonstration stays watchable.

    Batch size is 1 because this page is meant to run on the deployed instance,
    where the cross-slot reduction's Galois keys do not fit. See
    `docs/DEPLOYMENT.md`.
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
    )
    return ExperimentConfig.from_dict(data)


def variant(samples: int, activation: str, interval: int) -> ExperimentConfig:
    data = base_config(samples).to_dict()
    data.pop("derived", None)
    data.update(activation=activation, baseline_interval=interval)
    return ExperimentConfig.from_dict(data)


def budget_for(samples: int, activation: str) -> dict[str, int]:
    return variant(samples, activation, 1).depth_budget()


# --- the arithmetic, stated before anything is measured -----------------------

st.subheader("1. The arithmetic, before any run")

st.markdown(
    "Everything below is predicted from the modulus chain alone. The runs then "
    "either confirm it or they do not - and if they do not, the depth accounting "
    "is wrong and that is the finding."
)

preview_samples = 24
depth = base_config(preview_samples).ckks_params().max_depth
rows = {"Model": [], "Cost per training step": [], "Fixed intervals that fit": []}
feasible: dict[str, list[int]] = {}
for name in (BEFORE, AFTER):
    per_step = budget_for(preview_samples, name)["depth_per_step"]
    fits = [n for n in range(1, 6) if n * per_step <= depth]
    feasible[name] = fits
    act = ACTIVATIONS[name]
    rows["Model"].append(f"{name} (degree {act.degree})")
    rows["Cost per training step"].append(f"{per_step} levels")
    rows["Fixed intervals that fit"].append(", ".join(str(n) for n in fits))

st.table(rows)

TUNED = max(feasible[BEFORE])
st.markdown(
    f"The chain provides **{depth} levels**. On `{BEFORE}` the largest fixed "
    f"interval that fits is **{TUNED}** - so {TUNED} is what tuning finds, and "
    f"{TUNED} refreshes least often while still working.\n\n"
    f"On `{AFTER}` a step costs one level more, and "
    f"**{TUNED} no longer fits**. Nothing about the refresh policy changed. The "
    "budget moved underneath it."
)

st.divider()

# --- the demonstration --------------------------------------------------------

st.subheader("2. Run it")

samples = st.select_slider(
    "Samples (each one is a training step at batch size 1)",
    options=[16, 24, 32, 40],
    value=16,
    help="Fewer samples finish sooner. The outcome does not depend on this - the "
         "depth budget does not care how many steps are taken, only what each costs.",
)

st.caption(
    "Six encrypted runs; the two that fail stop as soon as the ciphertext is "
    "exhausted, so they cost a few seconds each. Key generation happens once and "
    "is reused by every run. Measured at 24 samples on a development machine: "
    "89 s for all six. A hosted instance is slower, so the default is 16."
)

if st.button("Run the demonstration", type="primary"):
    plan = [
        ("act1", BEFORE, 1, Mode.FHE_BASELINE, "wasteful"),
        ("act1", BEFORE, TUNED, Mode.FHE_BASELINE, "tuned"),
        ("act1", BEFORE, TUNED + 1, Mode.FHE_BASELINE, "past the cliff"),
        ("act1", BEFORE, TUNED, Mode.FHE_ADAPTIVE, "no interval given"),
        ("act2", AFTER, TUNED, Mode.FHE_BASELINE, "the same tuned value"),
        ("act2", AFTER, TUNED, Mode.FHE_ADAPTIVE, "no interval given"),
    ]

    progress = st.progress(0.0)
    status = st.empty()
    collected: list[dict[str, object]] = []

    dataset = load_dataset(base_config(samples).dataset)
    for index, (act, activation, interval, mode, note) in enumerate(plan):
        config = variant(samples, activation, interval)
        label = (
            f"{activation}, {'adaptive' if mode == Mode.FHE_ADAPTIVE else f'fixed every {interval}'}"
        )
        status.info(f"Running {index + 1} of {len(plan)}: {label} ({note})")
        split = prepare(
            dataset,
            test_fraction=config.test_fraction,
            seed=config.split_seed,
            n_samples=config.n_samples,
            feature_range=config.feature_range,
        )
        started = time.perf_counter()
        result = run_single(config, mode, split)
        elapsed = time.perf_counter() - started
        per_step = config.depth_budget()["depth_per_step"]
        collected.append(
            {
                "act": act,
                "activation": activation,
                "policy": "adaptive" if mode == Mode.FHE_ADAPTIVE else f"fixed every {interval}",
                "interval": None if mode == Mode.FHE_ADAPTIVE else interval,
                "note": note,
                "per_step": per_step,
                "needed": None if mode == Mode.FHE_ADAPTIVE else interval * per_step,
                "status": result.status,
                "reason": result.reason,
                "refreshes": result.metrics.get("refreshes"),
                "accuracy": result.metrics.get("final_test_accuracy"),
                "seconds": elapsed,
            }
        )
        progress.progress((index + 1) / len(plan))

    status.empty()
    st.session_state["moving_target"] = {
        "samples": samples,
        "depth": depth,
        "tuned": TUNED,
        "results": collected,
    }

# --- what was measured --------------------------------------------------------

record = st.session_state.get("moving_target")
if record is None:
    st.info("Press **Run the demonstration** above. Nothing on this page is pre-recorded.")
    st.stop()


def render(rows_: list[dict[str, object]]) -> None:
    table = {
        "Policy": [], "Levels needed per refresh cycle": [], "Outcome": [],
        "Refreshes": [], "Test accuracy": [], "Seconds": [],
    }
    for row in rows_:
        needed = row["needed"]
        table["Policy"].append(f"{row['policy']} ({row['note']})")
        table["Levels needed per refresh cycle"].append(
            "decided per step" if needed is None
            else f"{row['interval']} x {row['per_step']} = {needed}"
        )
        table["Outcome"].append(
            "SUCCEEDED" if row["status"] == "SUCCEEDED" else f"{row['status']}"
        )
        table["Refreshes"].append(
            row["refreshes"] if row["status"] == "SUCCEEDED" else "-"
        )
        table["Test accuracy"].append(
            f"{row['accuracy']:.3f}" if row["status"] == "SUCCEEDED"
            and isinstance(row["accuracy"], (int, float)) else "-"
        )
        table["Seconds"].append(f"{row['seconds']:.1f}")
    st.table(table)


results = record["results"]
act1 = [r for r in results if r["act"] == "act1"]
act2 = [r for r in results if r["act"] == "act2"]
tuned = record["tuned"]

st.divider()
st.subheader(f"Act 1 - tuning the interval on `{BEFORE}`")
st.caption(f"Budget: {record['depth']} levels. A step costs {act1[0]['per_step']}.")
render(act1)

failed_cliff = next((r for r in act1 if r["status"] != "SUCCEEDED"), None)
if failed_cliff is not None and failed_cliff["reason"]:
    st.error(
        f"**Fixed every {failed_cliff['interval']} failed, as the arithmetic said it "
        f"would.** The failure is real and is recorded as a failure:\n\n"
        f"`{failed_cliff['reason']}`"
    )

st.markdown(
    f"So the interval is tuned to **{tuned}**: the largest value that still works, "
    "and therefore the one that refreshes least. A practitioner would now write "
    "that number into a configuration file and move on."
)

st.divider()
st.subheader(f"Act 2 - the same tuned value on `{AFTER}`")
st.caption(
    "One field changed: the polynomial activation, from degree 1 to degree 3 - a "
    "better approximation of the sigmoid, chosen for modelling reasons. The "
    "refresh policy was not touched."
)
render(act2)

baseline_after = next((r for r in act2 if r["interval"] is not None), None)
adaptive_after = next((r for r in act2 if r["interval"] is None), None)
adaptive_before = next((r for r in act1 if r["interval"] is None), None)

if baseline_after is not None and baseline_after["status"] != "SUCCEEDED":
    st.error(
        f"**The tuned interval of {tuned} now fails.** "
        f"{tuned} x {baseline_after['per_step']} = {baseline_after['needed']} levels "
        f"against a budget of {record['depth']}.\n\n`{baseline_after['reason']}`"
    )

st.divider()
st.subheader("Verdict")

verdict: list[str] = []
if baseline_after is not None and adaptive_after is not None:
    if baseline_after["status"] != "SUCCEEDED" and adaptive_after["status"] == "SUCCEEDED":
        verdict.append(
            f"The tuned fixed interval of {tuned} succeeded before the change and "
            f"{baseline_after['status']} after it. The adaptive controller succeeded "
            "both times, and was never given an interval."
        )
    else:
        # The honest branch: if the tuned value survived, say so.
        verdict.append(
            f"The tuned interval of {tuned} returned {baseline_after['status']} after "
            f"the change and adaptive returned {adaptive_after['status']}. This run "
            "did not reproduce the failure - the depth budget for these two "
            "activations must be re-examined rather than explained away."
        )
if adaptive_before is not None and adaptive_after is not None:
    if (
        adaptive_before["status"] == "SUCCEEDED"
        and adaptive_after["status"] == "SUCCEEDED"
    ):
        verdict.append(
            f"Adaptive used {adaptive_before['refreshes']} refreshes before the change "
            f"and {adaptive_after['refreshes']} after it. It refreshed more often "
            "because each step now costs more - which is the adjustment a fixed "
            "interval cannot make."
        )

for line in verdict:
    st.markdown(f"- {line}")

st.info(
    "**What this does not claim.** Adaptive is not faster than every fixed "
    "interval - a correctly tuned one does the same work, and Act 1 shows it. "
    "The claim is narrower and harder to dismiss: the correct value is a function "
    "of a configuration that changes, and a constant does not track it."
)

metric_note()
