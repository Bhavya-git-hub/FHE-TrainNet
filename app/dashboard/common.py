"""Shared dashboard plumbing: state, the sidebar, and the honesty banner.

The refresh-kind banner appears on every page that can show a refresh. That is
deliberate repetition: a viewer who joins the demonstration at the bootstrapping
page must not have to have seen the overview to learn that this backend performs
no CKKS bootstrapping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crypto.backend import ParameterError, RefreshKind  # noqa: E402
from src.crypto.tenseal_backend import get_owner_zone  # noqa: E402
from src.data.loader import available_datasets  # noqa: E402
from src.experiments.config import ExperimentConfig, Mode  # noqa: E402
from src.model.activation import ACTIVATIONS  # noqa: E402
from src.noise.monitor import METRIC_DISCLAIMER, METRIC_NAME  # noqa: E402
from src.runtime import (  # noqa: E402
    BATCHED_PROFILE_PEAK_MB,
    LOW_MEMORY_THRESHOLD_MB,
    default_config_name,
    default_config_path,
    profile_note,
)

PAGE_ICON = "🔐"


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"FHE-TrainNet - {title}", page_icon=PAGE_ICON, layout="wide")
    st.title(title)


def refresh_banner(kind: RefreshKind | None = None, *, compact: bool = False) -> None:
    """State plainly what the active refresh primitive is.

    This is a *capability statement*, not an error, and the styling says so. An
    earlier version used `st.warning`, which paints the panel amber and made the
    overview page look as though something had gone wrong - readers asked what was
    broken. Nothing is broken: Microsoft SEAL simply does not implement CKKS
    bootstrapping, and this panel is the disclosure the specification requires
    (Master Prompt 12).

    The wording is unchanged. Only the framing is, because a disclosure that reads
    as a malfunction gets dismissed as noise, and this one needs to be read.
    """
    kind = kind or RefreshKind.CLIENT_AIDED

    if kind.is_bootstrapping:
        st.success(
            f"**Refresh primitive in use: {kind.label}**\n\n"
            "Capacity is restored by the library's own bootstrapping operation. The key "
            "holder is not involved."
        )
        return

    if compact:
        st.info(f"**Refresh primitive in use:** {kind.label}")
        return

    st.info(
        f"**Refresh primitive in use: {kind.label}**\n\n"
        "This is a property of the library, not a fault in this system. Microsoft SEAL "
        "implements no CKKS bootstrapping, so capacity is restored by a client-aided "
        "refresh - the key holder decrypts and re-encrypts. Everything else on this "
        "dashboard is real: the encryption, the encrypted training, the capacity metric, "
        "the controller and every decision it makes.\n\n"
        "It is labelled this way on every screen so that no result here can be mistaken "
        "for one produced by bootstrapping. See **OpenFHE Backend** for the optional "
        "container that does perform it."
    )


def metric_note() -> None:
    st.caption(f"**{METRIC_NAME}.** {METRIC_DISCLAIMER}")


def get_config() -> ExperimentConfig:
    """The configuration currently being edited, held in session state.

    The starting profile is chosen from the memory this process can actually see,
    because the batched profile needs ~1.9 GB for rotation keys and a 1 GB hosting
    tier will kill it mid-demonstration. See `src.runtime`.
    """
    if "config" not in st.session_state:
        st.session_state.config = ExperimentConfig.load(default_config_path())
    return st.session_state.config


def profile_banner() -> None:
    """Say so when a reduced profile was selected, and why."""
    note = profile_note()
    if note:
        st.caption(note)


def set_config(config: ExperimentConfig) -> None:
    st.session_state.config = config


def sidebar_controls() -> ExperimentConfig:
    """Every tunable the Master Prompt section 14 asks for, in one place."""
    config = get_config()
    data = config.to_dict()
    data.pop("derived", None)

    with st.sidebar:
        st.header("Experiment configuration")

        datasets = available_datasets()
        if not datasets:
            st.error("No datasets found. Run `python scripts/build_datasets.py` first.")
            st.stop()
        data["dataset"] = st.selectbox(
            "Dataset", datasets, index=datasets.index(config.dataset) if config.dataset in datasets else 0
        )
        data["n_samples"] = st.slider("Samples used", 20, 560, int(config.n_samples or 100), step=10)
        data["test_fraction"] = st.slider("Test fraction", 0.1, 0.5, config.test_fraction, 0.05)

        st.subheader("Model")
        names = list(ACTIVATIONS)
        data["activation"] = st.selectbox(
            "Polynomial activation", names,
            index=names.index(config.activation) if config.activation in names else 0,
            help="Higher degree approximates the sigmoid better but costs another modulus "
                 "level per training step.",
        )
        act = ACTIVATIONS[data["activation"]]
        st.caption(
            f"degree {act.degree}, costs {act.depth_cost} level(s), "
            f"max error vs sigmoid {act.approximation_error()['max_abs_error']:.4f}"
        )
        data["learning_rate"] = st.slider("Learning rate", 0.05, 3.0, config.learning_rate, 0.05)
        batch_options = [1, 8, 16, 32, 64, 128]
        data["batch_size"] = st.select_slider(
            "Batch size (ciphertext slots used)", batch_options,
            value=config.batch_size if config.batch_size in batch_options else 32,
            help="1 processes a single sample per step. That skips the cross-slot "
                 "reduction, which is the only operation needing Galois rotation keys - "
                 "measured at 1901 MB against 105 MB without them. Slower per sample, but "
                 "it is what fits a 1 GB host.",
        )
        if data["batch_size"] == 1:
            st.caption(
                "Batch 1: no rotation keys, ~105 MB context, one level cheaper per step."
            )
        elif default_config_name() == "cloud":
            # The profile was reduced because this host cannot afford rotation
            # keys. Moving this slider asks for them anyway, and the process is
            # killed at key generation rather than failing with a message - so
            # the warning has to arrive before the run, not after it.
            st.warning(
                f"**This host cannot run batch {data['batch_size']}.** The low-memory "
                f"profile was selected because the batched profile needs "
                f"{LOW_MEMORY_THRESHOLD_MB} MB (measured peak "
                f"{BATCHED_PROFILE_PEAK_MB} MB, mostly Galois rotation keys). "
                "Generating those keys here will have the process killed outright - "
                "no error message, just a restart. Set it back to 1."
            )
        data["epochs"] = st.slider("Epochs", 1, 20, config.epochs)

        st.subheader("CKKS parameters")
        degrees = [8192, 16384, 32768]
        data["poly_modulus_degree"] = st.selectbox(
            "poly_modulus_degree", degrees,
            index=degrees.index(config.poly_modulus_degree) if config.poly_modulus_degree in degrees else 1,
        )
        chain_text = st.text_input(
            "coeff_mod_bit_sizes", ", ".join(str(b) for b in config.coeff_mod_bit_sizes),
            help="Comma-separated prime bit sizes. Length minus two is the multiplicative depth.",
        )
        try:
            data["coeff_mod_bit_sizes"] = [int(p.strip()) for p in chain_text.split(",") if p.strip()]
        except ValueError:
            st.error("coeff_mod_bit_sizes must be a comma-separated list of integers.")
            data["coeff_mod_bit_sizes"] = list(config.coeff_mod_bit_sizes)
        data["scale_bits"] = st.slider("scale_bits", 15, 50, config.scale_bits)

        st.subheader("Refresh policy")
        data["baseline_interval"] = st.slider(
            "Fixed baseline: refresh every N steps", 1, 10, config.baseline_interval,
            help="The control condition. Too large and the run fails; too small and it wastes work.",
        )
        data["adaptive_safety_margin"] = st.slider(
            "Adaptive: safety margin (levels)", 0, 5, config.adaptive_safety_margin
        )
        use_precision = st.checkbox(
            "Adaptive also uses measured precision",
            value=config.adaptive_min_precision_bits is not None,
            help="Requires the data-owner key, so this makes the policy owner-assisted rather "
                 "than something the untrusted compute zone could run alone.",
        )
        data["adaptive_min_precision_bits"] = (
            st.slider("Minimum precision (bits)", 4.0, 30.0,
                      float(config.adaptive_min_precision_bits or 12.0), 0.5)
            if use_precision else None
        )

        st.subheader("Run")
        data["modes"] = st.multiselect(
            "Modes", list(Mode.ALL), default=list(config.modes),
        )
        data["trials"] = st.slider("Trials per mode", 1, 5, config.trials)

        try:
            updated = ExperimentConfig.from_dict(data)
        except (ParameterError, ValueError) as exc:
            st.error(f"**Invalid configuration**\n\n{exc}")
            st.stop()
        set_config(updated)

        budget = updated.depth_budget()
        st.divider()
        st.markdown("**Depth budget**")
        st.markdown(
            f"- chain provides **{budget['max_depth']}** level(s)\n"
            f"- one training step costs **{budget['depth_per_step']}**\n"
            f"- **{budget['steps_between_refresh']}** step(s) fit between refreshes"
        )
        if not budget["baseline_is_feasible"]:
            st.error(
                f"The fixed interval of {budget['baseline_interval']} needs "
                f"{budget['baseline_interval'] * budget['depth_per_step']} levels but only "
                f"{budget['max_depth']} exist. That mode is expected to FAIL - a legitimate "
                "result, recorded as such."
            )
        return updated


def owner_for(config: ExperimentConfig, *, spinner: str = "Generating CKKS keys...") -> Any:
    """Get (and cache for this session) the data-owner zone for these parameters.

    `generate_galois` is passed explicitly, and that is the whole point of this
    function. It defaults to True in `get_owner_zone`, so calling it without the
    argument generated Galois rotation keys unconditionally - measured at 1901 MB
    against 105 MB without them.

    The trainer had always got this right (`run_single` passes
    `model_cfg.needs_rotation_keys`), so the low-memory profile looked correct
    everywhere it was tested from the command line. The dashboard did not, and
    the dashboard is what the hosted deployment runs: selecting a batch of one
    specifically to avoid those keys, then building them anyway on the first
    encrypted page, put the process 1.6 GB deep on a tier that allows about one,
    and it was killed with no error a viewer could read.

    The flag is part of the cache key as well. Two zones for one parameter set
    differ by exactly this, and a key that cannot tell them apart hands back a
    context missing the keys the caller needs - or holding the ones it was trying
    not to pay for.
    """
    params = config.ckks_params()
    needs_galois = bool(config.depth_budget()["needs_rotation_keys"])
    key = (
        params.poly_modulus_degree,
        params.coeff_mod_bit_sizes,
        params.scale_bits,
        needs_galois,
    )
    cache = st.session_state.setdefault("_owner_cache", {})
    if key not in cache:
        detail = (
            "with rotation keys, ~1.9 GB" if needs_galois
            else "no rotation keys needed at batch 1, ~105 MB"
        )
        with st.spinner(f"{spinner} (20-35 s at n={params.poly_modulus_degree}; {detail}; "
                        "done once per parameter set, and never written to disk)"):
            cache[key] = get_owner_zone(params, generate_galois=needs_galois)
    return cache[key]


def latest_run() -> dict[str, Any] | None:
    return st.session_state.get("last_run")


def store_run(payload: dict[str, Any]) -> None:
    st.session_state["last_run"] = payload


def require_run() -> dict[str, Any]:
    """Pages that need results explain how to get them rather than erroring."""
    run = latest_run()
    if run is None:
        from src.experiments.registry import list_runs

        # The most recent run that actually has results - not merely the first
        # directory, which may be one that failed to write or was interrupted.
        usable = next((s for s in list_runs() if s.payload), None)
        if usable is not None:
            return usable.payload
        st.info(
            "No experiment has been run yet. Open **Run Experiment**, or run "
            "`python scripts/run_experiment.py --config configs/demo.yaml` from a terminal."
        )
        st.stop()
    return run


def stat(stats: dict[str, Any], key: str, spec: str = ".3f", missing: str = "not measured") -> str:
    """Format one aggregate from a run's comparison block, tolerating old files.

    Results written before a metric existed simply do not contain its key. Reading
    it with `stats[key]["mean"]` raises `KeyError` and takes the whole page down -
    which is exactly how the report generator broke on a run recorded twenty
    minutes before a metric was added. A run from an earlier version is not
    corrupt, and the honest rendering of a metric it never recorded is
    "not recorded", not a crash and not a zero.
    """
    entry = stats.get(key)
    if entry is None:
        return "not recorded"
    if not isinstance(entry, dict):
        return fmt(entry, spec, missing)
    return fmt(entry.get("mean"), spec, missing)


def fmt(value: Any, spec: str = ".3f", missing: str = "not measured") -> str:
    """Format a number, saying plainly when there isn't one.

    `None` becomes "not measured" rather than "0.000". The distinction is the
    product's spine and it survives all the way into the rendering.
    """
    if value is None:
        return missing
    if isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)
