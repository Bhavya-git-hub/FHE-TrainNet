"""OpenFHE backend - the one that really bootstraps, when it is available.

This page exists to make the distinction between the two backends impossible to
miss. It shows what the default backend cannot do, what this one can, and - when
this one is unavailable - says so instead of quietly leaving the impression that
bootstrapping happened somewhere.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.dashboard.common import page_setup, sidebar_controls
from src.crypto.backend import RefreshKind
from src.crypto.openfhe_backend import IMAGE_TAG, build_image, docker_status, probe, run_comparison

page_setup("OpenFHE Backend")
sidebar_controls()

st.markdown(
    """
The default backend (TenSEAL / Microsoft SEAL) **cannot bootstrap** - SEAL does
not implement CKKS bootstrapping. This optional backend runs OpenFHE, which does,
inside a Linux container.

The controller is the same in both. `src/bootstrapping/policies.py` is *mounted*
into the container rather than copied, so the code deciding when to bootstrap here
is the same code that decides when to refresh there. **Only the primitive differs**,
which is what makes the comparison worth making.
"""
)

st.subheader("Two backends, side by side")
st.table(
    {
        "": ["Library", "Refresh primitive", "Runs on Windows natively", "Typical cost",
             "Security consequence"],
        "TenSEAL (default)": [
            "Microsoft SEAL",
            "Client-aided refresh (decrypt + re-encrypt)",
            "Yes",
            "~0.2 s",
            "The key holder sees the intermediate weights",
        ],
        "OpenFHE (optional)": [
            "OpenFHE 1.5.1",
            "EvalBootstrap - actual CKKS bootstrapping",
            "No - Linux/macOS wheel only, so it runs in Docker",
            "Seconds, and needs a larger ring",
            "None - the key never leaves the owner",
        ],
    }
)

st.divider()
st.subheader("Status on this machine")

if st.button("Check status", type="primary"):
    with st.spinner("Asking Docker..."):
        st.session_state["openfhe_status"] = probe()

status = st.session_state.get("openfhe_status")
if status is None:
    status = docker_status()

c1, c2, c3 = st.columns(3)
c1.metric("Docker", "available" if status.available else "unavailable")
c2.metric("Image", "built" if status.image_present else "not built")
c3.metric("EvalBootstrap", "yes" if status.supports_native_bootstrap else "no")

if status.supports_native_bootstrap:
    st.success(f"**{RefreshKind.NATIVE_BOOTSTRAP.label}** - OpenFHE {status.openfhe_version}")
else:
    st.warning(
        f"**This backend is unavailable.** {status.reason}\n\n"
        "It is reported rather than worked around: the TenSEAL client-aided refresh is "
        "**not** substituted and **not** described as bootstrapping. Phase A - the TenSEAL "
        "path - is a complete deliverable without this backend."
    )
    st.caption(
        f"Refresh kind reported: `{status.refresh_kind.value}` - "
        f"is_bootstrapping = {status.refresh_kind.is_bootstrapping}"
    )

with st.expander("Build the image (slow the first time)"):
    st.code(
        "docker info                              # the daemon must be running\n"
        "python scripts/openfhe_backend.py --build\n"
        "python scripts/openfhe_backend.py --probe\n"
        "python scripts/openfhe_backend.py --run",
        language="bash",
    )
    if st.button(f"Build {IMAGE_TAG} now"):
        box = st.empty()
        with st.spinner("Building - this downloads Ubuntu and OpenFHE..."):
            result = build_image(progress=lambda m: box.info(m))
        st.session_state["openfhe_status"] = result
        (st.success if result.available else st.error)(result.reason or "Built.")
        st.rerun()

if status.supports_native_bootstrap:
    st.divider()
    st.subheader("Run the comparison with real bootstrapping")
    steps = st.slider("Steps", 4, 24, 12)
    depth = st.slider("Depth per step", 1, 8, 5)
    levels = st.slider("Usable levels", 4, 20, 10)
    if st.button("Run in container", type="primary"):
        with st.spinner("Running - key generation for a bootstrapping context is slow..."):
            started = time.perf_counter()
            payload = run_comparison(
                {"steps": steps, "depth_per_step": depth, "usable_levels": levels,
                 "ring_dim": 1 << 16, "num_slots": 8}
            )
        if not payload.get("ok"):
            st.error(f"Failed after {time.perf_counter() - started:.0f}s: {payload.get('reason')}")
        else:
            ctx = payload["context"]
            st.caption(
                f"Ring {ctx['ring_dim']}, {ctx['usable_levels']} usable levels "
                f"(+{ctx['bootstrap_depth']} reserved for bootstrapping), "
                f"setup {ctx['setup_seconds']:.0f}s"
            )
            st.dataframe(
                [
                    {
                        "Policy": name, "Status": r["status"],
                        "Bootstraps": r["bootstraps"], "Continues": r["continues"],
                        "Total seconds": round(r["seconds"], 1),
                        "Mean bootstrap (s)": (
                            "-" if r["bootstrap_seconds_mean"] is None
                            else round(r["bootstrap_seconds_mean"], 2)
                        ),
                    }
                    for name, r in payload["results"].items()
                ],
                use_container_width=True, hide_index=True,
            )
            st.success(
                "Every refresh above is an **actual CKKS bootstrapping call** "
                "(OpenFHE EvalBootstrap), not a client-aided refresh."
            )
