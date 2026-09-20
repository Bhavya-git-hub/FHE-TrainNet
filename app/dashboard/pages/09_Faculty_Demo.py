"""Faculty Demo - Master Prompt section 23.

A single guided sequence, in order, sized to be explained in five to ten minutes.
Each stage runs for real; nothing on this page is pre-recorded or staged, and the
stage that cannot be performed - actual CKKS bootstrapping - says so plainly
instead of being skipped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from app.dashboard.common import fmt, owner_for, page_setup, sidebar_controls
from app.visualization.figures import (
    accuracy_comparison,
    capacity_over_operations,
    decision_timeline,
    refresh_comparison,
)
from src.crypto.backend import RefreshKind
from src.crypto.capabilities import precision_bits
from src.data.loader import load_dataset, prepare
from src.experiments.config import Mode
from src.experiments.runner import run_single

page_setup("Faculty Demo")
config = sidebar_controls()

st.markdown(
    """
The complete story in one pass: a private record, encrypted; real computation on
ciphertexts; a model trained without the training engine ever seeing the data;
the capacity limit that forces refreshing; the controller deciding; and finally
the authorized decryption of the result.

Press the button and narrate as it goes.
"""
)

budget = config.depth_budget()
st.info(
    f"**Setup:** `{config.dataset}`, {config.n_samples} samples, activation "
    f"`{config.activation}`, CKKS n={config.poly_modulus_degree}, depth "
    f"{budget['max_depth']}, {budget['depth_per_step']} levels per step - so "
    f"{budget['steps_between_refresh']} step(s) fit between refreshes."
)

if st.button("Run the full demonstration", type="primary"):
    st.session_state["faculty"] = {}
    store = st.session_state["faculty"]
    overall = st.progress(0.0)

    # 1-2. The data, in the clear.
    st.header("1. The data the owner wants to keep private")
    dataset = load_dataset(config.dataset)
    split = prepare(
        dataset, test_fraction=config.test_fraction, seed=config.split_seed,
        n_samples=config.n_samples, feature_range=config.feature_range,
    )
    st.dataframe(
        {
            **{name: split.x_train[:5, i] for i, name in enumerate(split.feature_names)},
            "label": split.y_train[:5].astype(int),
        },
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"{len(split.y_train)} training and {len(split.y_test)} test records. "
        "These values are about to become invisible to the training engine."
    )
    overall.progress(0.1)

    # 3-4. Encryption and ciphertext metadata.
    st.header("2. Encrypted under CKKS")
    owner = owner_for(config)
    backend = owner.backend()
    record = split.x_train[0]
    t0 = time.perf_counter()
    enc = owner.encrypt(record.tolist(), lineage="demo")
    enc_ms = (time.perf_counter() - t0) * 1000
    size = backend.serialized_size(enc)
    c1, c2, c3 = st.columns(3)
    c1.metric("Encryption time", f"{enc_ms:.0f} ms")
    c2.metric("Ciphertext size", f"{size / 1e6:.2f} MB")
    c3.metric("Expansion", f"{size / max(1, record.nbytes):.0f}x")
    st.code(f"record  : {np.round(record, 5).tolist()}\n"
            f"as bytes: {enc.raw.serialize()[:40].hex()}...", language=None)
    back = owner.decrypt(enc)
    st.success(
        f"Authorized decryption returns {np.round(back[: len(record)], 5).tolist()} - "
        f"agreeing to {fmt(precision_bits(back[0], record[0]), '.1f')} bits. CKKS is "
        "approximate, and this is the size of that approximation."
    )
    overall.progress(0.2)

    # 5-6. Computation on ciphertexts.
    st.header("3. Real computation, without decrypting the operands")
    x = owner.encrypt([1.5] * 4)
    y = owner.encrypt([2.0] * 4)
    t0 = time.perf_counter()
    result = backend.add(backend.mul(x, y), x)
    ms = (time.perf_counter() - t0) * 1000
    got = owner.decrypt(result)[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("x * y + x", f"{got:.6f}")
    c2.metric("Exact", "4.500000")
    c3.metric("Time", f"{ms:.1f} ms")
    st.caption(
        "The multiplication consumed one modulus level. The compute zone performed it holding "
        "a public-only context: it could not have decrypted x or y."
    )
    overall.progress(0.3)

    # 7-11. Encrypted training under both policies.
    st.header("4. Training on encrypted data")
    st.caption(
        "The same workload twice: once with a fixed refresh schedule, once with the adaptive "
        "controller. Watch the levels fall and the controller decide."
    )
    results = {}
    for i, (mode, title) in enumerate(
        [(Mode.FHE_BASELINE, "Fixed policy"), (Mode.FHE_ADAPTIVE, "Adaptive policy")]
    ):
        st.subheader(f"4.{i + 1} {title}")
        box = st.empty()
        line = st.empty()

        def on_event(event: dict, _box=box, _line=line) -> None:
            if event.get("mode") != "encrypted":
                return
            cols = _box.columns(4)
            cols[0].metric("Step", event["step"])
            cols[1].metric("Levels", f"{event['levels_before']}->{event['levels_after']}")
            cols[2].metric("Refreshes", event["refreshes"])
            cols[3].metric("Elapsed", f"{event['elapsed']:.1f}s")
            if event["decision"] == "refresh":
                _line.error(f"REFRESH - {event['reason']}")
            else:
                _line.success(f"CONTINUE - {event['reason']}")

        run = run_single(config, mode, split, progress=on_event)
        results[mode] = run
        st.write(
            f"**{run.status}** - {fmt(run.metrics.get('train_seconds'), '.1f')}s, "
            f"{run.metrics.get('refreshes')} refresh(es), test accuracy "
            f"{fmt(run.metrics.get('final_test_accuracy'))}"
        )
        overall.progress(0.3 + 0.25 * (i + 1))

    # 9. Capacity and decisions.
    st.header("5. The capacity limit, and the decisions it forces")
    adaptive = results[Mode.FHE_ADAPTIVE]
    st.plotly_chart(
        capacity_over_operations(
            adaptive.capacity_series, adaptive.decisions,
            threshold=adaptive.metrics.get("depth_per_step"),
        ),
        use_container_width=True,
    )
    st.plotly_chart(decision_timeline(adaptive.decisions), use_container_width=True)

    kind = RefreshKind(adaptive.metrics["refresh_kind"])
    if kind.is_bootstrapping:
        st.success(f"Refresh primitive: {kind.label}")
    else:
        st.warning(
            f"**Refresh primitive: {kind.label}**\n\n"
            "This is the honest limit of this backend. Microsoft SEAL implements no CKKS "
            "bootstrapping, so capacity is restored by the key holder decrypting and "
            "re-encrypting. The controller, the metric and the decisions above are all real; "
            "the primitive is not bootstrapping and is never described as such."
        )
    overall.progress(0.85)

    # 12-14. Comparison.
    st.header("6. Fixed against adaptive")
    base, adap = results[Mode.FHE_BASELINE], results[Mode.FHE_ADAPTIVE]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Refreshes", f"{adap.metrics['refreshes']} vs {base.metrics['refreshes']}",
        delta=adap.metrics["refreshes"] - base.metrics["refreshes"], delta_color="inverse",
    )
    c2.metric(
        "Training seconds",
        f"{fmt(adap.metrics['train_seconds'], '.1f')} vs {fmt(base.metrics['train_seconds'], '.1f')}",
        delta=f"{adap.metrics['train_seconds'] - base.metrics['train_seconds']:.1f}s",
        delta_color="inverse",
    )
    c3.metric(
        "Test accuracy",
        f"{fmt(adap.metrics['final_test_accuracy'])} vs {fmt(base.metrics['final_test_accuracy'])}",
    )

    # 15. Authorized decryption of the final model.
    st.header("7. The authorized decryption")
    st.caption(
        "The model existed only as ciphertext throughout training. This is the first time its "
        "weights have been readable, and only the key holder can do this."
    )
    model = adap.metrics.get("model")
    if model:
        st.dataframe(
            {
                "Parameter": [*split.feature_names, "bias"],
                "Decrypted value": [f"{w:.6f}" for w in model["weights"]] + [f"{model['bias']:.6f}"],
            },
            use_container_width=True, hide_index=True,
        )
        st.success(
            f"Final test accuracy on the encrypted-trained model: "
            f"**{fmt(adap.metrics.get('final_test_accuracy'))}**"
        )
    else:
        st.error("No model was produced - the run did not complete. See the status above.")
    overall.progress(1.0)

    st.session_state["faculty"] = {"results": {k: v.to_dict() for k, v in results.items()}}
    st.balloons()

else:
    st.info("Press **Run the full demonstration** to begin. It takes roughly 1-3 minutes.")
