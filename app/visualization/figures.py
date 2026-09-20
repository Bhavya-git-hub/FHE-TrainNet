"""Plotly figures, built only from recorded run data.

Every function here takes measurements that a run actually produced. None of them
invent a point, interpolate across a gap, or substitute a default for a missing
value: a `None` in the data becomes a gap in the line, because a gap is the
honest rendering of "not measured".

The capacity charts carry their provenance in the axis title - "derived" for the
level count, "measured" for ciphertext size and precision - so a reader of the
chart cannot mistake one for the other.
"""

from __future__ import annotations

from typing import Any, Sequence

import plotly.graph_objects as go

from src.noise.monitor import METRIC_DISCLAIMER, METRIC_NAME

# One colour per mode, used consistently everywhere.
MODE_COLOURS = {
    "plaintext": "#6b7280",
    "fhe_baseline": "#d97706",
    "fhe_adaptive": "#2563eb",
    "fhe_no_refresh": "#dc2626",
}
MODE_LABELS = {
    "plaintext": "Plaintext reference",
    "fhe_baseline": "FHE + fixed policy",
    "fhe_adaptive": "FHE + adaptive policy",
    "fhe_no_refresh": "FHE + no refresh (control)",
}
STATE_COLOURS = {
    "safe": "#16a34a",
    "warning": "#d97706",
    "critical": "#dc2626",
    "exhausted": "#7f1d1d",
}


def _empty(message: str) -> go.Figure:
    """A figure that says why it is empty rather than showing a blank grid."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#6b7280"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20), height=300,
    )
    return fig


def _layout(fig: go.Figure, title: str, xaxis: str, yaxis: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def capacity_over_operations(
    series: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]] = (),
    *,
    threshold: int | None = None,
) -> go.Figure:
    """Levels remaining against operation index, annotated with refresh events."""
    if not series:
        return _empty("No capacity readings recorded for this run.")

    x = [r["index"] for r in series]
    levels = [r["levels_remaining"] for r in series]
    states = [r.get("state", "safe") for r in series]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=levels, mode="lines+markers", name="Levels remaining (derived)",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=7, color=[STATE_COLOURS.get(s, "#2563eb") for s in states]),
            hovertemplate="op %{x}<br>%{y} level(s) remaining<extra></extra>",
        )
    )
    if threshold is not None:
        fig.add_hline(
            y=threshold, line=dict(color="#dc2626", dash="dash", width=1.5),
            annotation_text=f"next step needs {threshold}", annotation_position="top left",
        )

    for record in decisions:
        if record.get("decision") == "refresh":
            fig.add_vline(
                x=record.get("iteration", 0), line=dict(color="#16a34a", dash="dot", width=1)
            )
    if any(d.get("decision") == "refresh" for d in decisions):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="#16a34a", dash="dot"), name="Refresh event",
            )
        )

    _layout(fig, METRIC_NAME, "Operation index", "Levels remaining (derived from modulus chain)")
    fig.add_annotation(
        text=METRIC_DISCLAIMER, xref="paper", yref="paper", x=0, y=-0.28, showarrow=False,
        align="left", font=dict(size=9, color="#6b7280"), xanchor="left",
    )
    fig.update_layout(margin=dict(b=110))
    return fig


def precision_over_operations(series: Sequence[dict[str, Any]]) -> go.Figure:
    """Measured canary precision. Gaps where no probe was taken."""
    points = [(r["index"], r.get("precision_bits")) for r in series]
    if not any(p[1] is not None for p in points):
        return _empty(
            "No canary precision was measured in this run.\n"
            "Enable use_canary in the configuration to record it."
        )
    fig = go.Figure(
        go.Scatter(
            x=[p[0] for p in points],
            # None values are passed through so Plotly draws a gap rather than a
            # line through a value that was never measured.
            y=[p[1] for p in points],
            mode="lines+markers", name="Measured precision",
            line=dict(color="#7c3aed", width=2), connectgaps=False,
            hovertemplate="op %{x}<br>%{y:.1f} bits<extra></extra>",
        )
    )
    return _layout(
        fig,
        "Measured precision of the canary probe (real CKKS approximation error)",
        "Operation index",
        "Bits of agreement with exact arithmetic (measured)",
    )


def ciphertext_size_over_operations(series: Sequence[dict[str, Any]]) -> go.Figure:
    """Measured serialized ciphertext size - the library's own capacity account."""
    points = [(r["index"], r.get("serialized_bytes")) for r in series]
    if not any(p[1] is not None for p in points):
        return _empty("Ciphertext size was not measured in this run (measure_every was too high).")
    fig = go.Figure(
        go.Scatter(
            x=[p[0] for p in points],
            y=[None if p[1] is None else p[1] / 1e6 for p in points],
            mode="lines+markers", name="Serialized size",
            line=dict(color="#0891b2", width=2), connectgaps=False,
            hovertemplate="op %{x}<br>%{y:.3f} MB<extra></extra>",
        )
    )
    return _layout(
        fig,
        "Serialized ciphertext size (measured) - falls one RNS limb per level consumed",
        "Operation index",
        "Megabytes (measured)",
    )


def loss_curves(runs: dict[str, list[dict[str, Any]]]) -> go.Figure:
    """Training loss per epoch for each mode."""
    fig = go.Figure()
    drawn = False
    for mode, epochs in runs.items():
        pts = [(e["epoch"] + 1, e.get("train_loss")) for e in epochs]
        if not any(p[1] is not None for p in pts):
            continue
        drawn = True
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                name=MODE_LABELS.get(mode, mode),
                line=dict(color=MODE_COLOURS.get(mode, "#333"), width=2), connectgaps=False,
            )
        )
    if not drawn:
        return _empty("No loss values were recorded.")
    return _layout(fig, "Training loss per epoch", "Epoch", "Mean squared error")


def accuracy_comparison(by_mode: dict[str, dict[str, Any]]) -> go.Figure:
    """Test accuracy per mode, with the measured range across trials."""
    modes, values, errors, texts = [], [], [], []
    for mode, stats in by_mode.items():
        stat = stats.get("test_accuracy", {})
        if stat.get("mean") is None:
            continue
        modes.append(MODE_LABELS.get(mode, mode))
        values.append(stat["mean"])
        errors.append((stat["max"] - stat["min"]) / 2 if stat.get("n", 0) > 1 else 0)
        texts.append(f"{stat['mean']:.3f}")
    if not modes:
        return _empty("No mode completed successfully, so there is no accuracy to compare.")
    fig = go.Figure(
        go.Bar(
            x=modes, y=values, text=texts, textposition="outside",
            error_y=dict(type="data", array=errors, visible=any(e > 0 for e in errors)),
            marker_color=[MODE_COLOURS.get(m, "#333") for m in by_mode if
                          by_mode[m].get("test_accuracy", {}).get("mean") is not None],
        )
    )
    fig.update_yaxes(range=[0, 1.05])
    return _layout(fig, "Test accuracy by mode", "", "Accuracy")


def timing_comparison(by_mode: dict[str, dict[str, Any]]) -> go.Figure:
    """Training time per mode, with refresh time shown as its own component.

    Splitting the bar matters: it shows how much of any difference between the
    policies came from refreshing rather than from anything else.
    """
    modes, train, refresh = [], [], []
    for mode, stats in by_mode.items():
        t = stats.get("train_seconds", {}).get("mean")
        if t is None:
            continue
        r = stats.get("refresh_seconds_total", {}).get("mean") or 0.0
        modes.append(MODE_LABELS.get(mode, mode))
        train.append(max(0.0, t - r))
        refresh.append(r)
    if not modes:
        return _empty("No timing was recorded.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=modes, y=train, name="Computation", marker_color="#2563eb"))
    fig.add_trace(go.Bar(x=modes, y=refresh, name="Refreshing", marker_color="#f59e0b"))
    fig.update_layout(barmode="stack")
    return _layout(fig, "Training time by mode", "", "Seconds")


def refresh_comparison(by_mode: dict[str, dict[str, Any]]) -> go.Figure:
    """Number of refresh events per mode."""
    modes, values = [], []
    for mode, stats in by_mode.items():
        v = stats.get("refreshes", {}).get("mean")
        if v is None:
            continue
        modes.append(MODE_LABELS.get(mode, mode))
        values.append(v)
    if not modes:
        return _empty("No refresh counts were recorded.")
    fig = go.Figure(
        go.Bar(
            x=modes, y=values, text=[f"{v:.1f}" for v in values], textposition="outside",
            marker_color="#7c3aed",
        )
    )
    return _layout(fig, "Refresh events by mode", "", "Refreshes")


def decision_timeline(decisions: Sequence[dict[str, Any]]) -> go.Figure:
    """Every controller decision in order, refreshes distinguished from continues."""
    if not decisions:
        return _empty("No decisions were recorded for this run.")
    cont = [d for d in decisions if d.get("decision") == "continue"]
    refs = [d for d in decisions if d.get("decision") == "refresh"]
    fig = go.Figure()
    if cont:
        fig.add_trace(
            go.Scatter(
                x=[d["iteration"] for d in cont], y=[d["levels_remaining"] for d in cont],
                mode="markers", name="CONTINUE",
                marker=dict(color="#16a34a", size=10, symbol="circle"),
                customdata=[[d.get("reason", "")] for d in cont],
                hovertemplate="step %{x}<br>%{y} level(s)<br>%{customdata[0]}<extra></extra>",
            )
        )
    if refs:
        fig.add_trace(
            go.Scatter(
                x=[d["iteration"] for d in refs], y=[d["levels_remaining"] for d in refs],
                mode="markers", name="REFRESH",
                marker=dict(color="#dc2626", size=14, symbol="triangle-up"),
                customdata=[[d.get("reason", "")] for d in refs],
                hovertemplate="step %{x}<br>%{y} level(s)<br>%{customdata[0]}<extra></extra>",
            )
        )
    needed = decisions[0].get("levels_needed")
    if needed is not None:
        fig.add_hline(
            y=needed, line=dict(color="#dc2626", dash="dash", width=1.5),
            annotation_text=f"depth needed per step = {needed}",
        )
    return _layout(fig, "Controller decisions", "Training step", "Levels remaining when deciding")


def scalability(points: Sequence[dict[str, Any]], x_key: str, y_key: str,
                x_label: str, y_label: str, title: str) -> go.Figure:
    """Generic workload-scaling chart used by the scalability page."""
    usable = [(p.get(x_key), p.get(y_key)) for p in points]
    usable = [(a, b) for a, b in usable if a is not None and b is not None]
    if not usable:
        return _empty(f"No scalability data recorded for {y_label}.")
    usable.sort()
    fig = go.Figure(
        go.Scatter(
            x=[u[0] for u in usable], y=[u[1] for u in usable], mode="lines+markers",
            line=dict(color="#2563eb", width=2), marker=dict(size=9),
        )
    )
    return _layout(fig, title, x_label, y_label)


def activation_fit(activation: Any, limit: float = 3.0) -> go.Figure:
    """The polynomial against the sigmoid it replaces, with the gap shaded."""
    import numpy as np

    from src.model.activation import sigmoid

    grid = np.linspace(-limit, limit, 300)
    poly = activation(grid)
    exact = np.asarray(sigmoid(grid), dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=grid, y=exact, mode="lines", name="True sigmoid",
                   line=dict(color="#6b7280", width=2, dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=grid, y=poly, mode="lines", name=f"{activation.name} (degree {activation.degree})",
                   line=dict(color="#2563eb", width=2.5))
    )
    fig.add_trace(
        go.Scatter(x=grid, y=np.abs(poly - exact), mode="lines", name="Absolute error",
                   line=dict(color="#dc2626", width=1.5), yaxis="y2")
    )
    fig.update_layout(
        yaxis2=dict(title="Absolute error", overlaying="y", side="right", showgrid=False)
    )
    return _layout(
        fig,
        f"Polynomial activation against the sigmoid it approximates (levels: {activation.depth_cost})",
        "Pre-activation value z", "Activation output",
    )
