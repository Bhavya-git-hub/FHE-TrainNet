"""Generate the report figures and tables from a recorded run.

    python scripts/make_report.py                 # newest run
    python scripts/make_report.py --run-id <id>

Produces the eight figures Master Prompt section 17 requires, plus a markdown
summary, under `reports/`. Every figure is drawn from files under `results/`;
this script computes nothing about the experiment and invents nothing. A figure
whose data was never measured is written as a labelled placeholder saying so,
rather than omitted or filled in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import require_dependencies  # noqa: E402

require_dependencies(script="scripts/make_report.py")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.experiments.registry import list_runs, load_run, load_series  # noqa: E402
from src.noise.monitor import METRIC_DISCLAIMER  # noqa: E402

REPORTS = ROOT / "reports"

COLOURS = {
    "plaintext": "#6b7280",
    "fhe_baseline": "#d97706",
    "fhe_adaptive": "#2563eb",
    "fhe_no_refresh": "#dc2626",
}
LABELS = {
    "plaintext": "Plaintext",
    "fhe_baseline": "FHE fixed",
    "fhe_adaptive": "FHE adaptive",
    "fhe_no_refresh": "FHE no refresh",
}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)


def _no_data(path: Path, title: str, why: str) -> None:
    """Write a figure that states the data is missing instead of faking it."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.text(0.5, 0.5, f"{title}\n\n{why}", ha="center", va="center", fontsize=10,
            color="#6b7280", wrap=True)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bar(path: Path, by_mode: dict, key: str, title: str, ylabel: str, fmt: str = "{:.2f}") -> bool:
    modes, values, errs = [], [], []
    for mode, stats in by_mode.items():
        stat = stats.get(key, {})
        if stat.get("mean") is None:
            continue
        modes.append(mode)
        values.append(stat["mean"])
        errs.append((stat["max"] - stat["min"]) / 2 if stat.get("n", 0) > 1 else 0.0)
    if not modes:
        _no_data(path, title, f"No mode recorded a value for '{key}'.")
        return False
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(
        [LABELS.get(m, m) for m in modes], values,
        yerr=errs if any(e > 0 for e in errs) else None, capsize=4,
        color=[COLOURS.get(m, "#333") for m in modes],
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value),
                ha="center", va="bottom", fontsize=9)
    _style(ax, title, "", ylabel)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_capacity(payload: dict, path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    drew = False
    for run in payload["runs"]:
        if run["mode"] == "plaintext" or run["trial"] != 0:
            continue
        series = load_series(payload["run_id"], "capacity", run["mode"], run["trial"])
        if not series:
            continue
        drew = True
        ax.plot(
            [r["index"] for r in series], [r["levels_remaining"] for r in series],
            marker="o", markersize=3.5, linewidth=1.6,
            color=COLOURS.get(run["mode"], "#333"), label=LABELS.get(run["mode"], run["mode"]),
        )
        decisions = load_series(payload["run_id"], "decisions", run["mode"], run["trial"])
        for d in decisions:
            if d.get("decision") == "refresh":
                ax.axvline(d.get("iteration", 0), color=COLOURS.get(run["mode"], "#333"),
                           alpha=0.22, linestyle="--", linewidth=1)
    if not drew:
        _no_data(path, "Capacity during training", "No encrypted run recorded capacity readings.")
        return False
    needed = next(
        (r["metrics"].get("depth_per_step") for r in payload["runs"] if r["mode"] != "plaintext"),
        None,
    )
    if needed:
        ax.axhline(needed, color="#dc2626", linestyle=":", linewidth=1.4,
                   label=f"depth needed per step ({needed})")
    _style(ax, "Ciphertext level remaining during training (derived)\nDashed verticals mark refresh events",
           "Operation index", "Levels remaining")
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, -0.06, METRIC_DISCLAIMER, fontsize=6, color="#6b7280", wrap=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_precision(payload: dict, path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    drew = False
    for run in payload["runs"]:
        if run["mode"] == "plaintext" or run["trial"] != 0:
            continue
        series = load_series(payload["run_id"], "capacity", run["mode"], run["trial"])
        pts = [(r["index"], r.get("precision_bits")) for r in series if r.get("precision_bits")]
        if not pts:
            continue
        drew = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=3,
                linewidth=1.6, color=COLOURS.get(run["mode"], "#333"),
                label=LABELS.get(run["mode"], run["mode"]))
    if not drew:
        _no_data(path, "Measured precision", "No canary precision was recorded (use_canary off).")
        return False
    _style(ax, "Measured precision of the canary probe (real CKKS approximation error)",
           "Operation index", "Bits of agreement")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_timeline(payload: dict, path: Path) -> bool:
    encrypted = [r for r in payload["runs"] if r["mode"] != "plaintext" and r["trial"] == 0]
    if not encrypted:
        _no_data(path, "Refresh timeline", "This run contains no encrypted mode.")
        return False
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.75 * len(encrypted)))
    for row, run in enumerate(encrypted):
        decisions = load_series(payload["run_id"], "decisions", run["mode"], run["trial"])
        cont = [d["iteration"] for d in decisions if d.get("decision") == "continue"]
        refs = [d["iteration"] for d in decisions if d.get("decision") == "refresh"]
        ax.scatter(cont, [row] * len(cont), marker="o", s=26, color="#16a34a",
                   label="continue" if row == 0 else None)
        ax.scatter(refs, [row] * len(refs), marker="^", s=80, color="#dc2626",
                   label="refresh" if row == 0 else None)
    ax.set_yticks(range(len(encrypted)))
    ax.set_yticklabels([LABELS.get(r["mode"], r["mode"]) for r in encrypted], fontsize=9)
    _style(ax, "Controller decisions over training", "Training step", "")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_loss(payload: dict, path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    drew = False
    for run in payload["runs"]:
        if run["trial"] != 0:
            continue
        series = load_series(payload["run_id"], "epochs", run["mode"], run["trial"])
        pts = [(e["epoch"] + 1, e.get("train_loss")) for e in series if e.get("train_loss") is not None]
        if not pts:
            continue
        drew = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=4,
                color=COLOURS.get(run["mode"], "#333"), label=LABELS.get(run["mode"], run["mode"]))
    if not drew:
        _no_data(path, "Training loss", "No epoch losses were recorded.")
        return False
    _style(ax, "Training loss per epoch", "Epoch", "Mean squared error")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_markdown(payload: dict, figures: dict[str, bool], path: Path) -> None:
    config = payload["config"]
    by_mode = payload["comparison"]["by_mode"]

    def cell(stats: dict[str, Any], key: str, fmt: str = "{:.3f}") -> str:
        """Render one statistic, tolerating results written by an earlier version.

        A metric added after a run was recorded is absent from that run's file,
        not zero in it. Reporting "not recorded" keeps old results readable
        instead of crashing the report or implying a measurement that never
        happened.
        """
        stat = stats.get(key)
        if stat is None:
            return "not recorded"
        if stat.get("mean") is None:
            return "not measured"
        text = fmt.format(stat["mean"])
        if stat.get("n", 0) > 1:
            text += f" ({fmt.format(stat['min'])}-{fmt.format(stat['max'])})"
        return text

    lines = [
        f"# Experiment report: `{payload['run_id']}`",
        "",
        "Generated by `scripts/make_report.py` from the files in "
        f"`results/{payload['run_id']}/`. Every number below traces to one of them.",
        "",
        "## Configuration",
        "",
        f"- dataset **{config['dataset']}** "
        f"({payload['split']['train_samples']} train / {payload['split']['test_samples']} test)",
        f"- activation **{config['activation']}**, learning rate {config['learning_rate']}, "
        f"batch size {config['batch_size']}, {config['epochs']} epochs",
        f"- CKKS n={config['poly_modulus_degree']}, chain `{config['coeff_mod_bit_sizes']}`, "
        f"scale 2^{config['scale_bits']}",
        f"- multiplicative depth **{config['derived']['max_depth']}**, "
        f"depth per training step **{config['derived']['depth_per_step']}**, "
        f"so **{config['derived']['steps_between_refresh']}** step(s) fit between refreshes",
        f"- trials per mode: {config['trials']}",
        "",
        "## Results",
        "",
        "| Mode | OK | Train (s) | Test accuracy | Loss | Refreshes | Refresh (s) | Enc. ops | Peak RSS above baseline (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, s in by_mode.items():
        lines.append(
            f"| {mode} | {s['succeeded']}/{s['trials']} | {cell(s, 'train_seconds', '{:.2f}')} "
            f"| {cell(s, 'test_accuracy')} | {cell(s, 'final_loss', '{:.4f}')} "
            f"| {cell(s, 'refreshes', '{:.1f}')} | {cell(s, 'refresh_seconds_total', '{:.2f}')} "
            f"| {cell(s, 'encrypted_ops_total', '{:.0f}')} "
            f"| {cell(s, 'peak_rss_above_baseline_mb', '{:.0f}')} |"
        )

    lines += ["", "## What the measurements show", ""]
    for line in payload["comparison"]["verdict"]:
        lines.append(f"- {line}")

    failures = [(m, s) for m, s in by_mode.items() if s["succeeded"] < s["trials"]]
    if failures:
        lines += ["", "## Runs that did not complete", ""]
        for mode, s in failures:
            for reason in s["reasons"]:
                lines.append(f"- **{mode}**: {reason}")

    lines += ["", "## Figures", ""]
    for name, drawn in figures.items():
        note = "" if drawn else "  *(no data - the figure states why)*"
        lines.append(f"- `{name}.png`{note}")

    lines += [
        "",
        "## Provenance",
        "",
        f"- configuration fingerprint `{payload['config_fingerprint']}`",
        f"- generated {payload['reproducibility']['timestamp_utc']} on "
        f"`{payload['reproducibility']['platform']}`",
        f"- TenSEAL {payload['reproducibility']['versions'].get('tenseal')}, "
        f"Python {payload['reproducibility']['versions'].get('python')}",
        "",
        f"> {METRIC_DISCLAIMER}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="defaults to the most recent run")
    parser.add_argument("--out-dir", default=str(REPORTS))
    args = parser.parse_args()

    if args.run_id:
        payload = load_run(args.run_id)
    else:
        runs = [r for r in list_runs() if r.payload]
        if not runs:
            print("No runs found under results/. Run scripts/run_experiment.py first.")
            return 1
        payload = runs[0].payload

    out = Path(args.out_dir) / payload["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    by_mode = payload["comparison"]["by_mode"]

    figures = {
        "01_training_time": _bar(out / "01_training_time.png", by_mode, "train_seconds",
                                 "Training time by mode", "Seconds", "{:.1f}"),
        "02_accuracy": _bar(out / "02_accuracy.png", by_mode, "test_accuracy",
                            "Test accuracy by mode", "Accuracy", "{:.3f}"),
        "03_capacity": figure_capacity(payload, out / "03_capacity.png"),
        "04_refresh_timeline": figure_timeline(payload, out / "04_refresh_timeline.png"),
        "05_refresh_count": _bar(out / "05_refresh_count.png", by_mode, "refreshes",
                                 "Refresh events by mode", "Refreshes", "{:.1f}"),
        "06_memory": _bar(out / "06_memory.png", by_mode, "peak_rss_above_baseline_mb",
                          "Peak memory above baseline by mode", "MB", "{:.0f}"),
        "07_cpu": _bar(out / "07_cpu.png", by_mode, "mean_cpu_percent",
                       "Mean CPU by mode", "Percent", "{:.0f}"),
        "08_precision": figure_precision(payload, out / "08_precision.png"),
        "09_loss": figure_loss(payload, out / "09_loss.png"),
    }

    write_markdown(payload, figures, out / "REPORT.md")

    drawn = sum(1 for v in figures.values() if v)
    print(f"Report for {payload['run_id']}")
    print(f"  {drawn}/{len(figures)} figures drawn from recorded data")
    for name, ok in figures.items():
        print(f"    {'ok     ' if ok else 'no data'} {name}.png")
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
