"""Measure how encrypted training scales, and plot it from the measurements.

    python scripts/run_scalability.py --axis samples --sizes 20 40 60 80
    python scripts/run_scalability.py --axis features
    python scripts/run_scalability.py --axis batch --sizes 8 16 32

Every point is a full encrypted training run. Nothing is extrapolated, and a point
that failed is kept in the table and marked rather than dropped from the chart -
a scaling curve that silently omits the sizes that did not work is worse than no
curve at all.

Writes `results/scalability/<timestamp>/` (CSV + JSON) and a figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import require_dependencies  # noqa: E402

require_dependencies(script="scripts/run_scalability.py")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loader import load_dataset, prepare  # noqa: E402
from src.experiments.config import ExperimentConfig, Mode  # noqa: E402
from src.experiments.runner import run_single  # noqa: E402

OUT = ROOT / "results" / "scalability"

AXES = {
    "samples": "Training samples",
    "features": "Model features",
    "batch": "Batch size (ciphertext slots)",
}


def build_split(config: ExperimentConfig, axis: str, size: int):
    dataset = load_dataset(config.dataset)
    split = prepare(
        dataset, test_fraction=config.test_fraction, seed=config.split_seed,
        n_samples=config.n_samples, feature_range=config.feature_range,
    )
    if axis == "features":
        split.x_train = split.x_train[:, :size]
        split.x_test = split.x_test[:, :size]
        split.feature_names = split.feature_names[:size]
    return split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo.yaml"))
    parser.add_argument("--axis", choices=sorted(AXES), default="samples")
    parser.add_argument("--sizes", type=int, nargs="*")
    parser.add_argument("--mode", default=Mode.FHE_ADAPTIVE, choices=list(Mode.ENCRYPTED))
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="held fixed while samples vary, so the step count scales with the workload",
    )
    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)
    dataset = load_dataset(base.dataset)

    if args.sizes:
        sizes = sorted(args.sizes)
    elif args.axis == "samples":
        sizes = [s for s in (40, 60, 80, 100) if s <= dataset.n_samples]
    elif args.axis == "features":
        sizes = list(range(1, min(dataset.n_features, 4) + 1))
    else:
        sizes = [8, 16, 32]

    print(f"Scalability sweep: {AXES[args.axis]} = {sizes}, mode={args.mode}")
    print("Each point is a full encrypted training run.\n")

    points: list[dict[str, Any]] = []
    for size in sizes:
        data = base.to_dict()
        data.pop("derived", None)
        if args.axis == "samples":
            data["n_samples"] = size
            # The batch size is held FIXED while samples vary, so the number of
            # training steps grows with the workload. Scaling it with the sample
            # count instead - as an earlier version did - kept the step count
            # pinned at one batch per epoch across most of the range, and the
            # resulting "scaling curve" measured the batching arithmetic rather
            # than the workload.
            data["batch_size"] = args.batch_size
        elif args.axis == "batch":
            data["batch_size"] = size
        config = ExperimentConfig.from_dict(data)
        split = build_split(config, args.axis, size)

        started = time.perf_counter()
        result = run_single(config, args.mode, split)
        point = {
            "size": size,
            "status": result.status,
            "reason": result.reason,
            "train_seconds": result.metrics.get("train_seconds"),
            "wall_seconds": time.perf_counter() - started,
            "steps": result.metrics.get("steps"),
            "refreshes": result.metrics.get("refreshes"),
            "encrypted_ops": result.metrics.get("encrypted_ops_total"),
            "peak_rss_above_baseline_mb": result.resources.get("peak_rss_above_baseline_mb"),
            "test_accuracy": result.metrics.get("final_test_accuracy"),
        }
        points.append(point)
        seconds = point["train_seconds"]
        shown = "not measured" if seconds is None else f"{seconds:.1f}s"
        print(
            f"  {size:>4}: {result.status:<10} {shown}  "
            f"{point['steps']} steps  {point['refreshes']} refreshes  "
            f"{point['encrypted_ops']} ops"
        )

    out = OUT / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "points.json").write_text(
        json.dumps({"axis": args.axis, "mode": args.mode, "points": points}, indent=2),
        encoding="utf-8",
    )
    with (out / "points.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)

    ok = [p for p in points if p["status"] == "SUCCEEDED"]
    failed = [p for p in points if p["status"] != "SUCCEEDED"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    label = AXES[args.axis]
    series = [
        ("train_seconds", "Training seconds", axes[0][0]),
        ("peak_rss_above_baseline_mb", "Peak RSS relative to run start (MB)", axes[0][1]),
        ("refreshes", "Refresh events", axes[1][0]),
        ("encrypted_ops", "Homomorphic operations", axes[1][1]),
    ]
    # A negative memory value is not an error: it means the process held less at
    # its peak than when sampling began, because allocations from the previous
    # point were freed during this one. Reported rather than clamped to zero.
    for key, ylabel, ax in series:
        pts = [(p["size"], p[key]) for p in ok if p.get(key) is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    color="#2563eb", linewidth=1.8)
        else:
            ax.text(0.5, 0.5, f"{ylabel}\nnot measured", ha="center", va="center",
                    transform=ax.transAxes, color="#6b7280")
        for p in failed:
            ax.axvline(p["size"], color="#dc2626", linestyle=":", linewidth=1.2)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)

    title = f"Encrypted training against {label.lower()} ({args.mode})"
    if failed:
        title += f"  -  dotted red marks {len(failed)} size(s) that did not complete"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "scalability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWritten to {out}")
    if failed:
        print(f"{len(failed)} size(s) did not complete and are marked on the chart:")
        for p in failed:
            print(f"  size {p['size']}: {p['status']} - {p['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
