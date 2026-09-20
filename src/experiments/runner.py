"""Run experiments and write down everything they measured.

One rule governs this module: a run reports what happened. A run that fails is
recorded as FAILED with the reason; a run that was cut short is PARTIAL; a
quantity that could not be measured is `None`. Nothing is rounded up into a
success, and no missing number is replaced by a plausible one.

The comparison between modes is only meaningful if they are genuinely comparable,
so plaintext, baseline-FHE and adaptive-FHE within one experiment share the same
dataset split, the same seeds, the same initial weights, the same model shape and
the same CKKS parameters. The only difference between the two encrypted modes is
the refresh policy.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.bootstrapping.policies import (
    AdaptivePolicy,
    BaselinePolicy,
    NoRefreshPolicy,
    RefreshPolicy,
)
from src.crypto.backend import CryptoError, RefreshKind
from src.crypto.tenseal_backend import get_owner_zone
from src.data.loader import Split, load_dataset, prepare
from src.evaluation.resources import ResourceSampler
from src.experiments.config import ExperimentConfig, Mode, reproducibility_snapshot
from src.noise.monitor import METRIC_DISCLAIMER, METRIC_NAME, CapacityMonitor
from src.training.encrypted_trainer import EncryptedTrainer
from src.training.plaintext_trainer import train_plaintext

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class RunResult:
    """One mode, one trial."""

    mode: str
    trial: int
    status: str
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    capacity_series: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "trial": self.trial,
            "status": self.status,
            "reason": self.reason,
            "metrics": self.metrics,
            "resources": self.resources,
            "epochs": self.epochs,
        }


def policy_for(mode: str, config: ExperimentConfig) -> RefreshPolicy:
    if mode == Mode.FHE_BASELINE:
        return BaselinePolicy(interval=config.baseline_interval)
    if mode == Mode.FHE_ADAPTIVE:
        return AdaptivePolicy(
            safety_margin=config.adaptive_safety_margin,
            min_precision_bits=config.adaptive_min_precision_bits,
        )
    if mode == Mode.FHE_NO_REFRESH:
        return NoRefreshPolicy()
    raise ValueError(f"No refresh policy applies to mode '{mode}'.")


def run_single(
    config: ExperimentConfig,
    mode: str,
    split: Split,
    *,
    trial: int = 0,
    progress: ProgressFn | None = None,
) -> RunResult:
    """Execute one mode once and return everything it measured."""
    model_cfg = config.model_config(split.x_train.shape[1])

    if mode == Mode.PLAINTEXT:
        with ResourceSampler() as sampler:
            started = time.perf_counter()
            result = train_plaintext(split, model_cfg, progress=progress)
            wall = time.perf_counter() - started
        return RunResult(
            mode=mode,
            trial=trial,
            status="SUCCEEDED",
            metrics={
                "train_seconds": result.train_seconds,
                "wall_seconds": wall,
                "setup_seconds": 0.0,
                "steps": result.steps,
                "final_train_accuracy": result.final_train_accuracy,
                "final_test_accuracy": result.final_test_accuracy,
                "final_loss": result.final_loss,
                "activation_range_exceeded": result.activation_range_exceeded,
                "diverged": result.diverged,
                "max_abs_z": result.max_abs_z,
                "warnings": result.warnings,
                # These have no meaning without encryption, so they are None -
                # not zero, which would read as "no refreshes were needed".
                "refreshes": None,
                "refresh_seconds_total": None,
                "refresh_seconds_mean": None,
                "refresh_kind": None,
                "encrypted_ops_total": None,
                "min_levels_remaining": None,
                "precision_bits_last": None,
                "model": result.model.to_dict(),
            },
            epochs=[vars(e) for e in result.epochs],
            resources=sampler.summary().to_dict(),
        )

    params = config.ckks_params()
    policy = policy_for(mode, config)

    setup_started = time.perf_counter()
    owner = get_owner_zone(params)
    keygen_seconds = time.perf_counter() - setup_started

    monitor = CapacityMonitor(params)
    trainer = EncryptedTrainer(
        owner,
        owner.backend(),
        model_cfg,
        policy,
        monitor,
        measure_every=config.measure_every,
        use_canary=config.use_canary,
    )

    with ResourceSampler() as sampler:
        started = time.perf_counter()
        try:
            enc = trainer.train(
                split, progress=progress, max_seconds=config.max_seconds_per_run
            )
            status, reason = enc.status, enc.reason
        except CryptoError as exc:
            # Reported with its reason rather than swallowed (SRS C3 / NFR-08).
            enc = None
            status, reason = "FAILED", f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - started

    log_summary = trainer.log.summary()
    cap_summary = monitor.summary()

    metrics: dict[str, Any] = {
        "train_seconds": enc.train_seconds if enc else wall,
        "wall_seconds": wall,
        "setup_seconds": (enc.setup_seconds if enc else 0.0) + keygen_seconds,
        "keygen_seconds": keygen_seconds,
        "steps": enc.steps if enc else 0,
        "final_train_accuracy": enc.final_train_accuracy if enc else None,
        "final_test_accuracy": enc.final_test_accuracy if enc else None,
        "final_loss": enc.final_loss if enc else None,
        "activation_range_exceeded": enc.activation_range_exceeded if enc else None,
        "diverged": enc.diverged if enc else None,
        "max_abs_z": enc.max_abs_z if enc else None,
        "warnings": enc.warnings if enc else [],
        "refreshes": log_summary["refreshes"],
        "continues": log_summary["continues"],
        "refresh_seconds_total": log_summary["refresh_seconds_total"],
        "refresh_seconds_mean": log_summary["refresh_seconds_mean"],
        "refresh_seconds_max": log_summary["refresh_seconds_max"],
        "refresh_kind": owner.refresh_kind().value,
        "refresh_kind_label": owner.refresh_kind().label,
        "is_actual_bootstrapping": owner.refresh_kind().is_bootstrapping,
        "encrypted_ops": enc.encrypted_ops if enc else {},
        "encrypted_ops_total": sum(enc.encrypted_ops.values()) if enc else 0,
        "policy": policy.to_dict(),
        "capacity": cap_summary,
        "min_levels_remaining": cap_summary["min_levels_remaining"],
        "precision_bits_first": cap_summary["precision_bits_first"],
        "precision_bits_last": cap_summary["precision_bits_last"],
        "depth_per_step": model_cfg.depth_per_step(),
        "max_depth": params.max_depth,
        "model": enc.model.to_dict() if enc and enc.model else None,
    }

    return RunResult(
        mode=mode,
        trial=trial,
        status=status,
        reason=reason,
        metrics=metrics,
        capacity_series=monitor.series(),
        decisions=trainer.log.rows(),
        epochs=[vars(e) for e in enc.epochs] if enc else [],
        resources=sampler.summary().to_dict(),
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    progress: ProgressFn | None = None,
    results_dir: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run every configured mode and trial, then write the results directory."""
    results_dir = Path(results_dir or RESULTS_DIR)
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{config.name}-{uuid.uuid4().hex[:6]}"
    out = results_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.dataset)
    split = prepare(
        dataset,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
        n_samples=config.n_samples,
        feature_range=config.feature_range,
    )

    runs: list[RunResult] = []
    for trial in range(config.trials):
        for mode in config.modes:
            if progress is not None:
                progress({"event": "mode_start", "mode": mode, "trial": trial})
            result = run_single(config, mode, split, trial=trial, progress=progress)
            runs.append(result)
            if progress is not None:
                progress(
                    {
                        "event": "mode_end",
                        "mode": mode,
                        "trial": trial,
                        "status": result.status,
                        "metrics": result.metrics,
                    }
                )

    payload = {
        "run_id": run_id,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "dataset": dataset.summary(),
        "split": {
            "train_samples": int(len(split.y_train)),
            "test_samples": int(len(split.y_test)),
            "clipped_fraction": split.clipped_fraction,
            "feature_names": split.feature_names,
        },
        "capacity_metric": {"name": METRIC_NAME, "disclaimer": METRIC_DISCLAIMER},
        "reproducibility": reproducibility_snapshot(
            {"refresh_kind": RefreshKind.CLIENT_AIDED.value}
        ),
        "runs": [r.to_dict() for r in runs],
        "comparison": compare(runs),
    }

    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    config.save(out / "config.yaml")
    _write_csvs(out, runs)
    return payload


def compare(runs: list[RunResult]) -> dict[str, Any]:
    """Aggregate across trials, and say plainly what the numbers show.

    The verdict is computed from the measurements. If adaptive refreshed more
    often, or took longer, the verdict says so - the project's claim is that the
    strategy is measurable, not that it always wins (Master Prompt 24).
    """
    by_mode: dict[str, list[RunResult]] = {}
    for run in runs:
        by_mode.setdefault(run.mode, []).append(run)

    summary: dict[str, Any] = {}
    for mode, group in by_mode.items():
        ok = [g for g in group if g.status == "SUCCEEDED"]
        summary[mode] = {
            "trials": len(group),
            "succeeded": len(ok),
            "statuses": [g.status for g in group],
            "reasons": [g.reason for g in group if g.reason],
            "train_seconds": _stat(ok, "train_seconds"),
            "test_accuracy": _stat(ok, "final_test_accuracy"),
            "final_loss": _stat(ok, "final_loss"),
            "refreshes": _stat(ok, "refreshes"),
            "refresh_seconds_total": _stat(ok, "refresh_seconds_total"),
            "encrypted_ops_total": _stat(ok, "encrypted_ops_total"),
            "peak_rss_mb": _stat_resource(ok, "peak_rss_mb"),
            # Order-independent: absolute RSS includes whatever the process had
            # already allocated, so the first encrypted run of a session carries
            # the key material and looks heavier than the policy made it.
            "peak_rss_above_baseline_mb": _stat_resource(ok, "peak_rss_above_baseline_mb"),
            "mean_cpu_percent": _stat_resource(ok, "mean_cpu_percent"),
        }

    verdict: list[str] = []
    base, adapt = summary.get(Mode.FHE_BASELINE), summary.get(Mode.FHE_ADAPTIVE)
    if base and adapt and base["succeeded"] and adapt["succeeded"]:
        b_ref, a_ref = base["refreshes"]["mean"], adapt["refreshes"]["mean"]
        b_t, a_t = base["train_seconds"]["mean"], adapt["train_seconds"]["mean"]
        if b_ref is not None and a_ref is not None:
            if a_ref < b_ref:
                verdict.append(
                    f"Adaptive performed {a_ref:.1f} refreshes on average against the fixed "
                    f"policy's {b_ref:.1f} ({(1 - a_ref / b_ref) * 100:.0f}% fewer)."
                )
            elif a_ref > b_ref:
                verdict.append(
                    f"Adaptive performed MORE refreshes ({a_ref:.1f}) than the fixed policy "
                    f"({b_ref:.1f}) in this configuration."
                )
            else:
                verdict.append(
                    f"Both policies performed {a_ref:.1f} refreshes: the fixed interval happened "
                    "to be optimal for this configuration."
                )
        if b_t and a_t:
            # Spelled out rather than signed. "+38% for adaptive" reads as though
            # adaptive took 38% longer, which is the opposite of what it means.
            change = (b_t - a_t) / b_t * 100
            direction = (
                f"adaptive was {change:.0f}% faster"
                if change > 0
                else f"adaptive was {-change:.0f}% slower"
            )
            verdict.append(
                f"Training time {a_t:.1f}s adaptive against {b_t:.1f}s fixed - {direction}."
            )
        b_acc, a_acc = base["test_accuracy"]["mean"], adapt["test_accuracy"]["mean"]
        if b_acc is not None and a_acc is not None:
            verdict.append(
                f"Test accuracy {a_acc:.3f} adaptive against {b_acc:.3f} fixed."
            )
    plain = summary.get(Mode.PLAINTEXT)
    if plain and adapt and plain["succeeded"] and adapt["succeeded"]:
        p_acc = plain["test_accuracy"]["mean"]
        a_acc = adapt["test_accuracy"]["mean"]
        if p_acc is not None and a_acc is not None:
            verdict.append(
                f"Encrypted training reached {a_acc:.3f} test accuracy against the plaintext "
                f"reference's {p_acc:.3f} (difference {a_acc - p_acc:+.3f})."
            )
    diverged = [run.mode for run in runs if run.metrics.get("diverged")]
    if diverged:
        verdict.append(
            "WARNING: in mode(s) "
            + ", ".join(sorted(set(diverged)))
            + " the pre-activation value left the range the polynomial activation is faithful "
            "on. Past that point the polynomial decreases as its input grows, the gradient "
            "inverts and the accuracy figures above do not represent successful learning. "
            "Lower the learning rate or use a lower-degree activation."
        )

    failed = [m for m, s in summary.items() if s["succeeded"] < s["trials"]]
    if failed:
        verdict.append(f"Mode(s) that did not complete every trial: {', '.join(failed)}.")

    return {"by_mode": summary, "verdict": verdict}


def _stat(runs: list[RunResult], key: str) -> dict[str, Any]:
    values = [r.metrics.get(key) for r in runs]
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _stat_resource(runs: list[RunResult], key: str) -> dict[str, Any]:
    values = [r.resources.get(key) for r in runs]
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _write_csvs(out: Path, runs: list[RunResult]) -> None:
    """Write the flat files the report and dashboard read."""
    import csv

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "mode", "trial", "status", "train_seconds", "test_accuracy", "final_loss",
                "refreshes", "refresh_seconds_total", "refresh_kind", "encrypted_ops_total",
                "min_levels_remaining", "precision_bits_last", "peak_rss_mb", "mean_cpu_percent",
            ]
        )
        for r in runs:
            m = r.metrics
            writer.writerow(
                [
                    r.mode, r.trial, r.status, m.get("train_seconds"),
                    m.get("final_test_accuracy"), m.get("final_loss"), m.get("refreshes"),
                    m.get("refresh_seconds_total"), m.get("refresh_kind"),
                    m.get("encrypted_ops_total"), m.get("min_levels_remaining"),
                    m.get("precision_bits_last"), r.resources.get("peak_rss_mb"),
                    r.resources.get("mean_cpu_percent"),
                ]
            )

    for r in runs:
        if r.capacity_series:
            _dump_csv(out / f"capacity_{r.mode}_t{r.trial}.csv", r.capacity_series)
        if r.decisions:
            _dump_csv(out / f"decisions_{r.mode}_t{r.trial}.csv", r.decisions)
        if r.epochs:
            _dump_csv(out / f"epochs_{r.mode}_t{r.trial}.csv", r.epochs)


def _dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
