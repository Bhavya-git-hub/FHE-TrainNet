"""Run an experiment from a configuration file and write its results.

    python scripts/run_experiment.py --config configs/demo.yaml
    python scripts/run_experiment.py --config configs/demo.yaml --modes plaintext fhe_adaptive
    python scripts/run_experiment.py --config configs/demo.yaml --set epochs=3 batch_size=16

Everything printed here comes from a measurement taken during this run. The
summary at the end states what the numbers show, including when they show the
adaptive policy did no better.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import require_dependencies  # noqa: E402

require_dependencies(script="scripts/run_experiment.py")

from src.experiments.config import ExperimentConfig, Mode  # noqa: E402
from src.experiments.runner import RESULTS_DIR, run_experiment  # noqa: E402


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [_coerce(v.strip()) for v in value.split(",")]
    return value


class Reporter:
    """Prints progress without drowning a long encrypted run in output."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.last = 0.0
        self.mode = ""

    def __call__(self, event: dict[str, Any]) -> None:
        if self.quiet:
            return
        kind = event.get("event")
        if kind == "mode_start":
            self.mode = event["mode"]
            print(f"\n--- {self.mode} (trial {event['trial']}) ---", flush=True)
            return
        if kind == "mode_end":
            m = event["metrics"]
            status = event["status"]
            acc = m.get("final_test_accuracy")
            print(
                f"    {status}: {m.get('train_seconds', 0):.1f}s, "
                f"accuracy {'n/a' if acc is None else f'{acc:.3f}'}, "
                f"refreshes {m.get('refreshes')}",
                flush=True,
            )
            return
        now = time.perf_counter()
        if now - self.last < 0.5:
            return
        self.last = now
        if event.get("mode") == "encrypted":
            print(
                f"    epoch {event['epoch']}/{event['epochs']} step {event['step']:>3}  "
                f"levels {event['levels_before']}->{event['levels_after']}"
                f"/{event['max_depth']} (needs {event['levels_needed']})  "
                f"{event['decision']:<8} refreshes={event['refreshes']}  "
                f"{event['elapsed']:.1f}s",
                flush=True,
            )
        elif event.get("mode") == "plaintext":
            print(
                f"    epoch {event['epoch']}/{event['epochs']}  loss {event['loss']:.4f}  "
                f"test acc {event['test_accuracy']:.3f}",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo.yaml"))
    parser.add_argument("--modes", nargs="*", choices=list(Mode.ALL))
    parser.add_argument("--trials", type=int)
    parser.add_argument("--name")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override configuration fields, e.g. --set epochs=3 batch_size=16",
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    overrides: dict[str, Any] = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects KEY=VALUE; got '{item}'")
        key, _, value = item.partition("=")
        overrides[key.strip()] = _coerce(value.strip())
    if overrides:
        data = config.to_dict()
        data.pop("derived", None)
        data.update(overrides)
        config = ExperimentConfig.from_dict(data)
    if args.modes:
        config.modes = tuple(args.modes)
    if args.trials:
        config.trials = args.trials
    if args.name:
        config.name = args.name

    budget = config.depth_budget()
    print(f"Experiment '{config.name}'  dataset={config.dataset}  modes={list(config.modes)}")
    print(
        f"Depth budget: chain provides {budget['max_depth']} level(s); one training step costs "
        f"{budget['depth_per_step']}; {budget['steps_between_refresh']} step(s) fit between "
        "refreshes."
    )
    if not budget["baseline_is_feasible"]:
        print(
            f"  NOTE: the fixed baseline interval of {budget['baseline_interval']} step(s) needs "
            f"{budget['baseline_interval'] * budget['depth_per_step']} levels but only "
            f"{budget['max_depth']} are available. That run is expected to FAIL, which is a "
            "legitimate result and will be recorded as such."
        )

    payload = run_experiment(
        config, progress=Reporter(args.quiet), results_dir=Path(args.out_dir)
    )

    print("\n=== Summary ===")
    for mode, stats in payload["comparison"]["by_mode"].items():
        acc = stats["test_accuracy"]["mean"]
        secs = stats["train_seconds"]["mean"]
        refs = stats["refreshes"]["mean"]
        print(
            f"  {mode:<16} {stats['succeeded']}/{stats['trials']} ok  "
            f"time={'n/a' if secs is None else f'{secs:8.2f}s'}  "
            f"acc={'n/a' if acc is None else f'{acc:.3f}'}  "
            f"refreshes={'n/a' if refs is None else f'{refs:.1f}'}"
        )
        for reason in stats["reasons"]:
            print(f"      reason: {reason}")
    print()
    for line in payload["comparison"]["verdict"]:
        print(f"  * {line}")
    print(f"\nResults written to: {Path(args.out_dir) / payload['run_id']}")

    failed = any(
        s["succeeded"] < s["trials"] and mode != Mode.FHE_NO_REFRESH
        for mode, s in payload["comparison"]["by_mode"].items()
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
