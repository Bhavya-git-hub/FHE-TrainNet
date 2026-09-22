"""Reading back what previous runs recorded.

The dashboard's history and comparison views read from here rather than holding
state, so a run made from the command line and a run made from the UI are the
same kind of object and appear in the same list.

A results directory that cannot be parsed is reported as broken rather than
skipped silently - a run that vanishes from the history is worse than one that
appears with an error next to it.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"


@dataclass
class RunSummary:
    """One stored experiment, as the history table sees it."""

    run_id: str
    path: Path
    timestamp: str = ""
    name: str = ""
    dataset: str = ""
    modes: list[str] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""
    error: str | None = None
    payload: dict[str, Any] | None = None

    def row(self) -> dict[str, Any]:
        comparison = (self.payload or {}).get("comparison", {}).get("by_mode", {})
        adaptive = comparison.get("fhe_adaptive", {})
        baseline = comparison.get("fhe_baseline", {})
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "experiment": self.name,
            "dataset": self.dataset,
            "modes": ", ".join(self.modes),
            "status": ", ".join(f"{k}={v}" for k, v in self.statuses.items()),
            "adaptive_refreshes": adaptive.get("refreshes", {}).get("mean"),
            "baseline_refreshes": baseline.get("refreshes", {}).get("mean"),
            "adaptive_seconds": adaptive.get("train_seconds", {}).get("mean"),
            "baseline_seconds": baseline.get("train_seconds", {}).get("mean"),
            "adaptive_accuracy": adaptive.get("test_accuracy", {}).get("mean"),
            "config_fingerprint": self.fingerprint,
            "error": self.error,
        }


# Run directories are named `<date>-<time>-<name>-<hex>`. Other directories live
# under `results/` too - `demos/` and `openfhe/` hold output from the demo and
# container scripts - and they are not runs. Matching the shape explicitly keeps
# them out of the history rather than listing them as broken runs.
RUN_DIR_PATTERN = re.compile(r"^\d{8}-\d{6}-")


def looks_like_a_run(path: Path) -> bool:
    return path.is_dir() and (
        bool(RUN_DIR_PATTERN.match(path.name)) or (path / "results.json").exists()
    )


def _recency_key(path: Path) -> tuple[int, str]:
    """Sort key for `list_runs`, newest first under `reverse=True`.

    A run directory carries its timestamp in its name, so sorting those by name
    orders them by time. A directory that holds a `results.json` without that
    prefix is still a real run - `results/reference/` is one, committed so a fresh
    deployment has measurements to show - but it has no timestamp, and sorting it
    by name put "reference" above every "2026...." because letters sort after
    digits. It became "the most recent run", so the dashboard and
    `make_report.py` reported on the shipped reference instead of the run the
    user had just finished, with nothing on screen saying so.

    This is the `demos/` defect from before, reintroduced by a directory added
    for a different reason. Any undated run now sorts below every dated one
    instead of relying on its name.
    """
    return (1, path.name) if RUN_DIR_PATTERN.match(path.name) else (0, path.name)


def list_runs(results_dir: Path | None = None) -> list[RunSummary]:
    """Every run directory, newest first.

    Sorted by the timestamp embedded in the directory name rather than by string
    order over everything in `results/`, which previously let `demos/` sort above
    the real runs and become "the most recent run". See `_recency_key` - the
    sorting is the whole point of this function and it has been wrong twice.
    """
    base = Path(results_dir or RESULTS_DIR)
    if not base.exists():
        return []
    summaries: list[RunSummary] = []
    for path in sorted(
        (p for p in base.iterdir() if looks_like_a_run(p)),
        key=_recency_key,
        reverse=True,
    ):
        results = path / "results.json"
        if not results.exists():
            summaries.append(
                RunSummary(
                    run_id=path.name, path=path,
                    error="no results.json (the run may have been interrupted)",
                )
            )
            continue
        try:
            payload = json.loads(results.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            summaries.append(
                RunSummary(run_id=path.name, path=path, error=f"unreadable: {exc}")
            )
            continue
        config = payload.get("config", {})
        summaries.append(
            RunSummary(
                run_id=payload.get("run_id", path.name),
                path=path,
                timestamp=payload.get("reproducibility", {}).get("timestamp_utc", ""),
                name=config.get("name", ""),
                dataset=config.get("dataset", ""),
                modes=[r["mode"] for r in payload.get("runs", [])],
                statuses={r["mode"]: r["status"] for r in payload.get("runs", [])},
                fingerprint=payload.get("config_fingerprint", ""),
                payload=payload,
            )
        )
    return summaries


def load_run(run_id: str, results_dir: Path | None = None) -> dict[str, Any]:
    base = Path(results_dir or RESULTS_DIR)
    path = base / run_id / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"No stored run with id '{run_id}' under {base}.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_series(run_id: str, kind: str, mode: str, trial: int = 0,
                results_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read one of the per-run CSVs (`capacity`, `decisions`, `epochs`)."""
    base = Path(results_dir or RESULTS_DIR)
    path = base / run_id / f"{kind}_{mode}_t{trial}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [{k: _coerce(v) for k, v in row.items()} for row in rows]


def _coerce(value: str) -> Any:
    """Turn CSV text back into numbers, preserving empty as None.

    An empty cell means the quantity was not measured. It becomes `None`, never
    `0`, so a chart shows a gap instead of a fabricated data point.
    """
    if value == "" or value is None:
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value
