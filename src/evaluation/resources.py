"""CPU and memory measurement.

Resource figures are among the easiest numbers to fabricate by accident: sample
at the wrong moment and you report the interpreter's idle footprint instead of
the run's. This sampler runs on a background thread for the duration of a run and
reports peak and mean.

If `psutil` is unavailable the readings are `None`, not `0.0`. A missing
measurement and a measurement of zero are different claims, and the results
files keep them different (Master Prompt 24).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only on installs without psutil
    psutil = None  # type: ignore[assignment]


@dataclass
class ResourceSummary:
    available: bool
    peak_rss_mb: float | None = None
    mean_rss_mb: float | None = None
    baseline_rss_mb: float | None = None
    peak_cpu_percent: float | None = None
    mean_cpu_percent: float | None = None
    samples: int = 0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "samples": self.samples,
            "peak_rss_mb": self.peak_rss_mb,
            "mean_rss_mb": self.mean_rss_mb,
            "baseline_rss_mb": self.baseline_rss_mb,
            "peak_rss_above_baseline_mb": (
                None
                if self.peak_rss_mb is None or self.baseline_rss_mb is None
                else self.peak_rss_mb - self.baseline_rss_mb
            ),
            "peak_cpu_percent": self.peak_cpu_percent,
            "mean_cpu_percent": self.mean_cpu_percent,
        }


class ResourceSampler:
    """Samples this process's RSS and CPU while a run is in progress."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._rss: list[float] = []
        self._cpu: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: float | None = None
        self._proc = psutil.Process() if psutil is not None else None

    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._proc is None:
            return
        self._baseline = self._proc.memory_info().rss / 1e6
        # First call establishes the interval for subsequent cpu_percent() reads;
        # its own return value is meaningless and is discarded.
        self._proc.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._proc is not None
        while not self._stop.is_set():
            try:
                self._rss.append(self._proc.memory_info().rss / 1e6)
                self._cpu.append(self._proc.cpu_percent(None))
            except Exception:  # noqa: BLE001 - a dead process ends sampling, reported via samples=0
                break
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> ResourceSummary:
        if psutil is None:
            return ResourceSummary(
                available=False,
                reason="psutil is not installed; CPU and memory were not measured.",
            )
        if not self._rss:
            return ResourceSummary(
                available=False,
                reason="the run finished before the sampler took a reading.",
            )
        cpu = [c for c in self._cpu if c is not None]
        return ResourceSummary(
            available=True,
            peak_rss_mb=max(self._rss),
            mean_rss_mb=sum(self._rss) / len(self._rss),
            baseline_rss_mb=self._baseline,
            peak_cpu_percent=max(cpu) if cpu else None,
            mean_cpu_percent=(sum(cpu) / len(cpu)) if cpu else None,
            samples=len(self._rss),
        )
