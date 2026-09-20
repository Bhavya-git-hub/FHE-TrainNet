"""One configuration object, saved with every result.

Reproducibility is a requirement (SRS NFR-10), and the way it is usually lost is
not malice but convenience: a constant typed at a call site, a default changed
between runs, a parameter that lived only in someone's shell history. So every
tunable in this project lives in `ExperimentConfig`, and the config is written
into each run's directory alongside a snapshot of the environment that produced
it.

Nothing secret goes into the snapshot. The secret key is represented, if at all,
by a truncated hash - enough to tell two runs apart, useless for decryption.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.crypto.backend import CKKSParams
from src.model.network import ModelConfig

ROOT = Path(__file__).resolve().parents[2]


class Mode:
    """The experiment modes the Technical Design section 6 requires."""

    PLAINTEXT = "plaintext"
    FHE_BASELINE = "fhe_baseline"
    FHE_ADAPTIVE = "fhe_adaptive"
    FHE_NO_REFRESH = "fhe_no_refresh"

    ALL = (PLAINTEXT, FHE_BASELINE, FHE_ADAPTIVE, FHE_NO_REFRESH)
    ENCRYPTED = (FHE_BASELINE, FHE_ADAPTIVE, FHE_NO_REFRESH)


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce a run, and nothing that must stay secret."""

    name: str = "demo"
    dataset: str = "iris_binary"
    n_samples: int | None = 100
    test_fraction: float = 0.3
    feature_range: float = 3.0
    split_seed: int = 20260916

    activation: str = "sigmoid_deg3"
    learning_rate: float = 0.8
    batch_size: int = 32
    epochs: int = 6
    init_scale: float = 0.1
    model_seed: int = 20260916

    poly_modulus_degree: int = 16384
    coeff_mod_bit_sizes: tuple[int, ...] = (45, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 40)
    scale_bits: int = 35

    # Refresh policy
    baseline_interval: int = 1
    adaptive_safety_margin: int = 0
    adaptive_min_precision_bits: float | None = None

    modes: tuple[str, ...] = (Mode.PLAINTEXT, Mode.FHE_BASELINE, Mode.FHE_ADAPTIVE)
    trials: int = 1
    measure_every: int = 1
    use_canary: bool = True
    max_seconds_per_run: float | None = 900.0

    notes: str = ""

    def __post_init__(self) -> None:
        self.coeff_mod_bit_sizes = tuple(int(b) for b in self.coeff_mod_bit_sizes)
        self.modes = tuple(self.modes)
        unknown = [m for m in self.modes if m not in Mode.ALL]
        if unknown:
            raise ValueError(f"Unknown mode(s) {unknown}. Valid modes: {list(Mode.ALL)}.")
        if self.trials < 1:
            raise ValueError(f"trials must be at least 1; got {self.trials}.")
        # Validates the CKKS parameters eagerly, so a bad config fails before a
        # long run rather than 30 seconds into key generation.
        self.ckks_params()

    def ckks_params(self) -> CKKSParams:
        return CKKSParams(
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            scale_bits=self.scale_bits,
        )

    def model_config(self, n_features: int) -> ModelConfig:
        return ModelConfig(
            n_features=n_features,
            activation=self.activation,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            init_scale=self.init_scale,
            seed=self.model_seed,
        )

    def depth_budget(self) -> dict[str, Any]:
        """How the configured chain compares with what one step needs.

        Surfaced in the dashboard before a run starts, because "your fixed
        interval cannot possibly work with this activation" is worth knowing
        before waiting for the failure.
        """
        params = self.ckks_params()
        per_step = self.model_config(1).depth_per_step()
        return {
            "max_depth": params.max_depth,
            "depth_per_step": per_step,
            "steps_between_refresh": params.max_depth // per_step if per_step else None,
            "baseline_interval": self.baseline_interval,
            "baseline_is_feasible": self.baseline_interval * per_step <= params.max_depth,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["coeff_mod_bit_sizes"] = list(self.coeff_mod_bit_sizes)
        data["modes"] = list(self.modes)
        data["derived"] = self.depth_budget()
        return data

    def fingerprint(self) -> str:
        """Stable hash of the configuration, for grouping repeated trials."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001 - dataclass introspection
        unexpected = set(data) - known - {"derived"}
        if unexpected:
            raise ValueError(
                f"Unknown configuration key(s): {sorted(unexpected)}. "
                f"Valid keys: {sorted(known)}."
            )
        clean = {k: v for k, v in data.items() if k in known}
        if "coeff_mod_bit_sizes" in clean:
            clean["coeff_mod_bit_sizes"] = tuple(clean["coeff_mod_bit_sizes"])
        if "modes" in clean:
            clean["modes"] = tuple(clean["modes"])
        return cls(**clean)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        if not p.exists():
            available = sorted(x.name for x in (ROOT / "configs").glob("*.yaml"))
            raise FileNotFoundError(
                f"Config not found: {p}. Available in configs/: {available}"
            )
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(
                {k: v for k, v in self.to_dict().items() if k != "derived"}, sort_keys=False
            ),
            encoding="utf-8",
        )


def reproducibility_snapshot(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Environment facts recorded with every run. Contains no key material."""
    import numpy

    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
    }
    for name in ("tenseal", "sklearn", "pandas", "plotly", "streamlit", "psutil"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            # Recorded as absent rather than omitted: a missing optional dependency
            # is a fact about the run.
            versions[name] = "not installed"

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": versions,
    }
    if extra:
        snapshot.update(extra)
    return snapshot
