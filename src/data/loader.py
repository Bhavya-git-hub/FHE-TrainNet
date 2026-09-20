"""Loading, validating and preparing datasets for encrypted training.

Two constraints shape this module and neither is ordinary ML hygiene:

1. Everything ships as a CSV in `datasets/`. Nothing is downloaded at run time,
   so the demonstration works on a disconnected laptop and the exact bytes used
   for a published result are in the repository.
2. Features are scaled to a bounded range, because the polynomial activation is
   only faithful on an interval. A feature that drifts outside it does not raise
   - it silently returns a wrong number. `feature_range` is therefore part of the
   experiment configuration, and `Dataset.range_warnings` reports any column that
   had to be clipped rather than clipping quietly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets"


class DatasetError(ValueError):
    """A dataset could not be loaded or does not meet the stated contract."""


@dataclass
class Dataset:
    """A loaded, validated numerical classification dataset."""

    name: str
    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    source: str = ""
    range_warnings: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "class_balance": {
                str(int(c)): int((self.labels == c).sum()) for c in np.unique(self.labels)
            },
            "source": self.source,
            "range_warnings": self.range_warnings,
        }


@dataclass
class Split:
    """A train/test split plus the scaler statistics used to produce it.

    The statistics are kept because the test set must be transformed with the
    *training* mean and scale. Recomputing them on the test set would leak and
    would also silently change the range the activation polynomial sees.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]
    clipped_fraction: float = 0.0


def available_datasets() -> list[str]:
    """Names of the CSVs bundled in `datasets/`."""
    if not DATASET_DIR.exists():
        return []
    return sorted(p.stem for p in DATASET_DIR.glob("*.csv"))


def load_dataset(name_or_path: str | Path) -> Dataset:
    """Load a bundled dataset by name, or any CSV by path.

    The CSV contract is one header row, numeric feature columns, and a final
    column named `label` holding 0/1. Violations raise `DatasetError` with the
    offending row, because "invalid dataset" is one of the failures the
    specification requires be reported understandably.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = DATASET_DIR / f"{name_or_path}.csv"
    if not path.exists():
        known = ", ".join(available_datasets()) or "(none found)"
        raise DatasetError(
            f"Dataset not found: {path}. Bundled datasets are: {known}. "
            "Pass a name from that list, or a path to a CSV whose last column is named 'label'."
        )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise DatasetError(f"{path.name} has no data rows (found {len(rows)} line(s)).")

    header = [h.strip() for h in rows[0]]
    if header[-1].lower() != "label":
        raise DatasetError(
            f"{path.name}: the last column must be named 'label'; found '{header[-1]}'. "
            f"Header was: {header}"
        )
    feature_names = header[:-1]
    if not feature_names:
        raise DatasetError(f"{path.name}: no feature columns before 'label'.")

    features: list[list[float]] = []
    labels: list[float] = []
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise DatasetError(
                f"{path.name} line {line_no}: expected {len(header)} values, found {len(row)}."
            )
        try:
            values = [float(v) for v in row]
        except ValueError as exc:
            raise DatasetError(
                f"{path.name} line {line_no}: every value must be numeric. {exc}"
            ) from exc
        if any(not np.isfinite(v) for v in values):
            raise DatasetError(f"{path.name} line {line_no}: contains NaN or infinity.")
        features.append(values[:-1])
        labels.append(values[-1])

    y = np.asarray(labels, dtype=float)
    unique = np.unique(y)
    if not set(unique.tolist()) <= {0.0, 1.0}:
        raise DatasetError(
            f"{path.name}: 'label' must contain only 0 and 1 for this binary prototype; "
            f"found {unique.tolist()}."
        )
    if len(unique) < 2:
        raise DatasetError(
            f"{path.name}: only one class present ({unique.tolist()}); accuracy would be "
            "meaningless."
        )

    return Dataset(
        name=path.stem,
        features=np.asarray(features, dtype=float),
        labels=y,
        feature_names=feature_names,
        source=str(path),
    )


def prepare(
    dataset: Dataset,
    *,
    test_fraction: float = 0.3,
    seed: int = 20260916,
    n_samples: int | None = None,
    feature_range: float = 3.0,
) -> Split:
    """Subsample, split, standardize and clip into the activation's safe range.

    `feature_range` is not cosmetic. CKKS evaluates a polynomial approximation of
    the activation, which is faithful only on a bounded interval; a standardized
    outlier at 8 sigma leaves that interval and the polynomial returns a confidently
    wrong value with no error raised. Clipping is therefore a correctness measure,
    and the fraction of values clipped is recorded and reported rather than hidden.
    """
    if not 0.05 <= test_fraction <= 0.9:
        raise DatasetError(f"test_fraction must be between 0.05 and 0.9; got {test_fraction}.")
    if feature_range <= 0:
        raise DatasetError(f"feature_range must be positive; got {feature_range}.")

    rng = np.random.default_rng(seed)
    x, y = dataset.features, dataset.labels

    if n_samples is not None:
        if n_samples < 4:
            raise DatasetError(f"n_samples must be at least 4 to form a split; got {n_samples}.")
        n_samples = min(n_samples, len(y))
        # Stratify the subsample so a small demo cannot accidentally become
        # single-class, which would make accuracy meaningless.
        idx = _stratified_subsample(y, n_samples, rng)
        x, y = x[idx], y[idx]

    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    n_test = max(2, int(round(len(y) * test_fraction)))
    if len(y) - n_test < 2:
        raise DatasetError(
            f"A {test_fraction:.0%} test split of {len(y)} samples leaves too few for training."
        )
    x_test, y_test = x[:n_test], y[:n_test]
    x_train, y_train = x[n_test:], y[n_test:]

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    # A constant column has zero spread; dividing by it would produce inf. Leave it
    # at its (zero) centred value instead of inventing variation for it.
    scale = np.where(scale < 1e-12, 1.0, scale)

    x_train_s = (x_train - mean) / scale
    x_test_s = (x_test - mean) / scale

    total = x_train_s.size + x_test_s.size
    clipped = int((np.abs(x_train_s) > feature_range).sum() + (np.abs(x_test_s) > feature_range).sum())
    x_train_s = np.clip(x_train_s, -feature_range, feature_range)
    x_test_s = np.clip(x_test_s, -feature_range, feature_range)

    return Split(
        x_train=x_train_s,
        y_train=y_train,
        x_test=x_test_s,
        y_test=y_test,
        mean=mean,
        scale=scale,
        feature_names=list(dataset.feature_names),
        clipped_fraction=clipped / total if total else 0.0,
    )


def _stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick `n` indices keeping the class proportions of `y`."""
    classes = np.unique(y)
    picks: list[np.ndarray] = []
    for cls in classes:
        pool = np.flatnonzero(y == cls)
        take = max(2, int(round(n * len(pool) / len(y))))
        take = min(take, len(pool))
        picks.append(rng.choice(pool, size=take, replace=False))
    idx = np.concatenate(picks)
    return idx[rng.permutation(len(idx))]
