"""End-to-end encrypted training, the experiment runner, and export.

`test_encrypted_training_matches_the_plaintext_mirror` is the load-bearing test
here. The two trainers are supposed to be running the same algorithm, and if they
ever drift apart the comparison between them becomes meaningless while still
producing perfectly plausible numbers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.bootstrapping.policies import AdaptivePolicy, BaselinePolicy, NoRefreshPolicy
from src.crypto.backend import CKKSParams, RefreshKind
from src.experiments.config import ExperimentConfig, Mode, reproducibility_snapshot
from src.experiments.registry import list_runs, load_run, load_series
from src.experiments.runner import run_experiment, run_single
from src.model.network import ModelConfig
from src.noise.monitor import CapacityMonitor
from src.training.encrypted_trainer import EncryptedTrainer
from src.training.plaintext_trainer import train_plaintext


def make_trainer(owner, params, split, policy, activation="sigmoid_deg1"):
    config = ModelConfig(
        n_features=split.x_train.shape[1],
        activation=activation,
        learning_rate=0.8,
        batch_size=14,
        epochs=1,
        seed=7,
    )
    monitor = CapacityMonitor(params)
    return EncryptedTrainer(owner, owner.backend(), config, policy, monitor), config, monitor


# -- encrypted training ---------------------------------------------------


def test_encrypted_training_runs_and_produces_a_model(owner, params, smoke_split) -> None:
    trainer, _, _ = make_trainer(owner, params, smoke_split, AdaptivePolicy())
    result = trainer.train(smoke_split)
    assert result.status == "SUCCEEDED", result.reason
    assert result.steps > 0
    assert result.model is not None
    assert result.final_test_accuracy is not None


def test_encrypted_training_matches_the_plaintext_mirror(owner, params, smoke_split) -> None:
    """Same algorithm, two arithmetics. Any gap must be CKKS approximation alone.

    The tolerance is tight on purpose. These test parameters measure about 13 bits
    of agreement per operation (~1.2e-4 relative) and a step chains several
    multiplications, so a correct implementation lands within a few parts in a
    thousand. Anything looser would not notice the two trainers taking different
    batch orders - which is exactly the bug this tolerance was tightened to catch,
    after the encrypted trainer was found drawing one extra permutation from the
    shared seed before the loop started.
    """
    trainer, config, _ = make_trainer(owner, params, smoke_split, AdaptivePolicy())
    encrypted = trainer.train(smoke_split)
    plain = train_plaintext(smoke_split, config)

    assert encrypted.status == "SUCCEEDED", encrypted.reason
    assert encrypted.steps == plain.steps
    np.testing.assert_allclose(
        encrypted.model.weights, plain.model.weights, rtol=5e-3, atol=5e-4,
        err_msg="the encrypted and plaintext trainers have drifted apart",
    )
    # The bias is compared on an absolute tolerance. It is a small number (~0.08)
    # reached by summing errors across the batch, so CKKS's absolute error floor
    # dominates it and a relative bound would be testing the wrong quantity.
    assert encrypted.model.bias == pytest.approx(plain.model.bias, abs=1e-3)
    # What the drift must not do is change the answer.
    assert encrypted.final_test_accuracy == pytest.approx(plain.final_test_accuracy, abs=1e-9)


def test_the_model_is_encrypted_throughout_training(owner, params, smoke_split) -> None:
    """Weight ciphertexts must not be decryptable by the compute zone."""
    trainer, _, _ = make_trainer(owner, params, smoke_split, AdaptivePolicy())
    weights, _ = trainer._encrypt_weights(
        __import__("src.model.network", fromlist=["Model"]).Model.initial(trainer.config), 14
    )
    with pytest.raises(Exception, match="(?i)secret_key"):
        weights[0].raw.decrypt()


def test_encrypted_ops_are_counted(owner, params, smoke_split) -> None:
    trainer, _, _ = make_trainer(owner, params, smoke_split, AdaptivePolicy())
    result = trainer.train(smoke_split)
    assert result.encrypted_ops["mul_ct_ct"] > 0
    assert result.encrypted_ops["polyval"] > 0
    assert sum(result.encrypted_ops.values()) == result.to_dict()["encrypted_ops_total"]


def test_adaptive_refreshes_no_more_often_than_a_conservative_baseline(
    owner, params, smoke_split
) -> None:
    adaptive, _, _ = make_trainer(owner, params, smoke_split, AdaptivePolicy())
    adaptive.train(smoke_split)
    baseline, _, _ = make_trainer(owner, params, smoke_split, BaselinePolicy(interval=1))
    baseline.train(smoke_split)
    assert adaptive.log.refresh_count <= baseline.log.refresh_count


def test_no_refresh_exhausts_capacity_and_is_recorded_as_failed(
    owner, params, smoke_split
) -> None:
    """The control that proves the limit is real."""
    trainer, _, _ = make_trainer(
        owner, params, smoke_split, NoRefreshPolicy(), activation="sigmoid_deg1"
    )
    trainer.config.epochs = 4
    result = trainer.train(smoke_split)
    assert result.status == "FAILED"
    assert "computation capacity" in (result.reason or "")
    assert trainer.log.refresh_count == 0


def test_depth_accounting_is_consistent_with_measurement(owner, params, smoke_split) -> None:
    trainer, _, monitor = make_trainer(owner, params, smoke_split, AdaptivePolicy())
    trainer.train(smoke_split)
    assert monitor.summary()["consistency_failures"] == 0, monitor.inconsistencies


def test_refresh_kind_is_recorded_on_every_decision(owner, params, smoke_split) -> None:
    trainer, _, _ = make_trainer(owner, params, smoke_split, BaselinePolicy(interval=1))
    trainer.train(smoke_split)
    assert trainer.log.records
    for record in trainer.log.rows():
        assert record["refresh_kind"] == RefreshKind.CLIENT_AIDED.value


# -- configuration --------------------------------------------------------


def test_config_round_trips_through_yaml(tmp_path: Path, smoke_config) -> None:
    path = tmp_path / "c.yaml"
    smoke_config.save(path)
    restored = ExperimentConfig.load(path)
    assert restored.to_dict() == smoke_config.to_dict()
    assert restored.fingerprint() == smoke_config.fingerprint()


def test_config_rejects_unknown_keys() -> None:
    """A typo must not be silently ignored into a different experiment."""
    with pytest.raises(ValueError, match="Unknown configuration key"):
        ExperimentConfig.from_dict({"epocs": 5})


def test_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        ExperimentConfig(modes=("plaintext", "fhe_magic"))


def test_config_validates_ckks_parameters_eagerly() -> None:
    from src.crypto.backend import ParameterError

    with pytest.raises(ParameterError):
        ExperimentConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 40, 60))


def test_depth_budget_flags_an_infeasible_baseline() -> None:
    config = ExperimentConfig.load(Path("configs") / "fast_smoke.yaml")
    config.baseline_interval = 99
    budget = config.depth_budget()
    assert budget["baseline_is_feasible"] is False


def test_fingerprint_changes_with_the_configuration(smoke_config) -> None:
    other = ExperimentConfig.from_dict({**{k: v for k, v in smoke_config.to_dict().items()
                                           if k != "derived"}, "epochs": 99})
    assert other.fingerprint() != smoke_config.fingerprint()


def test_reproducibility_snapshot_records_versions_and_no_secrets() -> None:
    snapshot = reproducibility_snapshot()
    assert "tenseal" in snapshot["versions"]
    assert snapshot["timestamp_utc"]
    blob = json.dumps(snapshot).lower()
    for forbidden in ("secret", "private_key", "sk="):
        assert forbidden not in blob


@pytest.mark.parametrize(
    "name", ["demo.yaml", "benchmark.yaml", "no_refresh_control.yaml", "fast_smoke.yaml"]
)
def test_every_shipped_config_is_valid(name: str) -> None:
    config = ExperimentConfig.load(Path("configs") / name)
    assert config.ckks_params().max_depth >= 1


# -- the runner and export ------------------------------------------------


def test_run_experiment_writes_a_complete_results_directory(
    tmp_path: Path, smoke_config
) -> None:
    payload = run_experiment(smoke_config, results_dir=tmp_path)
    out = tmp_path / payload["run_id"]
    assert (out / "results.json").exists()
    assert (out / "config.yaml").exists()
    assert (out / "summary.csv").exists()

    stored = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert stored["config"]["dataset"] == smoke_config.dataset
    assert stored["reproducibility"]["versions"]["tenseal"]
    assert stored["capacity_metric"]["name"].startswith("Ciphertext Level")
    assert "no CKKS noise budget" in stored["capacity_metric"]["disclaimer"]

    with (out / "summary.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["mode"] for r in rows} == set(smoke_config.modes)


def test_plaintext_metrics_are_none_not_zero(tmp_path: Path, smoke_config) -> None:
    """A plaintext run needed no refreshes; that is not the same as needing zero."""
    split_config = ExperimentConfig.from_dict(
        {**{k: v for k, v in smoke_config.to_dict().items() if k != "derived"},
         "modes": ["plaintext"]}
    )
    payload = run_experiment(split_config, results_dir=tmp_path)
    metrics = payload["runs"][0]["metrics"]
    for key in ("refreshes", "refresh_seconds_total", "refresh_kind", "encrypted_ops_total"):
        assert metrics[key] is None, f"{key} should be None for a plaintext run"


def test_registry_reads_back_what_the_runner_wrote(tmp_path: Path, smoke_config) -> None:
    payload = run_experiment(smoke_config, results_dir=tmp_path)
    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].run_id == payload["run_id"]
    assert runs[0].error is None

    restored = load_run(payload["run_id"], tmp_path)
    assert restored["run_id"] == payload["run_id"]

    series = load_series(payload["run_id"], "capacity", Mode.FHE_ADAPTIVE, 0, tmp_path)
    assert series, "capacity series should have been written"
    assert all("levels_remaining" in row for row in series)


def test_registry_reports_a_broken_directory_rather_than_hiding_it(tmp_path: Path) -> None:
    (tmp_path / "20260101-000000-broken").mkdir()
    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert "no results.json" in runs[0].error


def test_csv_empty_cells_read_back_as_none(tmp_path: Path) -> None:
    from src.experiments.registry import _coerce

    assert _coerce("") is None
    assert _coerce("3") == 3
    assert _coerce("3.5") == 3.5
    assert _coerce("True") is True


def test_comparison_verdict_is_generated_from_measurements(tmp_path: Path) -> None:
    config = ExperimentConfig.load(Path("configs") / "fast_smoke.yaml")
    config.modes = (Mode.PLAINTEXT, Mode.FHE_BASELINE, Mode.FHE_ADAPTIVE)
    payload = run_experiment(config, results_dir=tmp_path)
    verdict = " ".join(payload["comparison"]["verdict"])
    assert "refresh" in verdict.lower()
    assert "accuracy" in verdict.lower()


def test_registry_ignores_non_run_directories(tmp_path: Path, smoke_config) -> None:
    """`demos/` sorts above a timestamped run id and must not be mistaken for one.

    It previously was, which made the dashboard report "no experiment has been run
    yet" while several completed runs sat next to it on disk.
    """
    payload = run_experiment(smoke_config, results_dir=tmp_path)
    (tmp_path / "demos").mkdir()
    (tmp_path / "openfhe").mkdir()

    runs = list_runs(tmp_path)
    assert [r.run_id for r in runs] == [payload["run_id"]]
    assert all(r.error is None for r in runs)


def test_registry_orders_runs_newest_first(tmp_path: Path) -> None:
    for name in ("20260101-000000-a-aaa", "20260301-000000-c-ccc", "20260201-000000-b-bbb"):
        (tmp_path / name).mkdir()
    assert [r.run_id for r in list_runs(tmp_path)] == [
        "20260301-000000-c-ccc", "20260201-000000-b-bbb", "20260101-000000-a-aaa",
    ]


def test_an_undated_run_never_outranks_a_dated_one(tmp_path: Path) -> None:
    """`results/reference/` is a real run, but it is never the most recent one.

    It holds a `results.json`, so it is listed - it is a genuine recorded run,
    committed so a fresh deployment has measurements to display. But its name
    carries no timestamp, and sorting by name put "reference" above every
    "2026...." because letters sort after digits. It silently became "the latest
    run": the dashboard and `make_report.py` reported on the shipped reference
    instead of the run the user had just finished, and nothing on screen said so.

    Undo `_recency_key` and this passes for the dated runs while the reference
    quietly takes first place again.
    """
    for name in ("20260101-000000-a-aaa", "20260301-000000-c-ccc"):
        (tmp_path / name).mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "results.json").write_text(
        json.dumps({"run_id": "reference", "runs": []}), encoding="utf-8"
    )

    ids = [r.run_id for r in list_runs(tmp_path)]
    assert ids[0] == "20260301-000000-c-ccc"
    assert ids == ["20260301-000000-c-ccc", "20260101-000000-a-aaa", "reference"]
