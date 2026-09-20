"""Dataset loading, preprocessing, activations and the plaintext trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.loader import DatasetError, available_datasets, load_dataset, prepare
from src.model.activation import ACTIVATIONS, SIGMOID_DEG3, fit_polynomial, get_activation, sigmoid
from src.model.network import Model, ModelConfig, accuracy, batches, squared_loss
from src.training.plaintext_trainer import train_plaintext


# -- datasets -------------------------------------------------------------


def test_bundled_datasets_exist() -> None:
    names = available_datasets()
    assert {"iris_binary", "breast_cancer_top8", "two_moons"} <= set(names)


@pytest.mark.parametrize("name", ["iris_binary", "breast_cancer_top8", "two_moons"])
def test_every_bundled_dataset_loads(name: str) -> None:
    dataset = load_dataset(name)
    assert dataset.n_samples > 0
    assert dataset.n_features > 0
    assert set(np.unique(dataset.labels).tolist()) == {0.0, 1.0}
    assert len(dataset.feature_names) == dataset.n_features


def test_missing_dataset_lists_the_alternatives() -> None:
    with pytest.raises(DatasetError, match="Bundled datasets are"):
        load_dataset("not_a_dataset")


def test_rejects_wrong_label_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b,target\n1,2,0\n3,4,1\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="must be named 'label'"):
        load_dataset(path)


def test_rejects_non_numeric_value_naming_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b,label\n1,2,0\n3,oops,1\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="line 3"):
        load_dataset(path)


def test_rejects_ragged_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b,label\n1,2,0\n3,4\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="expected 3 values"):
        load_dataset(path)


def test_rejects_single_class(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b,label\n1,2,0\n3,4,0\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="only one class"):
        load_dataset(path)


def test_rejects_non_binary_labels(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b,label\n1,2,0\n3,4,2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="only 0 and 1"):
        load_dataset(path)


def test_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b,label\n1,nan,0\n3,4,1\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="NaN or infinity"):
        load_dataset(path)


# -- preprocessing --------------------------------------------------------


def test_prepare_standardises_using_training_statistics_only() -> None:
    split = prepare(load_dataset("breast_cancer_top8"), n_samples=200, seed=3)
    assert split.x_train.mean(axis=0) == pytest.approx(np.zeros(split.x_train.shape[1]), abs=0.2)
    # The test set must NOT be independently standardised.
    assert not np.allclose(split.x_test.mean(axis=0), 0.0, atol=1e-9)


def test_prepare_clips_into_the_activation_range_and_reports_it() -> None:
    split = prepare(load_dataset("breast_cancer_top8"), n_samples=300, seed=5, feature_range=2.0)
    assert np.abs(split.x_train).max() <= 2.0 + 1e-9
    assert np.abs(split.x_test).max() <= 2.0 + 1e-9
    assert split.clipped_fraction > 0


def test_subsample_keeps_both_classes() -> None:
    split = prepare(load_dataset("iris_binary"), n_samples=12, seed=1)
    combined = np.concatenate([split.y_train, split.y_test])
    assert set(np.unique(combined).tolist()) == {0.0, 1.0}


def test_split_is_deterministic_for_a_seed() -> None:
    a = prepare(load_dataset("two_moons"), n_samples=60, seed=42)
    b = prepare(load_dataset("two_moons"), n_samples=60, seed=42)
    assert np.array_equal(a.x_train, b.x_train)
    assert np.array_equal(a.y_test, b.y_test)


def test_rejects_impossible_split() -> None:
    with pytest.raises(DatasetError, match="test_fraction must be"):
        prepare(load_dataset("iris_binary"), test_fraction=0.99)


# -- activations ----------------------------------------------------------


def test_activation_ladder_costs_one_two_and_three_levels() -> None:
    assert ACTIVATIONS["sigmoid_deg1"].depth_cost == 1
    assert ACTIVATIONS["sigmoid_deg3"].depth_cost == 2
    assert ACTIVATIONS["sigmoid_deg5"].depth_cost == 3


def test_higher_degree_approximates_the_sigmoid_better() -> None:
    errors = [
        ACTIVATIONS[name].approximation_error(3.0)["max_abs_error"]
        for name in ("sigmoid_deg1", "sigmoid_deg3", "sigmoid_deg5")
    ]
    assert errors[0] > errors[1] > errors[2]


def test_approximation_error_is_measured_not_claimed() -> None:
    err = SIGMOID_DEG3.approximation_error(3.0)
    grid = np.linspace(-3, 3, 401)
    expected = float(np.abs(SIGMOID_DEG3(grid) - np.asarray(sigmoid(grid))).max())
    assert err["max_abs_error"] == pytest.approx(expected, rel=1e-9)


def test_fit_polynomial_beats_the_literature_constant() -> None:
    fitted = fit_polynomial(3, 3.0)
    assert (
        fitted.approximation_error(3.0)["rms_error"]
        < ACTIVATIONS["sigmoid_deg3_literature"].approximation_error(3.0)["rms_error"]
    )


def test_unknown_activation_lists_the_options() -> None:
    with pytest.raises(ValueError, match="Available:"):
        get_activation("not_an_activation")


def test_degree3_polynomial_stops_being_monotone() -> None:
    """The property that silently destroys training when the input range grows."""
    limit = ACTIVATIONS["sigmoid_deg3"].monotonic_limit
    assert limit is not None
    assert limit == pytest.approx(2.82, abs=0.05)
    # Past the limit the polynomial genuinely decreases as its input grows.
    act = ACTIVATIONS["sigmoid_deg3"]
    assert float(act(limit + 2.0)) < float(act(limit))


def test_linear_activation_is_monotone_everywhere() -> None:
    assert ACTIVATIONS["sigmoid_deg1"].monotonic_limit is None


def test_safe_range_is_the_tighter_of_fit_and_monotonicity() -> None:
    assert ACTIVATIONS["sigmoid_deg3"].safe_input_range(3.0) == pytest.approx(2.82, abs=0.05)
    assert ACTIVATIONS["sigmoid_deg1"].safe_input_range(3.0) == 3.0


def test_divergence_is_detected_and_reported(smoke_split) -> None:
    """A learning rate that pushes z out of range must be reported, not hidden.

    Measured: at lr=3.0 the pre-activation value leaves the degree-3 polynomial's
    monotone range and accuracy collapses, with no exception raised anywhere.
    """
    config = ModelConfig(
        n_features=smoke_split.x_train.shape[1], activation="sigmoid_deg3",
        learning_rate=2.0, batch_size=14, epochs=8,
    )
    result = train_plaintext(smoke_split, config)
    assert result.diverged is True
    assert result.warnings
    assert "inverts the gradient" in result.warnings[0]
    assert result.max_abs_z > ACTIVATIONS["sigmoid_deg3"].safe_input_range()


def test_a_sane_learning_rate_stays_in_range(smoke_split) -> None:
    config = ModelConfig(
        n_features=smoke_split.x_train.shape[1], activation="sigmoid_deg3",
        learning_rate=0.3, batch_size=14, epochs=6,
    )
    result = train_plaintext(smoke_split, config)
    assert result.diverged is False
    assert result.warnings == []


# -- model ----------------------------------------------------------------


def test_depth_per_step_is_three_plus_activation_cost() -> None:
    """A batched step pays for the cross-slot reduction."""
    for name in ("sigmoid_deg1", "sigmoid_deg3", "sigmoid_deg5"):
        config = ModelConfig(n_features=2, activation=name, batch_size=32)
        assert config.depth_per_step(32) == 3 + ACTIVATIONS[name].depth_cost


def test_a_single_slot_step_costs_one_level_less() -> None:
    """With one sample there is nothing to reduce, so `mm` is skipped.

    This is what removes the need for Galois keys, which measured 1901 MB against
    105 MB for the same context - the difference between fitting a 1 GB host and
    not.
    """
    for name in ("sigmoid_deg1", "sigmoid_deg3", "sigmoid_deg5"):
        config = ModelConfig(n_features=2, activation=name, batch_size=1)
        assert config.depth_per_step(1) == 2 + ACTIVATIONS[name].depth_cost
        assert config.depth_per_step(1) == config.depth_per_step(32) - 1


def test_rotation_keys_are_only_needed_for_a_real_batch() -> None:
    assert ModelConfig(n_features=2, batch_size=1).needs_rotation_keys is False
    assert ModelConfig(n_features=2, batch_size=16).needs_rotation_keys is True


def test_weights_are_not_initialised_to_zero() -> None:
    """Zero-valued CKKS ciphertexts have unbounded relative error."""
    model = Model.initial(ModelConfig(n_features=4))
    assert np.all(model.weights != 0.0)


def test_initialisation_is_reproducible() -> None:
    a = Model.initial(ModelConfig(n_features=3, seed=99))
    b = Model.initial(ModelConfig(n_features=3, seed=99))
    assert np.array_equal(a.weights, b.weights)
    assert a.bias == b.bias


def test_batches_drop_the_partial_tail() -> None:
    """Every step must consume identical depth, so no padded final batch."""
    rng = np.random.default_rng(0)
    result = batches(10, 4, rng)
    assert [len(b) for b in result] == [4, 4]


def test_prediction_threshold_follows_the_activation() -> None:
    config = ModelConfig(n_features=1, activation="sigmoid_deg3")
    model = Model(weights=np.array([1.0]), bias=0.0)
    act = config.activation_fn
    # At z = 0 the activation returns its midpoint, so the prediction must flip there.
    assert model.predict(np.array([[0.001]]), act)[0] == 1
    assert model.predict(np.array([[-0.001]]), act)[0] == 0


# -- plaintext trainer ----------------------------------------------------


def test_plaintext_training_learns(smoke_split) -> None:
    config = ModelConfig(n_features=smoke_split.x_train.shape[1], epochs=8, batch_size=14)
    result = train_plaintext(smoke_split, config)
    assert result.status if hasattr(result, "status") else True
    assert result.final_test_accuracy > 0.5
    assert result.epochs[-1].train_loss < result.epochs[0].train_loss


def test_plaintext_training_is_deterministic(smoke_split) -> None:
    config = ModelConfig(n_features=smoke_split.x_train.shape[1], epochs=3, batch_size=14)
    a = train_plaintext(smoke_split, config)
    b = train_plaintext(smoke_split, config)
    assert np.array_equal(a.model.weights, b.model.weights)


def test_loss_and_accuracy_helpers() -> None:
    assert squared_loss(np.array([0.5, 0.5]), np.array([0.0, 1.0])) == pytest.approx(0.25)
    assert accuracy(np.array([1, 0, 1]), np.array([1.0, 0.0, 0.0])) == pytest.approx(2 / 3)
