"""Tests for lib.model_development — the ModelBuilder hierarchy and LightGBM."""

from __future__ import annotations

import lightgbm as lgbm
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from lib.model_development import (
    ClassifierBuilder,
    LightGbmModelBuilder,
    ModelBuilder,
)


TARGET = "default_flag"
FEATURES = [f"x{i}" for i in range(5)]
# Small, quiet model so the suite stays fast and log-free.
FAST_PARAMS = {"n_estimators": 20, "num_leaves": 7, "verbose": -1}


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture
def modelling_frames() -> dict[str, pd.DataFrame]:
    """Train / test / holdout frames with features, target and an unused column."""
    X, y = make_classification(
        n_samples=600,
        n_features=len(FEATURES),
        n_informative=3,
        n_redundant=0,
        weights=[0.78],  # roughly the Taiwan default rate
        random_state=0,
    )
    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET] = y
    df["ID"] = np.arange(len(df))  # present in the data but not a model input
    return {
        "train": df.iloc[:400].reset_index(drop=True),
        "test": df.iloc[400:500].reset_index(drop=True),
        "holdout": df.iloc[500:].reset_index(drop=True),
    }


@pytest.fixture
def builder(modelling_frames) -> LightGbmModelBuilder:
    """Builder constructed from full frames, without a holdout sample."""
    return LightGbmModelBuilder(
        input_train_data=modelling_frames["train"],
        input_test_data=modelling_frames["test"],
        independent_variables=FEATURES,
        target_variable=TARGET,
        model_parameters=FAST_PARAMS,
    )


@pytest.fixture
def builder_with_holdout(modelling_frames) -> LightGbmModelBuilder:
    return LightGbmModelBuilder(
        input_train_data=modelling_frames["train"],
        input_test_data=modelling_frames["test"],
        independent_variables=FEATURES,
        target_variable=TARGET,
        model_parameters=FAST_PARAMS,
        input_holdout_data=modelling_frames["holdout"],
    )


@pytest.fixture
def split_data(modelling_frames) -> dict[str, pd.DataFrame]:
    """Pre-split X / y frames, shaped like the CSVs notebook 02 writes out."""
    return {
        f"{kind}_{name}": frame[cols]
        for name, frame in modelling_frames.items()
        for kind, cols in [("X", FEATURES), ("y", [TARGET])]
    }


# ----------------------------------------------------------------------------
# Class hierarchy
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("abstract_cls", [ModelBuilder, ClassifierBuilder])
def test_abstract_builders_cannot_be_instantiated(abstract_cls, modelling_frames):
    with pytest.raises(TypeError, match="abstract"):
        abstract_cls(
            modelling_frames["train"],
            modelling_frames["test"],
            FEATURES,
            TARGET,
            FAST_PARAMS,
        )


def test_lightgbm_builder_is_a_classifier_builder():
    assert issubclass(LightGbmModelBuilder, ClassifierBuilder)
    assert issubclass(LightGbmModelBuilder, ModelBuilder)


# ----------------------------------------------------------------------------
# Construction and model parameters
# ----------------------------------------------------------------------------
def test_init_builds_unfitted_lgbm_classifier(builder):
    assert isinstance(builder.model, lgbm.LGBMClassifier)
    assert builder.trained_model is None
    assert not builder.model.__sklearn_is_fitted__()


def test_init_leaves_split_attributes_unset(builder):
    for attr in ["X_train", "y_train", "X_test", "y_test", "X_holdout", "y_holdout"]:
        assert getattr(builder, attr) is None


def test_model_parameters_are_passed_to_lgbm(builder):
    params = builder.model.get_params()
    assert params["n_estimators"] == FAST_PARAMS["n_estimators"]
    assert params["num_leaves"] == FAST_PARAMS["num_leaves"]


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_random_state_is_injected_into_model(modelling_frames, seed):
    b = LightGbmModelBuilder(
        modelling_frames["train"],
        modelling_frames["test"],
        FEATURES,
        TARGET,
        FAST_PARAMS,
        random_state=seed,
    )
    assert b.model_parameters["random_state"] == seed
    assert b.model.get_params()["random_state"] == seed


def test_default_random_state_is_42(builder):
    assert builder.model.get_params()["random_state"] == 42


def test_random_state_argument_overrides_params_dict(modelling_frames):
    b = LightGbmModelBuilder(
        modelling_frames["train"],
        modelling_frames["test"],
        FEATURES,
        TARGET,
        {**FAST_PARAMS, "random_state": 1},
        random_state=99,
    )
    assert b.model.get_params()["random_state"] == 99


def test_callers_params_dict_is_not_mutated(modelling_frames, split_data):
    params = dict(FAST_PARAMS)
    LightGbmModelBuilder(
        modelling_frames["train"],
        modelling_frames["test"],
        FEATURES,
        TARGET,
        params,
        random_state=7,
    )
    LightGbmModelBuilder.from_split_data(
        split_data["X_train"],
        split_data["y_train"],
        split_data["X_test"],
        split_data["y_test"],
        params,
        random_state=8,
    )
    assert params == FAST_PARAMS


# ----------------------------------------------------------------------------
# split_data_by_independent_and_target_variables
# ----------------------------------------------------------------------------
def test_split_returns_self(builder):
    assert builder.split_data_by_independent_and_target_variables() is builder


def test_split_selects_only_independent_variables(builder):
    builder.split_data_by_independent_and_target_variables()
    assert list(builder.X_train.columns) == FEATURES
    assert list(builder.X_test.columns) == FEATURES


def test_split_produces_1d_targets_matching_input(builder, modelling_frames):
    builder.split_data_by_independent_and_target_variables()
    for split in ["train", "test"]:
        y = getattr(builder, f"y_{split}")
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1
        np.testing.assert_array_equal(y, modelling_frames[split][TARGET].to_numpy())


def test_split_preserves_row_counts(builder, modelling_frames):
    builder.split_data_by_independent_and_target_variables()
    assert (
        len(builder.X_train) == len(builder.y_train) == len(modelling_frames["train"])
    )
    assert len(builder.X_test) == len(builder.y_test) == len(modelling_frames["test"])


def test_split_copies_so_input_frames_are_not_modified(builder, modelling_frames):
    original = modelling_frames["train"].copy()
    builder.split_data_by_independent_and_target_variables()
    builder.X_train.iloc[:, 0] = -999.0
    pd.testing.assert_frame_equal(modelling_frames["train"], original)


def test_split_without_holdout_leaves_holdout_unset(builder):
    builder.split_data_by_independent_and_target_variables()
    assert builder.X_holdout is None
    assert builder.y_holdout is None


def test_split_with_holdout_populates_holdout(builder_with_holdout, modelling_frames):
    builder_with_holdout.split_data_by_independent_and_target_variables()
    assert list(builder_with_holdout.X_holdout.columns) == FEATURES
    np.testing.assert_array_equal(
        builder_with_holdout.y_holdout,
        modelling_frames["holdout"][TARGET].to_numpy(),
    )


# ----------------------------------------------------------------------------
# train / predict
# ----------------------------------------------------------------------------
def test_train_before_split_raises(builder):
    with pytest.raises(RuntimeError, match="Training data not set"):
        builder.train()


@pytest.mark.parametrize(
    "method", ["predict_train_set", "predict_test_set", "predict_holdout_set"]
)
def test_predict_before_train_raises(builder_with_holdout, method):
    builder_with_holdout.split_data_by_independent_and_target_variables()
    with pytest.raises(RuntimeError, match="Model not trained"):
        getattr(builder_with_holdout, method)()


def test_predict_holdout_without_holdout_data_raises(builder):
    builder.split_data_by_independent_and_target_variables().train()
    with pytest.raises(RuntimeError, match="Holdout data not provided"):
        builder.predict_holdout_set()


def test_train_fits_model_and_returns_self(builder):
    builder.split_data_by_independent_and_target_variables()
    assert builder.train() is builder
    assert builder.trained_model is builder.model
    assert builder.trained_model.__sklearn_is_fitted__()
    assert builder.trained_model.n_features_in_ == len(FEATURES)


def test_methods_chain(builder_with_holdout):
    result = (
        builder_with_holdout.split_data_by_independent_and_target_variables()
        .train()
        .predict_train_set()
        .predict_test_set()
        .predict_holdout_set()
    )
    assert result is builder_with_holdout


@pytest.mark.parametrize("split", ["train", "test", "holdout"])
def test_predictions_are_positive_class_probabilities(builder_with_holdout, split):
    b = builder_with_holdout
    b.split_data_by_independent_and_target_variables().train()
    getattr(b, f"predict_{split}_set")()

    predicted = getattr(b, f"{split}_predicted")
    X = getattr(b, f"X_{split}")
    assert predicted.shape == (len(X),)
    assert np.all((predicted >= 0) & (predicted <= 1))
    np.testing.assert_allclose(predicted, b.trained_model.predict_proba(X)[:, 1])


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
BASE_KEYS = {
    "model",
    "X_train",
    "y_train",
    "pred_train",
    "X_test",
    "y_test",
    "pred_test",
}
HOLDOUT_KEYS = {"X_holdout", "y_holdout", "pred_holdout"}


def test_run_without_holdout_returns_base_keys(builder):
    assert set(builder.run()) == BASE_KEYS


def test_run_with_holdout_returns_holdout_keys(builder_with_holdout):
    assert set(builder_with_holdout.run()) == BASE_KEYS | HOLDOUT_KEYS


def test_run_splits_data_automatically(builder):
    results = builder.run()
    assert list(results["X_train"].columns) == FEATURES
    assert len(results["pred_train"]) == len(results["y_train"])
    assert len(results["pred_test"]) == len(results["y_test"])


def test_run_results_reference_builder_state(builder_with_holdout):
    b = builder_with_holdout
    results = b.run()
    assert results["model"] is b.trained_model
    assert results["pred_test"] is b.test_predicted
    assert results["pred_holdout"] is b.holdout_predicted


def test_run_does_not_resplit_when_data_already_split(builder):
    builder.split_data_by_independent_and_target_variables()
    X_train = builder.X_train
    builder.run()
    assert builder.X_train is X_train


def test_run_is_reproducible_for_fixed_seed(modelling_frames):
    def fit(seed: int) -> np.ndarray:
        # Row / feature subsampling makes the seed matter.
        params = {**FAST_PARAMS, "subsample": 0.7, "subsample_freq": 1}
        return LightGbmModelBuilder(
            modelling_frames["train"],
            modelling_frames["test"],
            FEATURES,
            TARGET,
            params,
            random_state=seed,
        ).run()["pred_test"]

    np.testing.assert_array_equal(fit(1), fit(1))
    assert not np.array_equal(fit(1), fit(2))


def test_model_learns_signal(builder):
    """Sanity check: on informative synthetic data, test AUC beats chance well."""
    results = builder.run()
    assert roc_auc_score(results["y_test"], results["pred_test"]) > 0.75


# ----------------------------------------------------------------------------
# from_split_data
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "to_target",
    [
        pytest.param(lambda df: df, id="dataframe"),
        pytest.param(lambda df: df.iloc[:, 0], id="series"),
        pytest.param(lambda df: df.to_numpy(), id="2d-array"),
        pytest.param(lambda df: df.to_numpy().ravel(), id="1d-array"),
        pytest.param(lambda df: df.iloc[:, 0].tolist(), id="list"),
    ],
)
def test_from_split_data_flattens_targets(split_data, to_target):
    b = LightGbmModelBuilder.from_split_data(
        split_data["X_train"],
        to_target(split_data["y_train"]),
        split_data["X_test"],
        to_target(split_data["y_test"]),
        FAST_PARAMS,
    )
    assert b.y_train.ndim == 1
    assert b.y_test.ndim == 1
    np.testing.assert_array_equal(b.y_train, split_data["y_train"][TARGET])


def test_from_split_data_builds_model_with_seed(split_data):
    b = LightGbmModelBuilder.from_split_data(
        split_data["X_train"],
        split_data["y_train"],
        split_data["X_test"],
        split_data["y_test"],
        FAST_PARAMS,
        random_state=5,
    )
    assert isinstance(b.model, lgbm.LGBMClassifier)
    assert b.model.get_params()["random_state"] == 5
    assert b.trained_model is None


def test_from_split_data_without_holdout(split_data):
    b = LightGbmModelBuilder.from_split_data(
        split_data["X_train"],
        split_data["y_train"],
        split_data["X_test"],
        split_data["y_test"],
        FAST_PARAMS,
    )
    assert b.X_holdout is None
    assert b.y_holdout is None
    assert set(b.run()) == BASE_KEYS


def test_from_split_data_with_holdout(split_data):
    results = LightGbmModelBuilder.from_split_data(
        split_data["X_train"],
        split_data["y_train"],
        split_data["X_test"],
        split_data["y_test"],
        FAST_PARAMS,
        X_holdout=split_data["X_holdout"],
        y_holdout=split_data["y_holdout"],
    ).run()
    assert set(results) == BASE_KEYS | HOLDOUT_KEYS
    assert results["y_holdout"].ndim == 1
    assert len(results["pred_holdout"]) == len(split_data["X_holdout"])


def test_from_split_data_matches_constructor_path(modelling_frames, split_data):
    """Both entry points should produce the same model on the same data."""
    from_frames = LightGbmModelBuilder(
        modelling_frames["train"],
        modelling_frames["test"],
        FEATURES,
        TARGET,
        FAST_PARAMS,
    ).run()
    from_split = LightGbmModelBuilder.from_split_data(
        split_data["X_train"],
        split_data["y_train"],
        split_data["X_test"],
        split_data["y_test"],
        FAST_PARAMS,
    ).run()
    np.testing.assert_allclose(from_frames["pred_test"], from_split["pred_test"])
