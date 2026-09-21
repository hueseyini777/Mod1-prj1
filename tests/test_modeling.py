import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from house_pipeline.config import LEAKAGE_COLUMNS
from house_pipeline.modeling import (
    adjusted_r2,
    build_elastic_pipeline,
    drop_identifier,
    load_model,
    save_model,
    select_features_and_target,
    split_data,
    train_baseline,
    tune_elastic_net,
)


@pytest.fixture
def engineered_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 40
    grade = rng.integers(4, 11, size=n)
    last_known_change = rng.integers(1950, 2015, size=n)
    price = 10_000 * grade + 5_000_000 + rng.normal(0, 1, size=n)

    return pd.DataFrame(
        {
            "id": np.arange(n),
            "price": price,
            "sqft_price": price / 1500,
            "date": ["20140502T000000"] * n,
            "grade": grade,
            "last_known_change": last_known_change,
            "delta_lat": rng.random(n),
            "delta_long": rng.random(n),
        }
    )


def test_select_features_and_target_drops_leakage_columns(engineered_df):
    X, y = select_features_and_target(engineered_df)

    assert not any(col in X.columns for col in LEAKAGE_COLUMNS)
    pd.testing.assert_series_equal(y, engineered_df["price"], check_names=False)


def test_split_data_respects_test_size(engineered_df):
    X, y = select_features_and_target(engineered_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=0)

    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == round(len(X) * 0.25)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_adjusted_r2_is_near_perfect_for_a_noise_free_linear_fit():
    X = pd.DataFrame({"x": np.arange(50)})
    y = 3 * X["x"] + 7

    model = LinearRegression().fit(X, y)

    assert adjusted_r2(model, X, y) == pytest.approx(1.0, abs=1e-6)


def test_train_baseline_fits_on_selected_variables_only(engineered_df):
    X, y = select_features_and_target(engineered_df)
    X_train, _, y_train, _ = split_data(X, y)

    model = train_baseline(X_train, y_train, variables=["grade"])

    assert model.coef_.shape == (1,)


def test_drop_identifier_removes_the_id_column(engineered_df):
    X, _ = select_features_and_target(engineered_df)

    assert "id" in X.columns
    assert "id" not in drop_identifier(X).columns


def test_build_elastic_pipeline_has_the_expected_steps():
    pipeline = build_elastic_pipeline()

    assert list(pipeline.named_steps) == ["polynomial", "scaler", "model"]


def test_tune_elastic_net_returns_a_fitted_best_estimator(engineered_df):
    X, y = select_features_and_target(engineered_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=0)

    small_grid = {
        "model__regressor__alpha": [0.1],
        "model__regressor__l1_ratio": [0.5],
    }
    search = tune_elastic_net(X_train, y_train, param_grid=small_grid, cv=2)

    assert search.best_params_ == {
        "model__regressor__alpha": 0.1,
        "model__regressor__l1_ratio": 0.5,
    }
    # The fitted estimator can score the held-out set without error.
    search.best_estimator_.score(drop_identifier(X_test), y_test)


def test_save_and_load_model_round_trips_predictions(engineered_df, tmp_path):
    X, y = select_features_and_target(engineered_df)
    X_train, X_test, _, _ = split_data(X, y, test_size=0.3, random_state=0)

    pipeline = build_elastic_pipeline()
    pipeline.set_params(model__regressor__alpha=0.5, model__regressor__l1_ratio=0.5)
    pipeline.fit(drop_identifier(X_train), y.loc[X_train.index])

    model_path = tmp_path / "model.bin"
    save_model(pipeline, path=model_path)
    loaded = load_model(path=model_path)

    original_predictions = pipeline.predict(drop_identifier(X_test))
    loaded_predictions = loaded.predict(drop_identifier(X_test))
    np.testing.assert_allclose(original_predictions, loaded_predictions)
