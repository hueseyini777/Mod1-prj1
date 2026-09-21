"""Model training, tuning, and persistence for the King County house price model."""

from pathlib import Path
from typing import Any

import pandas as pd
import skops.io as sio
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from house_pipeline.config import LEAKAGE_COLUMNS, MODEL_DIR, MODEL_PATH

DEFAULT_PARAM_GRID = {
    "model__regressor__alpha": [0.01, 0.1, 1],
    "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
}


def select_features_and_target(
    df: pd.DataFrame, leakage_columns: list[str] = LEAKAGE_COLUMNS
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned, feature-engineered dataset into `X` and `y`.

    Drops columns that would leak the target or add no predictive value.
    """
    feature_columns = [col for col in df.columns if col not in leakage_columns]
    return df[feature_columns], df["price"]


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def adjusted_r2(model: RegressorMixin, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute R^2 adjusted for the number of predictors used by `model`."""
    n_samples = X.shape[0]
    n_features = X.shape[1]
    r2 = model.score(X, y)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def train_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, variables: list[str]
) -> LinearRegression:
    """Fit a plain linear regression on a small, hand-picked set of columns."""
    model = LinearRegression()
    model.fit(X_train[variables], y_train)
    return model


def drop_identifier(X: pd.DataFrame) -> pd.DataFrame:
    """Drop the `id` column, which carries no predictive signal."""
    return X.drop(columns=["id"])


def build_elastic_pipeline() -> Pipeline:
    """Build the polynomial + scaling + regularized regression pipeline.

    Polynomial expansion and scaling are fit inside the pipeline so they can
    be cross-validated per fold without leaking information from validation
    rows. The target is standardized within each fold via
    `TransformedTargetRegressor`, then predictions are converted back to
    dollars automatically.
    """
    return Pipeline(
        [
            ("polynomial", PolynomialFeatures(2, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=ElasticNet(max_iter=5000, tol=1e-4, precompute=True),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )


def tune_elastic_net(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, list[Any]] | None = None,
    cv: int = 5,
) -> GridSearchCV:
    """Tune the elastic net pipeline with grid search and cross-validation.

    `X_train` should contain the original feature columns (without `id`),
    not an already-expanded polynomial matrix.
    """
    param_grid = param_grid or DEFAULT_PARAM_GRID
    search = GridSearchCV(
        build_elastic_pipeline(),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=1,
        error_score="raise",
    )
    search.fit(drop_identifier(X_train), y_train)
    return search


def save_model(model: RegressorMixin, path: Path = MODEL_PATH) -> None:
    """Persist a fitted model/pipeline to disk with skops."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        sio.dump(model, f_out)


def load_model(path: Path = MODEL_PATH) -> RegressorMixin:
    """Load a model/pipeline previously saved with `save_model`.

    Trusts only the scikit-learn/numpy/scipy types skops itself flags as
    untrusted, since the pipeline built by this module never contains
    arbitrary user code.
    """
    untrusted_types = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=untrusted_types)
