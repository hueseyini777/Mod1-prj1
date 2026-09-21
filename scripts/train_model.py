"""CLI: build the dataset, tune the elastic net pipeline, and save the model.

Usage:
    uv run python scripts/train_model.py
"""

from house_pipeline.modeling import (
    save_model,
    select_features_and_target,
    split_data,
    tune_elastic_net,
)
from house_pipeline.pipeline import build_dataset


def main() -> None:
    dataset = build_dataset()
    X, y = select_features_and_target(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)

    search = tune_elastic_net(X_train, y_train)
    print(f"Best params: {search.best_params_}")

    model = search.best_estimator_
    test_r2 = model.score(X_test.drop(columns=["id"]), y_test)
    print(f"Test R^2: {test_r2:.3f}")

    save_model(model)
    print("Saved trained pipeline to model/model.bin")


if __name__ == "__main__":
    main()
