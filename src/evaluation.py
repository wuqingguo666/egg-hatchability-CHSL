from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.config import (
    OUTER_N_SPLITS,
    OUTER_RANDOM_STATE,
    OUTER_SHUFFLE,
)
from src.ctgan_augmentation import (
    augment_scaled_training_fold,
    generate_fold_synthetic_data,
)
from src.data import feature_sets, prepare_fold


def outer_cross_validator() -> KFold:
    """External evaluation splitter; separate from StackingRegressor.cv."""
    return KFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=OUTER_SHUFFLE,
        random_state=OUTER_RANDOM_STATE,
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_model(
    data: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
    model_builder: Callable,
    use_ctgan: bool = False,
) -> pd.DataFrame:
    """Evaluate one model with the paper's outer five-fold procedure."""
    columns = feature_sets(data)[feature_set_name]
    records = []

    for fold_number, (train_index, test_index) in enumerate(
        outer_cross_validator().split(data), start=1
    ):
        fold = prepare_fold(data, columns, train_index, test_index)
        x_train, y_train = fold.x_train, fold.y_train
        synthetic_count = 0

        if use_ctgan:
            synthetic = generate_fold_synthetic_data(data, train_index)
            x_train, y_train = augment_scaled_training_fold(fold, synthetic)
            synthetic_count = len(synthetic)

        model = model_builder()
        model.fit(x_train, y_train)
        prediction_scaled = model.predict(fold.x_test)
        prediction = fold.y_scaler.inverse_transform(
            np.asarray(prediction_scaled).reshape(-1, 1)
        ).ravel()

        records.append(
            {
                "model": model_name,
                "feature_set": feature_set_name,
                "outer_fold": fold_number,
                "train_size": len(train_index),
                "test_size": len(test_index),
                "synthetic_count": synthetic_count,
                **regression_metrics(fold.y_test, prediction),
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["mse", "rmse", "mae", "mape", "r2"]
    return (
        fold_metrics.groupby(["model", "feature_set"], sort=False)[metric_columns]
        .mean()
        .reset_index()
    )

