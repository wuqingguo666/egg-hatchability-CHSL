from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import TARGET_COLUMN


ANIMAL_CONTINUOUS = {
    "BreederAge",
    "AvgEggWeight",
    "EggWeightUniformity",
}
ENVIRONMENT_FEATURES = {
    "DaysAtTransfer",
    "EggWeightLoss",
    "EggStorageDuration",
    "Month_Sin",
    "Month_Cos",
}
HATCHERY_CONTINUOUS = {"SetEggs"}


@dataclass
class FoldData:
    feature_names: list[str]
    train_index: np.ndarray
    test_index: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler


def load_dataset(path) -> pd.DataFrame:
    data = pd.read_csv(path)
    validate_dataset(data)
    return data


def validate_dataset(data: pd.DataFrame) -> None:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    if data.shape != (1737, 125):
        raise ValueError(f"Expected shape (1737, 125), found {data.shape}")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names found")
    if data.isna().any().any():
        raise ValueError("Missing values found")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise TypeError("All cleaned columns must be numeric")


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]
    animal = {
        column
        for column in feature_columns
        if column in ANIMAL_CONTINUOUS
        or column in {"GPS", "PS"}
        or column.startswith("Line")
        or column.startswith("Farm")
    }
    environment = ENVIRONMENT_FEATURES
    hatchery = {
        column
        for column in feature_columns
        if column in HATCHERY_CONTINUOUS
        or column.startswith("Setter")
        or column.startswith("Hatcher")
    }

    classified = animal | environment | hatchery
    unclassified = set(feature_columns) - classified
    if unclassified:
        raise ValueError(f"Unclassified feature columns: {sorted(unclassified)}")

    def ordered(columns: set[str]) -> list[str]:
        return [column for column in feature_columns if column in columns]

    return {
        "A": ordered(animal),
        "AE": ordered(animal | environment),
        "AH": ordered(animal | hatchery),
        "AEH": feature_columns,
    }


def discrete_columns(data: pd.DataFrame) -> list[str]:
    continuous = ANIMAL_CONTINUOUS | ENVIRONMENT_FEATURES | HATCHERY_CONTINUOUS
    return [
        column
        for column in data.columns
        if column != TARGET_COLUMN and column not in continuous
    ]


def prepare_fold(
    data: pd.DataFrame,
    columns: list[str],
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> FoldData:
    """Fit both scalers on the current outer training fold only."""
    x_train_raw = data.iloc[train_index][columns]
    x_test_raw = data.iloc[test_index][columns]
    y_train_raw = data.iloc[train_index][[TARGET_COLUMN]]
    y_test = data.iloc[test_index][TARGET_COLUMN].to_numpy(dtype=float)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train_raw)
    x_test = x_scaler.transform(x_test_raw)
    y_train = y_scaler.fit_transform(y_train_raw).ravel()

    return FoldData(
        feature_names=columns,
        train_index=train_index,
        test_index=test_index,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )

