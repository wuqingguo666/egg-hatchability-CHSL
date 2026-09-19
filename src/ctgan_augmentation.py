import numpy as np
import pandas as pd
from ctgan import CTGAN

from src.config import LOW_HATCHABILITY_THRESHOLD, N_SYNTHETIC_SAMPLES, TARGET_COLUMN
from src.data import FoldData, discrete_columns


def generate_fold_synthetic_data(
    data: pd.DataFrame,
    train_index: np.ndarray,
    sample_count: int = N_SYNTHETIC_SAMPLES,
) -> pd.DataFrame:
    """Fit CTGAN exclusively on minority samples in one outer training fold."""
    training_fold = data.iloc[train_index]
    minority_training = training_fold[
        training_fold[TARGET_COLUMN] < LOW_HATCHABILITY_THRESHOLD
    ]
    if minority_training.empty:
        raise ValueError("The outer training fold has no low-hatchability samples")

    ctgan = CTGAN()
    ctgan.fit(minority_training, discrete_columns(data))
    synthetic = ctgan.sample(sample_count)
    return synthetic.loc[:, data.columns]


def augment_scaled_training_fold(
    fold: FoldData,
    synthetic: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform synthetic rows with scalers fitted on real training rows only."""
    synthetic_x = fold.x_scaler.transform(synthetic[fold.feature_names])
    synthetic_y = fold.y_scaler.transform(synthetic[[TARGET_COLUMN]]).ravel()
    return (
        np.vstack([fold.x_train, synthetic_x]),
        np.concatenate([fold.y_train, synthetic_y]),
    )

