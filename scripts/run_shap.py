import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, OUTPUT_DIR  # noqa: E402
from src.ctgan_augmentation import (  # noqa: E402
    augment_scaled_training_fold,
    generate_fold_synthetic_data,
)
from src.data import feature_sets, load_dataset, prepare_fold  # noqa: E402
from src.evaluation import outer_cross_validator  # noqa: E402
from src.models import build_hsl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Explain out-of-fold CHSL predictions")
    parser.add_argument(
        "--background-size",
        type=int,
        default=None,
        help="Optional SHAP background subsample; default uses the full training fold",
    )
    parser.add_argument("--local-index", type=int, default=54)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_dataset(DATA_PATH)
    columns = feature_sets(data)["AEH"]
    n_rows, n_features = len(data), len(columns)
    shap_matrix = np.empty((n_rows, n_features), dtype=float)
    scaled_matrix = np.empty((n_rows, n_features), dtype=float)
    base_values = np.empty(n_rows, dtype=float)

    for fold_number, (train_index, test_index) in enumerate(
        outer_cross_validator().split(data), start=1
    ):
        print(f"Explaining CHSL outer fold {fold_number}/5")
        fold = prepare_fold(data, columns, train_index, test_index)
        synthetic = generate_fold_synthetic_data(data, train_index)
        x_train, y_train = augment_scaled_training_fold(fold, synthetic)
        model = build_hsl()
        model.fit(x_train, y_train)

        background = pd.DataFrame(fold.x_train, columns=columns)
        if args.background_size is not None and args.background_size < len(background):
            background = shap.sample(
                background,
                args.background_size,
                random_state=42,
            )

        def predict_hatchability(values):
            scaled_prediction = model.predict(np.asarray(values))
            return fold.y_scaler.inverse_transform(
                np.asarray(scaled_prediction).reshape(-1, 1)
            ).ravel()

        explainer = shap.Explainer(predict_hatchability, background)
        test_frame = pd.DataFrame(fold.x_test, columns=columns)
        fold_explanation = explainer(test_frame)
        shap_matrix[test_index] = fold_explanation.values
        scaled_matrix[test_index] = fold.x_test
        base_values[test_index] = np.asarray(fold_explanation.base_values).ravel()

    explanation = shap.Explanation(
        values=shap_matrix,
        base_values=base_values,
        data=scaled_matrix,
        feature_names=columns,
    )

    shap_dir = OUTPUT_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shap_dir / "oof_shap_values.npz",
        values=shap_matrix,
        base_values=base_values,
        data=scaled_matrix,
        feature_names=np.asarray(columns),
    )
    importance = pd.Series(
        np.abs(shap_matrix).mean(axis=0), index=columns, name="mean_abs_shap"
    ).sort_values(ascending=False)
    importance.to_csv(shap_dir / "feature_importance.csv")

    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.bar(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    if not 0 <= args.local_index < n_rows:
        raise IndexError(f"local-index must be between 0 and {n_rows - 1}")
    shap.plots.waterfall(explanation[args.local_index], max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "local_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

