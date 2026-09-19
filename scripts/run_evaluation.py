import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, OUTPUT_DIR  # noqa: E402
from src.data import feature_sets, load_dataset  # noqa: E402
from src.evaluation import evaluate_model, summarize_fold_metrics  # noqa: E402
from src.models import baseline_model_builders, build_hsl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run final five-fold model evaluation")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data and configuration without training models",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_dataset(DATA_PATH)
    sets = feature_sets(data)
    print(f"Loaded {len(data)} rows and {data.shape[1] - 1} predictors")
    print("Feature-set sizes:", {name: len(columns) for name, columns in sets.items()})

    if args.validate_only:
        return

    results = []
    for feature_set_name in ("A", "AE", "AH", "AEH"):
        for model_name, model_builder in baseline_model_builders().items():
            print(f"Evaluating {model_name} on Set {feature_set_name}")
            results.append(
                evaluate_model(
                    data,
                    feature_set_name,
                    model_name,
                    model_builder,
                )
            )

    print("Evaluating CHSL on Set AEH")
    results.append(
        evaluate_model(
            data,
            feature_set_name="AEH",
            model_name="CHSL",
            model_builder=build_hsl,
            use_ctgan=True,
        )
    )

    fold_metrics = pd.concat(results, ignore_index=True)
    summary = summarize_fold_metrics(fold_metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "mean_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

